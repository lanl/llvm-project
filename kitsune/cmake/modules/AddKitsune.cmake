macro(add_kitsune_subdirectory name)
  add_llvm_subdirectory(KITSUNE TOOL ${name})
endmacro(add_kitsune_subdirectory)

macro(add_kitsune_library name)
  cmake_parse_arguments(ARG
    "SHARED"
    ""
    "ADDITIONAL_HEADERS"
    ${ARGN})
  set(srcs)

  if (srcs OR ARG_ADDITIONAL_HEADERS)
    set(srcs
      ADDITIONAL_HEADERS
      ${srcs}
      ${ARG_ADDITIONAL_HEADERS}) # It may contain unparsed unknown args.
      
  endif()

  if (ARG_SHARED)
    set(LIBTYPE SHARED)
  else()
    # llvm_add_library ignores BUILD_SHARED_LIBS if STATIC is explicitly set,
    # so we need to handle it here.
    if (BUILD_SHARED_LIBS)
      set(LIBTYPE SHARED OBJECT)
    else()
      set(LIBTYPE STATIC OBJECT)
    endif()
    set_property(GLOBAL APPEND PROPERTY KITSUNE_STATIC_LIBS ${name})
  endif()

  llvm_add_library(${name} ${LIBTYPE} ${ARG_UNPARSED_ARGUMENTS} ${srcs})

  if (TARGET ${name})
    target_link_libraries(${name} INTERFACE ${LLVM_COMMON_LIBS})

    if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
      set(export_to_kitsunetargets)
      if (${name} IN_LIST LLVM_DISTRIBUTION_COMPONENTS OR
          "kitsune-libraries" IN_LIST LLVM_DISTRIBUTION_COMPONENTS OR
          NOT LLVM_DISTRIBUTION_COMPONENTS)
        set(export_to_kitsunetargets EXPORT KitsuneTargets)
        set_property(GLOBAL PROPERTY KITSUNE_HAS_EXPORTS True)
      endif()

      install(TARGETS ${name}
        COMPONENT ${name}
        ${export_to_kitsunetargets}
        LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
        ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
        RUNTIME DESTINATION examples)

      if (NOT LLVM_ENABLE_IDE)
        add_llvm_install_targets(install-${name}
                                 DEPENDS ${name}
                                 COMPONENT ${name})
      endif()

      set_property(GLOBAL APPEND PROPERTY KITSUNE_LIBS ${name})
    endif()
    set_property(GLOBAL APPEND PROPERTY KITSUNE_EXPORTS ${name})
  else()
    # Add empty "phony" target
    add_custom_target(${name})
  endif()

  set_target_properties(${name} PROPERTIES FOLDER "Kitsune libraries")
endmacro(add_kitsune_library)

macro(add_kitsune_executable name)
  add_llvm_executable(${name} ${ARGN})
endmacro(add_kitsune_executable)

macro(add_cxx_llvmir_file name flags)
  get_filename_component(target ${name} NAME_WE)
  message(STATUS "added custom target: ${target}.ll")
  message(STATUS "   compile flags: ${flags}")
  add_custom_target(${target}.ll
	${CMAKE_CXX_COMPILER} ${flags} -S -emit-llvm ${name} -o ${target}.ll
	COMMENT "generating IR file: ${target}.ll..."
	DEPENDS ${name}
	)
  add_dependencies(${target}.ll clang)

endmacro(add_cxx_llvmir_file) 

macro(add_kitsune_example name)
  if ( NOT KITSUNE_BUILD_EXAMPLES )
    set(EXCLUDE_FROM_ALL ON)
  endif()
  
  set(CMAKE_C_COMPILER ${LLVM_BINARY_DIR}/bin/clang)
  set(CMAKE_CXX_COMPILER ${LLVM_BINARY_DIR}/bin/clang++)
  add_kitsune_executable(${name} 
                         ${ARGN}
                         DEPENDS clang)
endmacro(add_kitsune_example)

macro(add_kitsune_tool name)
  if (NOT KITSUNE_BUILD_TOOLS)
    set(EXCLUDE_FROM_ALL ON)
  endif()

  add_kitsune_executable(${name} ${ARGN})

  if (KITSUNE_BUILD_TOOLS)
    set(export_to_kitsunetargets)
    if (${name} IN_LIST LLVM_DISTRIBUTION_COMPONENTS OR
        NOT LLVM_DISTRIBUTION_COMPONENTS)
      set(export_to_kitsunetargets EXPORT KitsuneTargets)
      set_property(GLOBAL PROPERTY KITSUNE_HAS_EXPORTS True)
    endif()

    install(TARGETS ${name}
      ${export_to_kitsunetargets}
      RUNTIME DESTINATION bin
      COMPONENT ${name})

    if(NOT LLVM_ENABLE_IDE)
      add_llvm_install_targets(install-${name}
                               DEPENDS ${name}
                               COMPONENT ${name})
    endif()
    set_property(GLOBAL APPEND PROPERTY KITSUNE_EXPORTS ${name})
  endif()
endmacro()

macro(add_kitsune_symlink name dest)
  add_llvm_tool_symlink(${name} ${dest} ALWAYS_GENERATE)
  # Always generate install targets
  llvm_install_symlink(${name} ${dest} ALWAYS_GENERATE)
endmacro()


macro(get_kitsune_tapir_rt_flags arglist)
  list(LENGTH arglist length)
  if (length GREATER 0)
    message(FATAL "get_kitsune_tapir_rt_flags() : non-empty list passed!")
  endif()

  # The list of possible runtime targets for the compiler (via tapir).
  # These are expanded to match the ENABLE options in the kitsune cmake
  # configuration (and thus we match capitalization here). 
  set(_kitsune_rt_names CILKRTS;OPENCILK;QTHREADS;REALM;OPENMP;CUDART;OPENCL)
  set(_kitsune_rt_flags cilk;opencilk;qthreads;realm;openmp;cuda;opencl)  
  foreach(rt IN ITEMS ${_kitsune_rt_names})
    set(_ENABLE_VAR "KITSUNE_ENABLE_${rt}_TARGET")
    if (${_ENABLE_VAR})
      list(FIND _kitsune_rt_names ${rt} rtindex)
      if (rtindex GREATER_EQUAL 0)
        list(GET _kitsune_rt_flags ${rtindex} flag)
        list(APPEND ${arglist} ${flag})
      endif()
    endif()  
  endforeach()
endmacro()
