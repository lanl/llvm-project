#
# Copyright (c) 2020 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune/llvm project.  It is released under
# the LLVM license.
#

set(CMAKE_BUILD_TYPE Release)

# Note: Since we borrow LLVM functionality here we
# need to use LLVM_OPTIONAL_SOURCES here to avoid
# a error from the LLVM cmake infrastructure...
# FIXME: This is annoying...
set(LLVM_OPTIONAL_SOURCES
  complex.cpp
  matmult1.cpp
  normalize.cpp
  raytrace.cpp
  vecadd.cpp
)

get_kitsune_tapir_rt_flags(tapir_args)

foreach(src_file IN ITEMS ${LLVM_OPTIONAL_SOURCES})
  message(STATUS "  kokkos example: ${src_file}")
  get_filename_component(base ${src_file} NAME_WLE)
  string(TOLOWER ${base} ${src_file})

  add_kitsune_example(kokkos-${base}
      ${src_file}
  )

  install(TARGETS kokkos-${base} RUNTIME DESTINATION ${LLVM_BINARY_DIR}/kitsune/examples/kokkos)


  target_include_directories(kokkos-${base}
     PRIVATE ${KITSUNE_INCLUDE_DIR} ${Kokkos_INCLUDE_DIR})

  set_target_properties(kokkos-${base}
      PROPERTIES
        COMPILE_FLAGS "-frtti" #  -Rpass=\".*\""
        LINK_DIRECTORIES ${Kokkos_LIBRARY_DIR}
        LINK_LIBRARIES ${Kokkos_LINK_LIBS}
  )

  # Add a executable per enabled runtime target.
  foreach(rt IN ITEMS ${tapir_args})
    set(target_name kokkos-${base}.${rt})
    message(STATUS "    ${rt} example: ${target_name}")
    add_kitsune_example(${target_name}
      ${src_file}
      )

    install(TARGETS ${target_name} RUNTIME DESTINATION ${LLVM_BINARY_DIR}/kitsune/examples/kokkos)

  target_include_directories(${target_name}
     PRIVATE ${KITSUNE_INCLUDE_DIR})

    set_target_properties(${target_name}
      PROPERTIES
        COMPILE_FLAGS "-frtti -fkokkos -fkokkos-no-init -ftapir=${rt}"
        LINK_FLAGS "-ftapir=${rt}"
        LINK_DIRECTORIES ${Kokkos_LIBRARY_DIR}
        LINK_LIBRARIES ${Kokkos_LINK_LIBS}
      )
  endforeach()

endforeach()

configure_file(
  ${KITSUNE_SOURCE_DIR}/examples/kokkos/run-kokkos-examples.sh.in
  ${KITSUNE_BINARY_DIR}/run-kokkos-examples.sh
)
