# This module will set the following variables in your project:
#
#  OpenCilkRTS_FOUND        -- opencilk runtime was found.
#  OpenCilkRTS_INCLUDE_DIR  -- directory with opencilk runtime header file(s).
#  OpenCilkRTS_LIBRARY      -- full path to opencilk runtime library.
#  OpenCilkRTS_LIBRARY_DIR  -- path to where the opencilk runtime library lives.
#  OpenCilkRTS_LINK_LIBS    -- set of link libraries (e.g. -lopencilk)
#

message(STATUS "Looking for OpenCilkRTS...")

find_path(OpenCilkRTS_INCLUDE_DIR cilk/cilk.h
  PATHS /usr/local/include
        /opt/include
        /opt/local/include
        $ENV{OpenCilkRTS_PATH}/include
)

find_library(OpenCilkRTS_LIBRARY opencilk
  PATHS /usr/local/lib64
        /usr/local/lib
        /opt/lib64
        /opt/lib
        /opt/local/lib64
        /opt/local/lib
        $ENV{OpenCilkRTS_PATH}/lib
        $ENV{OpenCilkRTS_PATH}/lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenCilkRTS DEFAULT_MSG
				  OpenCilkRTS_INCLUDE_DIR
				  OpenCilkRTS_LIBRARY)


if (OpenCilkRTS_FOUND)
  message(STATUS "Found OpenCilk runtime system.")
  get_filename_component(OpenCilkRTS_LIBRARY_DIR
                         ${OpenCilkRTS_LIBRARY}
                         DIRECTORY
                         CACHE)

  # TODO: what do we do for the C personality?
  set(OpenCilkRTS_LINK_LIBS "-lopencilk -lopencilk-personality-cpp"
      CACHE STRING "List of libraries needed for the OpenCilk runtime system.")

  message(STATUS "OpenCilk runtime include directory : ${OpenCilkRTS_INCLUDE_DIR}")
  message(STATUS "OpenCilk runtime library directory : ${OpenCilkRTS_LIBRARY_DIR}")
  message(STATUS "OpenCilk runtime link libraries    : ${OpenCilkRTS_LINK_LIBS}")
else()
  message(STATUS "OpenCilk rutnime NOT found.")
endif()
