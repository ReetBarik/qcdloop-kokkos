cmake_minimum_required(VERSION 3.12...3.31)

# Disable the use of RPATHS - we probably are not
# that interested in relocatable binaries and it
# causes a lot of trouble on macOS
SET(CMAKE_SKIP_RPATH TRUE)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

project(qcdloop-kokkos)
include(GNUInstallDirs)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(VERSION 1.0.0)


# Set default CXXFLAGS but allow for environment override
# c.f. https://cmake.org/cmake/help/v3.31/envvar/CXXFLAGS.html
if (NOT CMAKE_CXX_FLAGS)
  # c.f. https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER_ID.html
  if (${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -fvisibility-inlines-hidden -fmessage-length=0 -ftree-vectorize -fstack-protector-strong -O2 -pipe -fext-numeric-literals")
  else()
    set(CMAKE_CXX_FLAGS "-Wall -Wextra")
  endif()
else()
  if (${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
    # Ensure -fext-numeric-literals is in CMAKE_CXX_FLAGS
    string(FIND "${CMAKE_CXX_FLAGS}" "-fext-numeric-literals" _found_ext_numeric_literals)
    if (_found_ext_numeric_literals EQUAL -1)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fext-numeric-literals" CACHE STRING "Update environment CXXFLAGS" FORCE)
    endif()
  endif()
endif()

# TODO: QCDLoop is currently only well tested on GNU, so other compiler
# and linker defaults should be added as they become known
if (${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address ${CMAKE_CXX_FLAGS}" CACHE STRING "debug CXXFLAGS" FORCE)
  set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address" CACHE STRING "debug linker flags" FORCE)
endif()

include_directories(src/qcdloop src)

find_package(Kokkos REQUIRED)


add_executable(tadpoleGPU_test examples/tadpoleGPU_test.cc)
target_link_libraries(tadpoleGPU_test PUBLIC Kokkos::kokkos)
target_compile_options(tadpoleGPU_test PRIVATE -w)

add_executable(bubbleGPU_test examples/bubbleGPU_test.cc)
target_link_libraries(bubbleGPU_test PUBLIC Kokkos::kokkos)
target_compile_options(bubbleGPU_test PRIVATE -w)

add_executable(triangleGPU_test examples/triangleGPU_test.cc)
target_link_libraries(triangleGPU_test PUBLIC Kokkos::kokkos)
target_compile_options(triangleGPU_test PRIVATE -w)

add_executable(boxGPU_test examples/boxGPU_test.cc)
target_link_libraries(boxGPU_test PUBLIC Kokkos::kokkos)
target_compile_options(boxGPU_test PRIVATE -w)

# Direct-per-leaf-call variant of boxGPU_test: bypasses ql::BO()'s full
# dispatch for sections that statically know their integral. See
# examples/boxGPU_test_direct.cc for why this exists.
add_executable(boxGPU_test_direct examples/boxGPU_test_direct.cc)
target_link_libraries(boxGPU_test_direct PUBLIC Kokkos::kokkos)
target_compile_options(boxGPU_test_direct PRIVATE -w)

find_package(xpmath_kokkos REQUIRED)

foreach(be dd ff qf tf)
  add_executable(boxGPU_test_${be} examples/boxGPU_test_xp.cc)
  target_link_libraries(boxGPU_test_${be} PUBLIC xpmath_kokkos::xpmath_kokkos)
  target_compile_definitions(boxGPU_test_${be} PRIVATE XPMATH_BACKEND_${be})
  target_compile_options(boxGPU_test_${be} PRIVATE -w)
endforeach()

# quadmath.h is host-only. Kokkos sends every C++ source of a Kokkos target
# through nvcc_wrapper or hipcc, so this file is compiled by an explicit host
# g++ invocation and linked in as a finished object.
if(DEFINED ENV{DD_HOST_CXX} AND NOT "$ENV{DD_HOST_CXX}" STREQUAL "")
  set(_dd_host_cxx "$ENV{DD_HOST_CXX}")
else()
  set(_dd_host_cxx "${CMAKE_CXX_COMPILER}")
  if(_dd_host_cxx MATCHES "hipcc|nvcc")
    execute_process(
      COMMAND which g++
      OUTPUT_VARIABLE _dd_host_cxx
      OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY
    )
  endif()
endif()
set(QUAD_INPUTS_OBJ "${CMAKE_CURRENT_BINARY_DIR}/quad_inputs.o")
add_custom_command(
  OUTPUT "${QUAD_INPUTS_OBJ}"
  COMMAND "${_dd_host_cxx}" -c
          "${CMAKE_CURRENT_SOURCE_DIR}/examples/quad_inputs.cc"
          -O2 -fext-numeric-literals
          -o "${QUAD_INPUTS_OBJ}"
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/quad_inputs.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/quad_inputs.h"
  COMMENT "Compiling quad inputs with host g++"
  VERBATIM
)
set_source_files_properties("${QUAD_INPUTS_OBJ}" PROPERTIES
  EXTERNAL_OBJECT TRUE
  GENERATED TRUE
)
foreach(be dd ff qf tf)
  target_sources(boxGPU_test_${be} PRIVATE "${QUAD_INPUTS_OBJ}")
endforeach()
foreach(tgt tadpoleGPU_test bubbleGPU_test triangleGPU_test boxGPU_test boxGPU_test_direct)
  target_sources(${tgt} PRIVATE "${QUAD_INPUTS_OBJ}")
endforeach()
find_library(QUADMATH_LIBRARY
  NAMES libquadmath.so quadmath
  PATHS /soft/compilers/gcc/13.3.0/x86_64-suse-linux/lib64
  NO_DEFAULT_PATH
)
if(NOT QUADMATH_LIBRARY)
  set(QUADMATH_LIBRARY quadmath)
endif()
foreach(be dd ff qf tf)
  target_link_libraries(boxGPU_test_${be} PUBLIC "${QUADMATH_LIBRARY}")
endforeach()
foreach(tgt tadpoleGPU_test bubbleGPU_test triangleGPU_test boxGPU_test boxGPU_test_direct)
  target_link_libraries(${tgt} PUBLIC "${QUADMATH_LIBRARY}")
endforeach()

