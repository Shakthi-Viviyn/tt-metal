cmake_minimum_required(VERSION 3.16)
project(ttnn-standalone CXX)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(DEFINED ENV{CONFIG})
    message(STATUS "CONFIG is set, CMAKE_BUILD_TYPE being set to $ENV{CONFIG}")
    set(CMAKE_BUILD_TYPE $ENV{CONFIG})
elseif(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Release build is the default" FORCE)
endif()

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(WARNING "Not using the Clang compiler!")
else()
  message(STATUS "Found Clang-${CLANG_VERSION_MAJOR} here: ${CMAKE_CXX_COMPILER}")
  if(CLANG_VERSION_MAJOR LESS 17)
    message(WARNING "Clang version is less than 17!")
  endif()
endif()

# Add local ttnn-install to CMAKE_PREFIX_PATH if it exists
set(TTNN_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ttnn-install")
if(EXISTS ${TTNN_INSTALL_DIR})
    list(APPEND CMAKE_PREFIX_PATH ${TTNN_INSTALL_DIR})
endif()

# Find TT-NN package (required for standalone mode)
find_package(xtensor REQUIRED)  # TODO: remove once fixed in metal
find_package(TT-NN REQUIRED)

# All dependencies are handled by the TT-NN package

#       _                     _         _
#  ___ | |_  __ _  _ __    __| |  __ _ | |  ___   _ __    ___
# / __|| __|/ _` || '_ \  / _` | / _` || | / _ \ | '_ \  / _ \
# \__ \| |_| (_| || | | || (_| || (_| || || (_) || | | ||  __/
# |___/ \__|\__,_||_| |_| \__,_| \__,_||_| \___/ |_| |_| \___|

add_executable(ttnn-standalone ttnn-standalone.cpp)
set_property(TARGET ttnn-standalone PROPERTY CXX_STANDARD 20)
set_property(TARGET ttnn-standalone PROPERTY EXCLUDE_FROM_ALL TRUE)

target_link_libraries(ttnn-standalone PRIVATE TTNN::TTNN)
target_include_directories(
    ttnn-standalone
    PRIVATE
        "$ENV{TT_METAL_RUNTIME_ROOT}/ttnn/cpp"
)

target_precompile_headers(ttnn-standalone PRIVATE ttnn-precompiled.hpp)

find_path(TURBOJPEG_INCLUDE_DIR turbojpeg.h HINTS ENV TURBOJPEG_INCLUDE_DIR)
find_library(TURBOJPEG_LIBRARY NAMES turbojpeg HINTS ENV TURBOJPEG_LIB_DIR)
set(TTNN_CODEGEN_BATCH_SIZE "1" CACHE STRING "Batch size compiled into the selected generated graph")
set(TTNN_CODEGEN_STANDALONE_CPP
    "${CMAKE_CURRENT_SOURCE_DIR}/ttnn-standalone.cpp"
    CACHE FILEPATH
    "Generated TTNN standalone C++ file to include in the benchmark")
get_filename_component(TTNN_CODEGEN_GRAPH_DIR "${TTNN_CODEGEN_STANDALONE_CPP}" DIRECTORY)

if(NOT TURBOJPEG_INCLUDE_DIR OR NOT TURBOJPEG_LIBRARY)
    message(FATAL_ERROR
        "TurboJPEG is required for ttnn-codegen-benchmark. "
        "Install libturbojpeg-dev or set TURBOJPEG_INCLUDE_DIR and TURBOJPEG_LIB_DIR.")
endif()

add_executable(ttnn-codegen-benchmark benchmark_codegen_e2e.cpp)
set_property(TARGET ttnn-codegen-benchmark PROPERTY CXX_STANDARD 20)
set_property(TARGET ttnn-codegen-benchmark PROPERTY EXCLUDE_FROM_ALL TRUE)
target_link_libraries(ttnn-codegen-benchmark PRIVATE TTNN::TTNN ${TURBOJPEG_LIBRARY})
target_include_directories(
    ttnn-codegen-benchmark
    PRIVATE
        ${TURBOJPEG_INCLUDE_DIR}
        ${TTNN_CODEGEN_GRAPH_DIR}
        "$ENV{TT_METAL_RUNTIME_ROOT}/ttnn/cpp"
)
target_compile_definitions(
    ttnn-codegen-benchmark
    PRIVATE
        TTNN_CODEGEN_BATCH_SIZE=${TTNN_CODEGEN_BATCH_SIZE}
        TTNN_CODEGEN_STANDALONE_CPP="${TTNN_CODEGEN_STANDALONE_CPP}"
        TTNN_CODEGEN_GRAPH_DIR="${TTNN_CODEGEN_GRAPH_DIR}"
)
target_precompile_headers(
    ttnn-codegen-benchmark
    PRIVATE
        "${TTNN_CODEGEN_GRAPH_DIR}/ttnn-precompiled.hpp"
)
