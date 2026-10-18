cmake_minimum_required(VERSION 3.16)

project(huffman
    VERSION 2.0.0
    DESCRIPTION "Canonical Huffman codec and throughput benchmark"
    LANGUAGES CXX)

# A benchmark is only meaningful when it is optimised, so default to Release
# instead of an implicit (and slow) unoptimised build.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

option(HUFFMAN_BUILD_TESTS "Build the test suite" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(HUFFMAN_BUILD_BENCH_TOOLS "Build the measurement helpers from bench/" ON)

# Warnings are applied per target instead of globally, so that consumers of the
# library are not forced to inherit them.
function(huffman_enable_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
    endif()
endfunction()

add_library(huffman_core STATIC src/huffman.cpp)
add_library(huffman::core ALIAS huffman_core)
target_include_directories(huffman_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(huffman_core PUBLIC cxx_std_20)
huffman_enable_warnings(huffman_core)

add_executable(huffman src/main.cpp)
target_link_libraries(huffman PRIVATE huffman::core)
huffman_enable_warnings(huffman)

if(HUFFMAN_BUILD_TESTS)
    enable_testing()

    add_executable(huffman_tests tests/test_huffman.cpp)
    target_link_libraries(huffman_tests PRIVATE huffman::core)
    huffman_enable_warnings(huffman_tests)
    # Keep the internal assertions active even in a release build: the suite is
    # there to catch exactly what they check.
    if(MSVC)
        target_compile_options(huffman_tests PRIVATE /UNDEBUG)
    else()
        target_compile_options(huffman_tests PRIVATE -UNDEBUG)
    endif()

    add_test(NAME huffman_tests COMMAND huffman_tests)
endif()

if(HUFFMAN_BUILD_BENCH_TOOLS)
    # Measurement helper for `bench/measure.ps1`; not part of the library.
    add_executable(huffman_freq_probe bench/freq_probe.cpp)
    target_compile_features(huffman_freq_probe PRIVATE cxx_std_20)
    huffman_enable_warnings(huffman_freq_probe)
endif()
