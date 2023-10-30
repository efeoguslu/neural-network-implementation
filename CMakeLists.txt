cmake_minimum_required(VERSION 3.26)
project(neural_network_implementation)

set(CMAKE_CXX_STANDARD 14)

include_directories(cmake-build-debug)
include_directories(datasets)
include_directories(mainProject)
include_directories(tests)

add_executable(neural_network_implementation
        cmake-build-debug/Testing/Temporary/LastTest.log
        cmake-build-debug/build.ninja
        cmake-build-debug/cmake_install.cmake
        cmake-build-debug/CMakeCache.txt
        cmake-build-debug/neural_network_implementation
        mainProject/activation.c
        mainProject/activation.h
        mainProject/helpers.c
        mainProject/helpers.h
        tests/multiplier.c
        tests/orGate.c
        main.c)
