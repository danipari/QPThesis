# cmake_minimum_required(VERSION <specify CMake version here>)
project(Thesis)

set(CMAKE_CXX_STANDARD 14)
cmake_minimum_required(VERSION 3.17)

MATH(EXPR stack_size "16 * 1024 * 1024") # 16 Mb
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--stack,${stack_size}")

set(EIGEN_DIR "C:\\Users\\Daniel\\CLionProjects\\lib\\eigen")
if(NOT EXISTS ${EIGEN_DIR})
    message(FATAL_ERROR "Please check that the set Eigen directory is valid!")
endif()
include_directories(${EIGEN_DIR})

set(BOOST_DIR "C:\\Users\\Daniel\\CLionProjects\\lib\\boost_1_75_0")
if(NOT EXISTS ${BOOST_DIR})
    message(FATAL_ERROR "Could not find boost!")
endif()
include_directories(${BOOST_DIR})

set(EIGEN_UN_DIR "C:\\Users\\Daniel\\CLionProjects\\lib\\eigen\\unsupported")
if(NOT EXISTS ${EIGEN_UN_DIR})
    message(FATAL_ERROR "Could not find fftw!")
endif()
include_directories(${EIGEN_UN_DIR})

add_executable(Thesis main.cpp tools3bp.cpp Torus.cpp Torus.h Solver3BP.cpp Solver3BP.h interp.cpp interp.h QPCollocationSolver.cpp QPCollocationSolver.h tools3bp.h)
include_directories(Thesis "${CMAKE_CURRENT_SOURCE_DIR}/lib/Eigen")
include_directories(Thesis "${CMAKE_CURRENT_SOURCE_DIR}/lib/Boost")
include_directories(Thesis "${CMAKE_CURRENT_SOURCE_DIR}/lib/unsupported")