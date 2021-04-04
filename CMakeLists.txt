# cmake_minimum_required(VERSION <specify CMake version here>)
project(Thesis)

set(CMAKE_CXX_STANDARD 14)

if (CMAKE_BUILD_TYPE MATCHES Debug)

    cmake_minimum_required(VERSION 3.17)

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

    add_executable(Thesis main.cpp tools3bp.cpp Torus.cpp Torus.h Solver3BP.cpp Solver3BP.h interp.cpp interp.h QPCollocationSolver.cpp QPCollocationSolver.h tools3bp.h POCollocationSolver.cpp POCollocationSolver.h PeriodicOrbit.cpp PeriodicOrbit.h)
    include_directories(Thesis "${CMAKE_CURRENT_SOURCE_DIR}/lib/Eigen")
    include_directories(Thesis "${CMAKE_CURRENT_SOURCE_DIR}/lib/Boost")
    include_directories(Thesis "${CMAKE_CURRENT_SOURCE_DIR}/lib/unsupported")

endif()

## LINUX BUILD CONFIG
#if (CMAKE_BUILD_TYPE MATCHES Debug)
#
#    cmake_minimum_required(VERSION 3.16.3)
#
#    set(EIGEN_DIR "/mnt/c/Users/Daniel/CLionProjects/lib/eigen")
#    if(NOT EXISTS ${EIGEN_DIR})
#        message(FATAL_ERROR "Please check that the set Eigen directory is valid!")
#    endif()
#    include_directories(${EIGEN_DIR})
#
#    set(BOOST_DIR "/mnt/c/Users/Daniel/CLionProjects/lib/boost_1_75_0")
#    if(NOT EXISTS ${BOOST_DIR})
#        message(FATAL_ERROR "Could not find boost!")
#    endif()
#    include_directories(${BOOST_DIR})
#
#    set(EIGEN_UN_DIR "/mnt/c/Users/Daniel/CLionProjects/lib/eigen/unsupported")
#    if(NOT EXISTS ${EIGEN_UN_DIR})
#        message(FATAL_ERROR "Could not find fftw!")
#    endif()
#    include_directories(${EIGEN_UN_DIR})
#
#    add_executable(Thesis main.cpp tools3bp.cpp Torus.cpp Torus.h Solver3BP.cpp Solver3BP.h interp.cpp interp.h QPCollocationSolver.cpp QPCollocationSolver.h tools3bp.h)
#    include_directories(Thesis "${CMAKE_CURRENT_SOURCE_DIR}/lib/Eigen")
#    include_directories(Thesis "${CMAKE_CURRENT_SOURCE_DIR}/lib/Boost")
#    include_directories(Thesis "${CMAKE_CURRENT_SOURCE_DIR}/lib/unsupported")
#
#endif()
