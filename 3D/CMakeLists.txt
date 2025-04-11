cmake_minimum_required(VERSION 3.12.0)
project(lab08 LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(main_3D src/main_3D.cpp src/FisherKolmogorov3D.cpp)
deal_ii_setup_target(main_3D)
