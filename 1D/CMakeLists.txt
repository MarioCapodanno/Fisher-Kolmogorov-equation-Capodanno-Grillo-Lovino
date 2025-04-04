cmake_minimum_required(VERSION 3.12.0)
project(lab06 LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(main_1D src/main_1D.cpp src/FisherKolmogorov1D.cpp)
deal_ii_setup_target(main_1D)
