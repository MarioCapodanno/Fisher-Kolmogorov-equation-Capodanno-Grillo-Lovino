cmake_minimum_required(VERSION 3.12.0)
project(lab08 LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(main_3D src/main_3D.cpp src/FisherKolmogorov3D.cpp)
deal_ii_setup_target(main_3D)


# --- Default parameter file -----------------------------------------------
set(PARAM_FILE "${CMAKE_BINARY_DIR}/parameters.prm")

file(WRITE "${PARAM_FILE}" [=[
# Mesh & geometry parameters
subsection Mesh & geometry parameters
  set Degree = 1
end

# Physical constants
subsection Physical constants
  set Dext = 1.0
  set Daxn = 10.0
  set Alpha coefficient = 0.1
end

# Time stepping parameters
subsection Time stepping parameters
  set T = 1.0
  set deltat = 0.01
  set Theta = 1.0
end

# Solver parameters
subsection Solver parameters
  set Max Newton iterations = 1000
  set Newton tolerance = 1e-6
  set Max CG iterations = 1000
  set CG tolerance factor = 1e-6
end
]=])
# --------------------------------------------------------------------------
