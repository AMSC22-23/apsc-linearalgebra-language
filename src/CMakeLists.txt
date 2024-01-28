cmake_minimum_required(VERSION 3.12.0)
# Enable this to compile with debug flags
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3")
project(apsc-Linear-Algebra-Language LANGUAGES CXX C)

include(./cmake_shared/cmake-common.cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find MPI package
find_package(MPI REQUIRED)

SET(EIGEN3_DIR_LOCAL $ENV{EIGEN3_INCLUDE_DIR})        #local installation
SET(EIGEN3_DIR_PCAFRICA $ENV{mkEigenInc})             #pcafrica/mk module

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/AMSC-CodeExamples/Examples/src/
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/ ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/algorithms/cg/
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/algorithms/gmres/
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/algorithms/bicg/
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/preconditioners/parallel/spai/
  ${EIGEN3_DIR_LOCAL} ${EIGEN3_DIR_PCAFRICA})
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

#Define executables
add_executable(full_matrix_cg demo/full_matrix_cg.cpp)

add_executable(MPI_full_matrix_cg demo/MPI_full_matrix_cg.cpp)
target_link_libraries(MPI_full_matrix_cg MPI::MPI_CXX)

add_executable(MPI_sparse_matrix_cg demo/MPI_sparse_matrix_cg.cpp)
target_link_libraries(MPI_sparse_matrix_cg MPI::MPI_CXX)

add_executable(MPI_sparse_matrix_gmres_spai demo/MPI_sparse_matrix_gmres_spai.cpp)
target_link_libraries(MPI_sparse_matrix_gmres_spai MPI::MPI_CXX)

add_executable(apsc_language demo/linear_algebra.cpp)
target_link_libraries(apsc_language MPI::MPI_CXX)
