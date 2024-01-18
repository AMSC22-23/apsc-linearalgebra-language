/*
 * test_tools.cpp
 *
 *  Created on: Jan 3, 2024
 *      Author: Ying Zhang
 */

#include <chrono>
#include <cstddef>
#include <iostream>

#include "MatrixWithVecSupport.hpp"
#include "tools.hpp"
#include "utils.hpp"

using std::cout;
using std::endl;

int main(int argc, char *argv[]) {
  using namespace apsc::LinearAlgebra;

  std::cout << "Creating a spd matrix A..." << std::endl;
  constexpr std::size_t n = 6;
  MatrixWithVecSupport<double, std::vector<double>, ORDERING::ROWMAJOR> A(n, n);
  // dense and random fill
  // A.fillRandom();
  // spd fill
  Utils::default_spd_fill<decltype(A), double>(A);
  std::cout << "matrix A:\n" << A;
  std::cout << "A has been created successfully" << std::endl;

  std::cout << "Creating a spd matrix B..." << std::endl;
  constexpr std::size_t m = 6;
  MatrixWithVecSupport<double, std::vector<double>, ORDERING::ROWMAJOR> B(m, m);
  // dense and random fill
  // B.fillRandom();
  // spd fill
  Utils::default_spd_fill<decltype(B), double>(B);
  std::cout << "matrix B:\n" << B;
  std::cout << "B has been created successfully" << std::endl;

  std::cout << "A * B = \n"
            << Tools::multiply_two_matrix<decltype(A)>(A, B) << std::endl;
  std::cout << "A - B = \n"
            << Tools::subtract_two_matrix<decltype(A)>(A, B) << std::endl;
  std::cout << "A + B = \n"
            << Tools::add_two_matrix<decltype(A)>(A, B) << std::endl;

  return 1;
}
