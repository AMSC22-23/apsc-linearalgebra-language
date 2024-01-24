/*
 * test_mbase.cpp
 *
 *  Created on: Jan 1, 2024
 *      Author: Ying Zhang
 */

#include <chrono>
#include <cstddef>
#include <cstring>
#include <iostream>

#include "Matrix/Matrix.hpp"
#include "MatrixWithVecSupport.hpp"
#include "Vector.hpp"
#include "cg.hpp"
#include "spai_mbase.hpp"
// #include "cg_mpi.hpp"
#include "tools.hpp"

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
  const unsigned size = A.rows();
  std::cout << "A has been created successfully" << std::endl;

  // Check A properties
  std::cout << "Matrix size:" << A.rows() << " X " << A.cols() << std::endl;
  int nonZeros = 0;
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      if (A(i, j) != 0) {
        nonZeros += 1;
      }
    }
  }
  std::cout << "Non zero entries:" << nonZeros << std::endl;
  // Check symmetry
  MatrixWithVecSupport<double, std::vector<double>, ORDERING::ROWMAJOR> AT(n,
                                                                           n);
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      AT(i, j) = A(j, i);
    }
  }
  // std::cout << "Norm of skew-symmetric part: " << (AT-A).norm() << std::endl;

  // get M
  std::cout << "Creating the Matrix M(THE PRECONDITIONING OF A WITH SPARSE "
               "APPROXIMATE INVERSES)..."
            << std::endl;
  MatrixWithVecSupport<double, std::vector<double>, ORDERING::ROWMAJOR> M(size,
                                                                          size);
  int max_iter = 10;
  double epsilon = 0.6;
  auto start_time = std::chrono::high_resolution_clock::now();
  LinearAlgebra::SPAI_MBASE<decltype(A), decltype(epsilon)>(A, M, max_iter,
                                                            epsilon);
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  std::cout << "Time taken by SPAI_MBASE: " << duration.count()
            << " microseconds" << std::endl;
  std::cout << "matrix M:\n" << M << "\n";
  std::cout << "\nM * A:\n" << Tools::multiply_two_matrix(M, A) << std::endl;
  // std::cout << "(M*A-identityMatrix).norm() =  "<<
  // (M*A-identityMatrix).norm() << std::endl;

  // identityMatrix
  MatrixWithVecSupport<double, Vector<double>, ORDERING::ROWMAJOR>
      identityMatrix(size, size);
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      identityMatrix(i, j) = 0;
      if (i == j) {
        identityMatrix(i, j) = 1;
      }
    }
  }

  // with hand-made CG and identityMatrix
  MatrixWithVecSupport<double, Vector<double>, ORDERING::ROWMAJOR> AA(size,
                                                                      size);
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      AA(i, j) = A(i, j);
    }
  }
  Vector e(size, static_cast<double>(1));
  Vector b = AA * e;
  Vector x(size, static_cast<double>(0));
  double tol = 1.e-4;
  int result, maxit = 1000;
  result = LinearAlgebra::CG<decltype(AA), decltype(x),
                             decltype(identityMatrix), decltype(tol)>(
      AA, x, b, identityMatrix, maxit, tol);  // Solve system
  std::cout << "hand-made CG with identityMatrix: " << std::endl;
  std::cout << "CG flag = " << result << std::endl;
  std::cout << "maxit = 1000, iterations performed = " << maxit << std::endl;
  std::cout << "effective error =  " << (x - e).norm() << std::endl;

  // with hand-made CG and M
  MatrixWithVecSupport<double, Vector<double>, ORDERING::ROWMAJOR> MM(size,
                                                                      size);
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      MM(i, j) = M(i, j);
    }
  }
  x = x * 0.0;
  AA = Tools::multiply_two_matrix(MM, AA);
  Vector bb = MM * b;
  result =
      LinearAlgebra::CG<decltype(AA), decltype(x), decltype(MM), decltype(tol)>(
          AA, x, bb, identityMatrix, maxit, tol);  // Solve system
  std::cout << "hand-made CG with SPAI: " << std::endl;
  std::cout << "CG flag = " << result << std::endl;
  std::cout << "maxit = 1000, iterations performed = " << maxit << std::endl;
  std::cout << "effective error =  " << (x - e).norm() << std::endl;

  return 0;
}
