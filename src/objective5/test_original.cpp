/*
 * test_original.cpp
 *
 *  Created on: Dec 25, 2023
 *      Author: Ying Zhang
 */

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>

using SpMat = Eigen::SparseMatrix<double>;
using SpVec = Eigen::VectorXd;

using std::cout;
using std::endl;

int main(int argc, char *argv[]) {
  // using namespace apsc::LinearAlgebra;

  std::cout << "Reading the spd matrix A..." << std::endl;
  SpMat A;
  // std::cout << "Current path is " << std::filesystem::current_path() << '\n';
  std::string path =
      "/home/jellyfish/shared-folder/matvetcg-chen-zhang-guo/src/objective5/"
      "mat.mtx";
  Eigen::loadMarket(A, path);
  // std::cout << "\nmatrix A:\n" << A;
  const unsigned size = A.rows();
  std::cout << "A has been loaded successfully" << std::endl;

  // Check A properties
  std::cout << "Matrix size:" << A.rows() << " X " << A.cols() << std::endl;
  std::cout << "Non zero entries:" << A.nonZeros() << std::endl;
  SpMat B = SpMat(A.transpose()) - A;  // Check symmetry
  std::cout << "Norm of skew-symmetric part: " << B.norm() << std::endl;

  // with Eigen CG
  const SpVec e = SpVec::Ones(size);
  SpVec b = A * e;
  SpVec x = SpVec::Zero(size);
  const double tol = 1.e-2;
  const int maxit = 1000;
  int result;

  Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper> cg;
  cg.setMaxIterations(maxit);
  cg.setTolerance(tol);
  x = cg.compute(A).solve(b);

  std::cout << "Eigen native CG" << std::endl;
  std::cout << "#iterations:       " << cg.iterations() << std::endl;
  std::cout << "relative residual: " << cg.error() << std::endl;
  std::cout << "effective error:   " << (x - e).norm() << std::endl;

  // with Eigen BiCGSTAB
  x = 0 * x;

  Eigen::BiCGSTAB<SpMat> bicgstab;
  bicgstab.setMaxIterations(maxit);
  bicgstab.setTolerance(tol);
  x = bicgstab.compute(A).solve(b);

  std::cout << "Eigen native BiCGSTAB" << std::endl;
  std::cout << "#iterations:       " << bicgstab.iterations() << std::endl;
  std::cout << "relative residual: " << bicgstab.error() << std::endl;
  std::cout << "effective error:   " << (x - e).norm() << std::endl;

  return 0;
}
