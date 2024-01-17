#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <mpi.h>
#include <set>
#include <spai.hpp>

using SpMat=Eigen::SparseMatrix<double, Eigen::ColMajor>;
using SpVec=Eigen::VectorXd;

constexpr int size = 32;

template<typename EigenSolver>
double cond(EigenSolver s) {
  std::set<double> eigenvalues;
  for (auto e : s.eigenvalues()) {
    eigenvalues.insert(e.real());
  }
  auto min_eigenvalue = eigenvalues.begin();
  auto max_eigenvalue = eigenvalues.end();
  max_eigenvalue--;
  return *max_eigenvalue / *min_eigenvalue;
}

int main(int argc, char *argv[]) {
  MPI_Init(NULL, NULL);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  SpMat A;
  A.resize(size, size);
  //Create tridiagonal
  for (int i = 0; i < size; i++) {
    A.insert(i, i) = 2.0;
    if (i > 0) {
      A.insert(i, i - 1) = -1.0;
    }
    if (i < size - 1) {
      A.insert(i, i + 1) = -1.0;
    }
  }
  A.makeCompressed();
  std::cout << A << std::endl;

  Eigen::MatrixXd full_A;
  full_A = Eigen::MatrixXd(A);
  Eigen::EigenSolver<decltype(full_A)> eigensolver_A(full_A);
  if (eigensolver_A.info() != Eigen::Success) abort();
  std::cout << "K(A) = " << cond(eigensolver_A) << std::endl;

  std::cout << "Creating the approximate inverse of A" << std::endl;
  LinearAlgebra::Preconditioner::SPAI<double, decltype(A), Eigen::MatrixXd, Eigen::VectorXd> M(A);
  M.set_tollerance(1e-10);
  M.set_max_inter(1000);
  std::cout << "Approximate inverse" << std::endl << M.get_m() << std::endl;
  std::cout << "Eigen A's inverse" << std::endl << full_A.inverse() << std::endl;
  SpMat M_x_A = M.get_m() * A;
  std::cout << "M @ A:" << std::endl << M_x_A << std::endl;
  std::cout << "Verifying (M * A) Frobenius norm: " << M_x_A.norm() << std::endl;
  auto I = Eigen::MatrixXd::Identity(size, size);
  std::cout << "Verifying identity Frobenius norm: " << I.norm() << std::endl;


  // std::cout << "[" << std::endl;
  // for (int i=0; i<M_x_A.rows(); ++i) {
  //   for (int j=0; j<M_x_A.cols(); ++j) {
  //     if (j == M_x_A.cols()-1 && i < M_x_A.rows()-1)
  //       std::cout << M_x_A.coeffRef(i, j) << ";" << std::endl;
  //     else if (i < M_x_A.rows()-1)
  //       std::cout << M_x_A.coeffRef(i, j) << ",";
  //   }
  // }
  // std::cout << "]" << std::endl;

  MPI_Finalize();
  return 0;
}
