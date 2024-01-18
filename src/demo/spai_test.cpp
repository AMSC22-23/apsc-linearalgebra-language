#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <mpi.h>
#include <set>
#include <spai/spai.hpp>
#include <stdlib.h>
#include "Eigen/src/Core/Matrix.h"
#include <spai/csc.hpp>
#include "spai/cspai.hpp"
#include "utils.hpp"

using SpMat=Eigen::SparseMatrix<double, Eigen::ColMajor>;
using SpVec=Eigen::VectorXd;


int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  int size = 40;
  double tol = 1e-3;
  std::size_t max_iter = 100;

  if (argc > 1) {
    size = atoi(argv[1]);
  }
  if (argc > 2) {
    tol = atof(argv[2]);
  }
  if (argc > 3) {
    max_iter = atoi(argv[3]);
  }
  if (mpi_rank == 0) {
    std::cout << "=========================== Problem size = " << size << " ===========================" <<  std::endl;
  }

  //cpp implementation
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
  auto I = Eigen::MatrixXd::Identity(size, size);
  {
    // // std::cout << A << std::endl;
    // Eigen::MatrixXd full_A;
    // full_A = Eigen::MatrixXd(A);
    // Eigen::EigenSolver<decltype(full_A)> eigensolver_A(full_A);
    // if (eigensolver_A.info() != Eigen::Success) abort();
    // // std::cout << "K(A) = " << cond(eigensolver_A) << std::endl;

    // LinearAlgebra::Preconditioner::SPAI<double, decltype(A), Eigen::MatrixXd, Eigen::VectorXd> M;
    // M.set_tollerance(tol);
    // M.set_max_inter(max_iter);
    // std::chrono::high_resolution_clock::time_point begin =
    //     std::chrono::high_resolution_clock::now();
    // M.init(A);
    // std::chrono::high_resolution_clock::time_point end =
    //     std::chrono::high_resolution_clock::now();
    // long long diff =
    //     std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
    //         .count();
    // std::cout << "Time spent: " << diff << " ms" << std::endl;
    // // std::cout << "Approximate inverse" << std::endl << M.get_m() << std::endl;
    // // std::cout << "Eigen A's inverse" << std::endl << full_A.inverse() << std::endl;
    // SpMat M_x_A = M.get_m() * A;
    // // std::cout << "M @ A:" << std::endl << M_x_A << std::endl;
    // apsc::LinearAlgebra::Utils::print_matlab_matrix(Eigen::SparseMatrix<double>(M_x_A), "M_x_A_cpp.txt"); //this method takes an object with coeffRef method available and the Product type of Eigen does not have it
    // std::cout << "Verifying (M * A) Frobenius norm: " << M_x_A.norm() << std::endl;
    // std::cout << "Verifying identity Frobenius norm: " << I.norm() << std::endl;
  }

  //C implementation
  {
    CSC<double> CSC_A;
    // CSC_A.create_diagonal(size, size, 2.0);
    CSC_A.map_external_buffer(A.outerIndexPtr(), A.valuePtr(), A.innerIndexPtr(), A.rows(), A.cols(), A.nonZeros());
    std::chrono::high_resolution_clock::time_point begin =
        std::chrono::high_resolution_clock::now();
    CSC<double> M = CSPAI<double, Eigen::MatrixXd, 1>(&CSC_A, tol, max_iter, 1);
    MPI_Barrier(MPI_COMM_WORLD);
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    long long diff =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
            .count();
    // M.print();
    // std::cout << "M non zero: " << M.countNonZero << std::endl;
    auto eigen_M = Eigen::Map<Eigen::SparseMatrix<double>>(size, size, M.countNonZero, M.offset, M.flatRowIndex, M.flatData);
    if (mpi_rank == 0) {
      std::cout << eigen_M << std::endl;
    }
    if (mpi_rank == 0) {
      std::cout << "Time spent: " << diff << " ms" << std::endl;
    }
    auto& M_x_A = eigen_M * A;
    // std::cout << "M x A =" << std::endl << M_x_A << std::endl;
    apsc::LinearAlgebra::Utils::print_matlab_matrix(Eigen::SparseMatrix<double>(M_x_A), "M_x_A_c.txt"); //this method takes an object with coeffRef method available and the Product type of Eigen does not have it
    
    // std::cout << "Identity size = " << I.rows() << " x " << I.cols() << std::endl;
    // std::cout << "M_x_A size = " << M_x_A.rows() << " x " << M_x_A.cols() << std::endl;
    if (mpi_rank == 0) {
      std::cout << "Verifying (M * A) Frobenius norm: " << M_x_A.norm() << std::endl;
      std::cout << "Verifying identity Frobenius norm: " << I.norm() << std::endl;
    }

    CSC_A.destoy();
    M.destoy();
  }

  MPI_Finalize();
  return 0;
}
