#include <mpi.h>
#include <stdlib.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <csc.hpp>
#include <iostream>
#include <set>
#include <spai.hpp>
#include <utils.hpp>

#include "EigenStructureMap.hpp"

using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using SpVec = Eigen::VectorXd;

int main(int argc, char* argv[]) {
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
    std::cout << "=========================== Problem size = " << size
              << " ===========================" << std::endl;
  }

  // cpp implementation
  SpMat A;
  A.resize(size, size);
  // Create tridiagonal
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
    apsc::LinearAlgebra::CSC<double> CSC_A;
    // CSC_A.create_diagonal(size, size, 2.0);
    CSC_A.map_external_buffer(A.outerIndexPtr(), A.valuePtr(),
                              A.innerIndexPtr(), A.rows(), A.cols(),
                              A.nonZeros());
    std::chrono::high_resolution_clock::time_point begin =
        std::chrono::high_resolution_clock::now();
    apsc::LinearAlgebra::Preconditioners::ApproximateInverse::SPAI<double,
                                                             Eigen::MatrixXd, 1>
        precond(&CSC_A, tol, max_iter, 1);
    MPI_Barrier(MPI_COMM_WORLD);
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    long long diff =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
            .count();
    const apsc::LinearAlgebra::CSC<double>& M = precond.get_M();
    // M.print();
    // std::cout << "M non zero: " << M.non_zeros << std::endl;
    auto eigen_M = M.to_eigen<Eigen::SparseMatrix<double>>(size);

    if (mpi_rank == 0) {
      std::cout << "Approximate inverse matrix: " << std::endl;
      std::cout << "[" << std::endl;
      for (int i = 0; i < eigen_M.rows(); ++i) {
        for (int j = 0; j < eigen_M.cols(); ++j) {
          std::cout << eigen_M.coeffRef(i, j);
          if (j < eigen_M.cols() - 1)
            std::cout << ",";
          else if (j == eigen_M.cols() - 1 && i < eigen_M.rows() - 1)
            std::cout << ";";
        }
      }
      std::cout << "]" << std::endl;
    }
    if (mpi_rank == 0) {
      std::cout << "Time spent: " << diff << " ms" << std::endl;
    }
    auto& M_x_A = eigen_M * A;
    // std::cout << "M x A =" << std::endl << M_x_A << std::endl;
    apsc::LinearAlgebra::Utils::print_matlab_matrix(
        Eigen::SparseMatrix<double>(M_x_A),
        "M_x_A_c.txt");  // this method takes an object with coeffRef method
                         // available and the Product type of Eigen does not
                         // have it

    // std::cout << "Identity size = " << I.rows() << " x " << I.cols() <<
    // std::endl; std::cout << "M_x_A size = " << M_x_A.rows() << " x " <<
    // M_x_A.cols() << std::endl;
    if (mpi_rank == 0) {
      std::cout << "Verifying (M * A) Frobenius norm: " << M_x_A.norm()
                << std::endl;
      std::cout << "Verifying identity Frobenius norm: " << I.norm()
                << std::endl;
    }

    CSC_A.destoy();
  }

  MPI_Finalize();
  return 0;
}
