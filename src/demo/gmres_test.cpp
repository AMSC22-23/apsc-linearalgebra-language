#include <mpi.h>
#include <stdint.h>

#include <Eigen/Sparse>
#include <MPIContext.hpp>
#include <MPIMatrix.hpp>
#include <MPISparseMatrix.hpp>
#include <Matrix/Matrix.hpp>
#include <MatrixWithVecSupport.hpp>
#include <Parallel/Utilities/partitioner.hpp>
#include <Vector.hpp>
#include <csc.hpp>
#include <cspai.hpp>
#include <iostream>
#include <utils.hpp>

#include "assert.hpp"

#define DEBUG 0
#define LOAD_MATRIX_FROM_FILE 0
#define ACCEPT_ONLY_SQUARE_MATRIX 1

using EigenVectord = Eigen::VectorXd;

int main(int argc, char *argv[]) {
#if LOAD_MATRIX_FROM_FILE == 0
  if (argc < 2) {
    std::cerr << "Please specify the problem size as input argument"
              << std::endl;
    return 0;
  }
  const int size = atoi(argv[1]);
#endif
  MPI_Init(&argc, &argv);
  int mpi_rank;
  int mpi_size;
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  MPI_Comm_rank(mpi_comm, &mpi_rank);
  MPI_Comm_size(mpi_comm, &mpi_size);

  Eigen::SparseMatrix<double, Eigen::ColMajor> A;
#if LOAD_MATRIX_FROM_FILE == 0
  if (mpi_rank == 0) {
    A.resize(size, size);
    for (int i = 0; i < size; i++) {
      A.insert(i, i) = 2.0;
      if (i > 0) {
        A.insert(i, i - 1) = -1.0;
      }
      if (i < size - 1) {
        A.insert(i, i + 1) = -1.0;
      }
    }
  }
#else
  if (argc < 2) {
    if (mpi_rank == 0) {
      std::cerr << "Input matrix not provided, terminating" << std::endl;
    }
    MPI_Finalize();
    return 0;
  }
  apsc::LinearAlgebra::Utils::EigenUtils::load_sparse_matrix<decltype(A),
                                                             double>(argv[1],
                                                                     A);
#if ACCEPT_ONLY_SQUARE_MATRIX == 1
  ASSERT(A.rows() == A.cols(),
         "The provided matrix is not square" << std::endl);
#endif
#endif
  std::cout << "Launching GMRES with a sparse MPI matrix with size: "
            << A.rows() << "x" << A.cols() << ", non zero: " << A.nonZeros()
            << std::endl;

  A.makeCompressed();

  // Maintain whole vectors in each processess
  EigenVectord e;
  EigenVectord b(A.rows());
  if (!mpi_rank) {
    e.resize(A.rows());
    e.fill(1.0);
    b = A * e;
#if DEBUG == 1
    std::cout << "e vector:" << std::endl << e << std::endl;
    std::cout << "b vector:" << std::endl << b << std::endl;
#endif
  }
  // Initialise processes b vector
  MPI_Bcast(b.data(), b.size(), MPI_DOUBLE, 0, mpi_comm);

#if (DEBUG == 1)
  apsc::LinearAlgebra::Utils::MPI_matrix_show(PA, A, mpi_rank, mpi_size,
                                              mpi_comm);
#endif

  int r;
  // Test preconditioned GMRES with SPAI preconditioner
  {
    if (mpi_rank == 0) {
      std::cout << "================================= GMRES with SPAI "
                   "preconditioner ================================="
                << std::endl;
    }
    EigenVectord x;
    EigenVectord y;
    // Algebra:
    // AMy = b
    // x = My
    CSC<double> CSC_A;
    CSC_A.map_external_buffer(A.outerIndexPtr(), A.valuePtr(),
                              A.innerIndexPtr(), A.rows(), A.cols(),
                              A.nonZeros());
    const CSC<double> M =
        LinearAlgebra::Preconditioners::ApproximateInverse::CSPAI<
            double, Eigen::MatrixXd, 0>(&CSC_A, 0.1, 30, 1);
    const auto eigen_M =
        EigenStructureMap<Eigen::SparseMatrix<double>, double,
                          decltype(M)>::create_map(size, size, M.countNonZero,
                                                   M.offset, M.flatRowIndex,
                                                   M.flatData)
            .structure();

    const Eigen::SparseMatrix<double> AM = A * eigen_M;

    // retrive y, the unpreconditioned solver is called as the preconditioner is
    // embedded in the input matrix
    r = apsc::LinearAlgebra::Utils::GMRES::solve_MPI<decltype(AM), decltype(b),
                                                     double, decltype(e), 0>(
        AM, b, y, e, GMRES_MAX_ITER(size),
        MPIContext(mpi_comm, mpi_rank, mpi_size),
        objective_context(0, mpi_size,
                          std::string("test_gmres_with_precon_matrix_AM") +
                              std::string("_MPISIZE") +
                              std::to_string(mpi_size) + ".log",
                          std::string(argv[1])));
    ASSERT(r == 0, "A*M*y = b solver failed, can not retrieve the solution x"
                       << std::endl);
    x = eigen_M * y;
    if (mpi_rank == 0) {
      // std::cout << "Solution vector:" << std::endl;
      // std::cout << "[";
      // for (int i=0; i<x.size(); i++) {
      //   std::cout << x[i] << ",";
      // }
      // std::cout << std::endl << "[" << std::endl;
      std::cout << "Error norm:                                "
                << (x - e).norm() << endl;
    }
  }

  // Test no preconditioned GMRES
  {
    if (mpi_rank == 0) {
      std::cout << "================================= GMRES with no "
                   "preconditioner ================================="
                << std::endl;
    }
    EigenVectord x;
    r = apsc::LinearAlgebra::Utils::GMRES::solve_MPI<decltype(A), decltype(b),
                                                     double, decltype(e)>(
        A, b, x, e, GMRES_MAX_ITER(size),
        MPIContext(mpi_comm, mpi_rank, mpi_size),
        objective_context(0, mpi_size,
                          std::string("test_gmres_with_no_precon_matrix_A") +
                              std::string("_MPISIZE") +
                              std::to_string(mpi_size) + ".log",
                          std::string(argv[1])));
  }

  MPI_Finalize();
  return r;
}
