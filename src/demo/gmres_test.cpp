#include <mpi.h>
#include <stdint.h>

#include <Eigen/Sparse>
#include <FullMatrix.hpp>
#include <MPIContext.hpp>
#include <MPIFullMatrix.hpp>
#include <MPISparseMatrix.hpp>
#include <Matrix/Matrix.hpp>
#include <Parallel/Utilities/partitioner.hpp>
#include <Vector.hpp>
#include <csc.hpp>
#include <iostream>
#include <spai.hpp>
#include <string>
#include <utils.hpp>

#include "assert.hpp"

#define DEBUG 0
#define LOAD_MATRIX_FROM_FILE 1
#define ACCEPT_ONLY_SQUARE_MATRIX 1

using EigenVectord = Eigen::VectorXd;

// Note, due to SPAI preconditioner setup requirements
// we have to store the SparseMatrix A on each processes.
// Further development can release this constraint by splitting
// the SPAI setup in a more parallel fashion.
int main(int argc, char* argv[]) {
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
  int global_size;
  // SPAI setup param
  double tol = 1e-3;
  std::size_t max_iter = 100;

  Eigen::SparseMatrix<double, Eigen::ColMajor> A;
#if LOAD_MATRIX_FROM_FILE == 0
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
  A.makeCompressed();
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

  if (argc > 2) {
    tol = atof(argv[2]);
  }
  if (argc > 3) {
    max_iter = atoi(argv[3]);
  }

  global_size = A.rows();
  // Enable this if memory optimisation are added in order to not store global
  // matrix on each process MPI_Bcast(&global_size, 1, MPI_INT, 0, mpi_comm);

  if (mpi_rank == 0) {
    std::cout << "Launching GMRES with a sparse MPI matrix with size: "
              << global_size << "x" << global_size
              << ", non zero: " << A.nonZeros() << std::endl;
  }

  // Maintain whole vectors in each processess
  EigenVectord e(global_size);
  EigenVectord b(global_size);
  e.fill(1.0);
  if (mpi_rank == 0) {
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

  // We have global A on each processes
  apsc::LinearAlgebra::CSC<double> CSC_A;
  CSC_A.map_external_buffer(A.outerIndexPtr(), A.valuePtr(), A.innerIndexPtr(),
                            A.rows(), A.cols(), A.nonZeros());

  apsc::LinearAlgebra::Preconditioners::ApproximateInverse::SPAI<
      double, Eigen::MatrixXd, 1>
      precond(&CSC_A, tol, max_iter, 1);
  auto& M = precond.get_M();
  const Eigen::Map<Eigen::SparseMatrix<double>> eigen_M =
      M.to_eigen<Eigen::SparseMatrix<double>>(global_size);

  Eigen::SparseMatrix<double> AM = A * eigen_M;

  // Testing sequential GMRES
  {
    if (mpi_rank == 0) {
      std::cout << "================================= SEQUENTIAL GMRES "
                   "================================="
                << std::endl
                << std::endl;
    }
    // Test preconditioned GMRES with SPAI preconditioner
    int r;
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

      // retrive y, the unpreconditioned solver is called as the preconditioner
      // is embedded in the input matrix
      r = apsc::LinearAlgebra::Utils::Solvers::GMRES::solve<
          decltype(AM), decltype(b), double, decltype(e), 0>(
          AM, b, y, e, GMRES_MAX_ITER(global_size),
          MPIContext(mpi_comm, mpi_rank, mpi_size),
          objective_context(
              0, mpi_size,
              std::string("sequential_gmres_with_precon_matrix_AM") +
                  std::string("_epsilon") + std::to_string(tol) +
                  std::string("_MPISIZE") + std::to_string(mpi_size) + ".log",
              std::string(argv[1])));
      ASSERT(r == 0, "A*M*y = b solver failed, can not retrieve the solution x"
                         << std::endl);
      // Recover initial unknown
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
      r = apsc::LinearAlgebra::Utils::Solvers::GMRES::solve<
          decltype(A), decltype(b), double, decltype(e)>(
          A, b, x, e, GMRES_MAX_ITER(global_size),
          MPIContext(mpi_comm, mpi_rank, mpi_size),
          objective_context(
              0, mpi_size,
              std::string("sequential_gmres_with_no_precon_matrix_A") +
                  std::string("_epsilon") + std::to_string(tol) +
                  std::string("_MPISIZE") + std::to_string(mpi_size) + ".log",
              std::string(argv[1])));
    }

    if (mpi_rank == 0) {
      std::cout << std::endl
                << std::endl
                << "================================= PARALLEL GMRES "
                   "================================="
                << std::endl
                << std::endl;
    }
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

      // Define an MPI version of AM
      apsc::LinearAlgebra::MPISparseMatrix<decltype(AM), decltype(e),
                                           decltype(AM)::IsRowMajor
                                               ? apsc::ORDERINGTYPE::ROWWISE
                                               : apsc::ORDERINGTYPE::COLUMNWISE>
          PAM;
      PAM.setup(AM, mpi_comm);

      // retrive y, the unpreconditioned solver is called as the preconditioner
      // is embedded in the input matrix
      r = apsc::LinearAlgebra::Utils::Solvers::GMRES::solve_MPI<
          decltype(PAM), decltype(b), double, decltype(e), 0>(
          PAM, b, y, e, GMRES_MAX_ITER(global_size),
          MPIContext(mpi_comm, mpi_rank, mpi_size),
          objective_context(
              0, mpi_size,
              std::string("parallel_gmres_with_precon_matrix_AM") +
                  std::string("_epsilon") + std::to_string(tol) +
                  std::string("_MPISIZE") + std::to_string(mpi_size) + ".log",
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

      // Define an MPI version of A
      apsc::LinearAlgebra::MPISparseMatrix<decltype(A), decltype(e),
                                           decltype(A)::IsRowMajor
                                               ? apsc::ORDERINGTYPE::ROWWISE
                                               : apsc::ORDERINGTYPE::COLUMNWISE>
          PA;
      PA.setup(A, mpi_comm);

      r = apsc::LinearAlgebra::Utils::Solvers::GMRES::solve_MPI<
          decltype(PA), decltype(b), double, decltype(e)>(
          PA, b, x, e, GMRES_MAX_ITER(global_size),
          MPIContext(mpi_comm, mpi_rank, mpi_size),
          objective_context(
              0, mpi_size,
              std::string("parallel_gmres_with_no_precon_matrix_A") +
                  std::string("_epsilon") + std::to_string(tol) +
                  std::string("_MPISIZE") + std::to_string(mpi_size) + ".log",
              std::string(argv[1])));
    }
  }

  MPI_Finalize();
  return 0;
}
