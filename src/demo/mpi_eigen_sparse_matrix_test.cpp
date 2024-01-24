#include <mpi.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <MPISparseMatrix.hpp>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <utils.hpp>
#include <vector>

#define DEBUG_EIGEN_INTERNAL_STRUCTURE 0
#define DEBUG_LOCAL_MATRIX 0

constexpr std::size_t size = 10000;

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  int mpi_rank;
  int mpi_size;
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  MPI_Comm_rank(mpi_comm, &mpi_rank);
  MPI_Comm_size(mpi_comm, &mpi_size);

  // Create the global full matrix
  Eigen::SparseMatrix<double, Eigen::ColMajor> A;
  if (mpi_rank == 0) {
    A.resize(size, size);
    for (unsigned i = 0; i < size; i++) {
      A.insert(i, i) = 1.0;
      if (i > 0) {
        A.insert(i, i - 1) = -1.0;
      }
      if (i < size - 1) {
        A.insert(i, i + 1) = -1.0;
      }
    }
  }
  A.makeCompressed();

  if (mpi_rank == 0) {
    // std::cout << "Matrix A" << std::endl << A << std::endl;
#if (DEBUG_EIGEN_INTERNAL_STRUCTURE == 1)
    for (int i = 0; i < size; i++) {
      int k_start = A.outerIndexPtr()[i];
      int k_end = A.outerIndexPtr()[i + 1];

      for (int k = k_start; k < k_end; k++) {
        int j = A.innerIndexPtr()[k];
        double v = A.valuePtr()[k];
        if constexpr (decltype(A)::IsRowMajor) {
          std::cout << "A(" << i << "," << j << ") = " << v << std::endl;
        } else {
          std::cout << "A(" << j << "," << i << ") = " << v << std::endl;
        }
      }
    }
#endif
  }

  // Create a Vector
  Eigen::VectorXd x(size);
  for (std::size_t i = 0; i < size; i++) {
    x[i] = 1.0;
  }
  MPI_Barrier(mpi_comm);

  // Create the MPI full matrix
  apsc::MPISparseMatrix<decltype(A), decltype(x),
                        decltype(A)::IsRowMajor
                            ? apsc::ORDERINGTYPE::ROWWISE
                            : apsc::ORDERINGTYPE::COLUMNWISE>
      PA;
  PA.setup(A, mpi_comm);
  int rank = 0;
#if DEBUG_LOCAL_MATRIX == 1
  while (rank < mpi_size) {
    if (mpi_rank == rank) {
      std::cout << "Process rank=" << mpi_rank << " Local Matrix=" << std::endl;
      std::cout << PA.getLocalMatrix();
    }
    rank++;
    MPI_Barrier(mpi_comm);
  }
#endif

  // Product
  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  PA.product(x);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  apsc::LinearAlgebra::Vector<double> res;
  std::cout << std::endl
            << "product time = "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                   .count()
            << "[ns]" << std::endl;

  begin = std::chrono::steady_clock::now();
  PA.AllCollectGlobal(res);
  end = std::chrono::steady_clock::now();
  std::cout << std::endl
            << "collection time = "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                   .count()
            << "[ns]" << std::endl;
  if (mpi_rank == 0) {
    // std::cout << "Product result:" << std::endl << res << std::endl;
  }

  MPI_Finalize();

  return 0;
}
