#include <mpi.h>

#include <MPIMatrix.hpp>
#include <Matrix/Matrix.hpp>
#include <MatrixWithVecSupport.hpp>
#include <Vector.hpp>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <utils.hpp>

#include "Parallel/Utilities/partitioner.hpp"

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
  apsc::LinearAlgebra::MatrixWithVecSupport<
      double, apsc::LinearAlgebra::Vector<double>,
      apsc::LinearAlgebra::ORDERING::COLUMNMAJOR>
      A(size, size);
  if (mpi_rank == 0) {
    apsc::LinearAlgebra::Utils::default_spd_fill<decltype(A), double>(A);
  }

  // Create a Vector
  apsc::LinearAlgebra::Vector<double> x(size, 1.0);

  // Create the MPI full matrix
  apsc::MPIMatrix<decltype(A), decltype(x), apsc::ORDERINGTYPE::COLUMNWISE> PA;
  PA.setup(A, mpi_comm);
  int rank = 0;
#if DEBUG_LOCAL_MATRIX == 1
  while (rank < mpi_size) {
    if (mpi_rank == rank) {
      std::cout << "Process rank=" << mpi_rank << " Local Matrix=";
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
  std::cout << "product time = "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                   .count()
            << "[ns]" << std::endl;
  PA.AllCollectGlobal(res);
  if (mpi_rank == 0) {
    // std::cout << "Product result:" << std::endl << res << std::endl;
  }

  MPI_Finalize();

  return 0;
}
