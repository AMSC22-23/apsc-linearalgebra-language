#include <mpi.h>

#include <MPIContext.hpp>
#include <MPIFullMatrix.hpp>
#include <Matrix/Matrix.hpp>
#include <FullMatrix.hpp>
#include <Vector.hpp>
#include <chrono>
#include <cstddef>
#include <iostream>

using std::cout;
using std::endl;

#define DEBUG 0
#define USE_PRECONDITIONER 0

constexpr uint8_t objective_id = 2;

int main(int argc, char *argv[]) {
  using namespace apsc::LinearAlgebra;

  if (argc < 2) {
    std::cerr << "Please define the problem size as argument" << std::endl;
    return 0;
  }

  const int size = atoi(argv[1]);

  MPI_Init(&argc, &argv);
  int mpi_rank;
  int mpi_size;
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  MPI_Comm_rank(mpi_comm, &mpi_rank);
  MPI_Comm_size(mpi_comm, &mpi_size);

  FullMatrix<double, Vector<double>,
                       apsc::LinearAlgebra::ORDERING::ROWMAJOR>
      A;
  if (mpi_rank == 0) {
    A.resize(size, size);
    cout << "Launching CG with problem (SPD matrix) size of " << size << "x"
         << size << endl;
    Utils::default_spd_fill<
        FullMatrix<double, Vector<double>, ORDERING::ROWMAJOR>,
        double>(A);
  }

  // Maintain whole vectors in each processes
  Vector<double> e;
  Vector<double> b(size);
  // Initialise processes b vector
  if (!mpi_rank) {
    e.resize(size);
    e.fill(1.0);
    b = A * e;
  }
  MPI_Bcast(b.data(), b.size(), MPI_DOUBLE, 0, mpi_comm);

  apsc::LinearAlgebra::MPIFullMatrix<decltype(A), decltype(b),
                  decltype(A)::ordering == ORDERING::ROWMAJOR
                      ? apsc::ORDERINGTYPE::ROWWISE
                      : apsc::ORDERINGTYPE::COLUMNWISE>
      PA;
  PA.setup(A, mpi_comm);
#if (DEBUG == 1)
  apsc::LinearAlgebra::Utils::MPI_matrix_show(PA, A, mpi_rank, mpi_size,
                                              mpi_comm);
#endif

#if USE_PRECONDITIONER == 0
  auto r = apsc::LinearAlgebra::Utils::Solvers::ConjugateGradient::solve_MPI<
      decltype(PA), decltype(b), double, decltype(e)>(
      PA, b, e, MPIContext(mpi_comm, mpi_rank, mpi_size),
      objective_context(objective_id, mpi_size,
                        "objective" + std::to_string(objective_id) +
                            "_MPISIZE" + std::to_string(mpi_size) + ".log"),
      1);
#else
  // Setup the preconditioner, all the processes for now..
  FullMatrix<double, Vector<double>,
                       apsc::LinearAlgebra::ORDERING::ROWMAJOR>
      P(size, size);
  for (unsigned i = 0; i < size; i++) {
    P(i, i) = 1.0;
  }
  MPI_Barrier(mpi_comm);
  auto r =
      cg_solve_mpi<decltype(PA), decltype(b), double, size, decltype(P),
                   decltype(e)>(PA, b, e, P, MPIContext(&mpi_comm, mpi_rank));
#endif
  MPI_Finalize();

  return r;
}
