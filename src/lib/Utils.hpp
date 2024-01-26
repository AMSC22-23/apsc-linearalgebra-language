#ifndef UTILS_HPP
#define UTILS_HPP

#include <mpi.h>

#include <MPIContext.hpp>
#include <assert.hpp>
#include <cassert>
#include <cg_mpi.hpp>
#include <chrono>
#include <fstream>
#include <gmres.hpp>
#include <iostream>
#include <ObjectiveContext.hpp>
#include <string>
#include <type_traits>
#include <unsupported/Eigen/SparseExtra>

using std::cout;
using std::endl;

#define PRODUCE_OUT_FILE 1
#define CG_MAX_ITER(i) (20 * i)
#define CG_TOL 1e-8;

#define GMRES_MAX_ITER(i) (20 * i)
#define GMRES_TOL 1e-8;

namespace apsc::LinearAlgebra
{
namespace Utils {
template <typename Mat, typename Scalar>
void default_spd_fill(Mat &m) {
  ASSERT((m.rows() == m.cols()), "The matrix must be squared!");
  const Scalar diagonal_value = static_cast<Scalar>(2.0);
  const Scalar upper_diagonal_value = static_cast<Scalar>(-1.0);
  const Scalar lower_diagonal_value = static_cast<Scalar>(-1.0);
  const auto size = m.rows();

  for (unsigned i = 0; i < size; ++i) {
    m(i, i) = diagonal_value;
    if (i > 0) {
      m(i, i - 1) = upper_diagonal_value;
    }
    if (i < size - 1) {
      m(i, i + 1) = lower_diagonal_value;
    }
  }
}

template <typename Vector>
double vector_norm_2(Vector v) {
  double norm = 0.0;
  for (int i = 0; i < v.size(); i++) {
    norm += v[i] * v[i];
  }
  return std::sqrt(norm);
}

template <typename SparseMatrix>
void print_matlab_matrix(SparseMatrix m, std::string file_name) {
  std::ofstream out_file(file_name);
  if (!out_file.is_open()) {
    std::cerr << "Error opening the output" << endl;
    return;
  }
  out_file << "[" << endl;
  for (int i = 0; i < m.rows(); ++i) {
    for (int j = 0; j < m.cols(); ++j) {
      out_file << m.coeffRef(i, j);
      if (j < m.cols() - 1)
        out_file << ",";
      else if (j == m.cols() - 1 && i < m.rows() - 1)
        out_file << ";";
    }
  }
  out_file << "]" << endl;
  out_file.close();
}

namespace MPIUtils {
class MPIRunner {
 public:
  int mpi_rank;
  int mpi_size;
  MPI_Comm communicator;

  MPIRunner(int *argc, char **argv[]) {
    MPI_Init(argc, argv);
    communicator = MPI_COMM_WORLD;
    MPI_Comm_rank(communicator, &mpi_rank);
    MPI_Comm_size(communicator, &mpi_size);
  }

  ~MPIRunner() { MPI_Finalize(); }
};
}  // namespace MPIUtils

namespace EigenUtils {
template <typename Matrix, typename Scalar>
void load_sparse_matrix(const std::string file_name, Matrix &mat) {
  static_assert(
      std::is_same_v<Matrix, Eigen::SparseMatrix<Scalar, Eigen::RowMajor>> ||
          std::is_same_v<Matrix, Eigen::SparseMatrix<Scalar, Eigen::ColMajor>>,
      "Matrix type must be an Eigen::SparseMatrix of type Scalar");
  ASSERT(Eigen::loadMarket(mat, file_name),
         "Failed to load matrix from " << file_name);
}
}  // namespace EigenUtils

template <typename MPIMatrix>
void MPI_matrix_show(MPIMatrix MPIMat, const int mpi_rank, const int mpi_size,
                     MPI_Comm mpi_comm) {
  int rank = 0;
  while (rank < mpi_size) {
    if (mpi_rank == rank) {
      std::cout << "Process rank=" << mpi_rank << " Local Matrix=" << endl;
      std::cout << MPIMat.getLocalMatrix();
    }
    rank++;
    MPI_Barrier(mpi_comm);
  }
}

namespace Solvers {
namespace GMRES {
template <typename MPILhs, typename Rhs, typename Scalar, typename ExactSol,
          int SHOW_ERROR_NORM = 1, typename... Preconditioner>
int solve(MPILhs &A, Rhs &b, Rhs &x, ExactSol &e, int restart,
          const MPIContext mpi_ctx, ObjectiveContext obj_ctx,
          Preconditioner &...P) {
  constexpr std::size_t P_size = sizeof...(P);
  static_assert(P_size < 2, "Please specify max 1 preconditioner");

#if PRODUCE_OUT_FILE == 0
  (void)obj_ctx;
#endif

  const int size = b.size();

  x.resize(size);
  x.fill(0.0);
  int max_iter = GMRES_MAX_ITER(size);
  Scalar tol = GMRES_TOL;

  if constexpr (P_size == 0) {
    auto id = Eigen::IdentityPreconditioner();
    std::chrono::high_resolution_clock::time_point begin =
        std::chrono::high_resolution_clock::now();
    auto result =
        apsc::LinearAlgebra::LinearSolvers::Sequential::GMRES<MPILhs, Rhs,
                                                          decltype(id)>(
            A, x, b, id, restart, max_iter, tol);
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    long long diff =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
            .count();
    if (mpi_ctx.mpi_rank() == 0) {
      cout << "(Time spent: " << diff << "[µs]" << endl;
      cout << "Solution with GMRES:" << endl;
      cout << "iterations performed:                      " << max_iter << endl;
      cout << "tolerance achieved:                        " << tol << endl;
      if constexpr (SHOW_ERROR_NORM)
        cout << "Error norm:                                " << (x - e).norm()
             << endl;
#if PRODUCE_OUT_FILE == 1
      {
        obj_ctx.write(static_cast<long long>(size), ',',
                      static_cast<long long>(diff), ',',
                      static_cast<long long>(max_iter), ',',
                      static_cast<long long>(result));
      }
#endif
#if DEBUG == 1
      cout << "Result vector:                             " << x << endl;
#endif
    }
    return result;
  } else {
    std::chrono::high_resolution_clock::time_point begin =
        std::chrono::high_resolution_clock::now();
    auto result =
        apsc::LinearAlgebra::LinearSolvers::Sequential::GMRES<MPILhs, Rhs,
                                                          Preconditioner...>(
            A, x, b, P..., restart, max_iter, tol);
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    long long diff =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
            .count();

    if (mpi_ctx.mpi_rank() == 0) {
      cout << "(Time spent: " << diff << "[µs]" << endl;
      cout << "Solution with GMRES:" << endl;
      cout << "iterations performed:                      " << max_iter << endl;
      cout << "tolerance achieved:                        " << tol << endl;
      if constexpr (SHOW_ERROR_NORM)
        cout << "Error norm:                                " << (x - e).norm()
             << endl;
#if PRODUCE_OUT_FILE == 1
      {
        obj_ctx.write(static_cast<long long>(size), ',',
                      static_cast<long long>(diff), ',',
                      static_cast<long long>(max_iter), ',',
                      static_cast<long long>(result));
      }
#endif
#if DEBUG == 1
      cout << "Result vector:                             " << x << endl;
#endif
    }
    return result;
  }
}
template <typename MPILhs, typename Rhs, typename Scalar, typename ExactSol,
          int SHOW_ERROR_NORM = 1, typename... Preconditioner>
int solve_MPI(MPILhs &A, Rhs &b, Rhs &x, ExactSol &e, int restart,
              const MPIContext mpi_ctx, ObjectiveContext obj_ctx,
              Preconditioner &...P) {
  constexpr std::size_t P_size = sizeof...(P);
  static_assert(P_size < 2, "Please specify max 1 preconditioner");

#if PRODUCE_OUT_FILE == 0
  (void)obj_ctx;
#endif

  const int size = b.size();

  x.resize(size);
  x.fill(0.0);
  int max_iter = GMRES_MAX_ITER(size);
  Scalar tol = GMRES_TOL;

  if constexpr (P_size == 0) {
    auto id = Eigen::IdentityPreconditioner();
    std::chrono::high_resolution_clock::time_point begin =
        std::chrono::high_resolution_clock::now();
    auto result =
        apsc::LinearAlgebra::LinearSolvers::MPI::GMRES<MPILhs, Rhs, decltype(id)>(
            A, x, b, id, restart, max_iter, tol);
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    long long diff =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
            .count();
    decltype(diff) diff_sum = 0;

    MPI_Reduce(&diff, &diff_sum, 1, MPI_LONG_LONG, MPI_SUM, 0,
               mpi_ctx.mpi_comm());

    if (mpi_ctx.mpi_rank() == 0) {
      cout << "(Time spent by all processes: " << diff_sum
           << ", total processes: " << mpi_ctx.mpi_size() << ")" << endl;
      cout << "Mean elapsed time = " << diff_sum / mpi_ctx.mpi_size() << "[µs]"
           << endl;
      cout << "Solution with GMRES:" << endl;
      cout << "iterations performed:                      " << max_iter << endl;
      cout << "tolerance achieved:                        " << tol << endl;
      if constexpr (SHOW_ERROR_NORM)
        cout << "Error norm:                                " << (x - e).norm()
             << endl;
#if PRODUCE_OUT_FILE == 1
      {
        obj_ctx.write(static_cast<long long>(size), ',',
                      static_cast<long long>(diff_sum / mpi_ctx.mpi_size()),
                      ',', static_cast<long long>(max_iter), ',',
                      static_cast<long long>(result));
      }
#endif
#if DEBUG == 1
      cout << "Result vector:                             " << x << endl;
#endif
    }
    return result;
  } else {
    std::chrono::high_resolution_clock::time_point begin =
        std::chrono::high_resolution_clock::now();
    auto result = apsc::LinearAlgebra::LinearSolvers::MPI::GMRES<MPILhs, Rhs,
                                                             Preconditioner...>(
        A, x, b, P..., restart, max_iter, tol);
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    long long diff =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
            .count();
    decltype(diff) diff_sum = 0;

    MPI_Reduce(&diff, &diff_sum, 1, MPI_LONG_LONG, MPI_SUM, 0,
               mpi_ctx.mpi_comm());

    if (mpi_ctx.mpi_rank() == 0) {
      cout << "(Time spent by all processes: " << diff_sum
           << ", total processes: " << mpi_ctx.mpi_size() << ")" << endl;
      cout << "Mean elapsed time = " << diff_sum / mpi_ctx.mpi_size() << "[µs]"
           << endl;
      cout << "Solution with GMRES:" << endl;
      cout << "iterations performed:                      " << max_iter << endl;
      cout << "tolerance achieved:                        " << tol << endl;
      if constexpr (SHOW_ERROR_NORM)
        cout << "Error norm:                                " << (x - e).norm()
             << endl;
#if PRODUCE_OUT_FILE == 1
      {
        obj_ctx.write(static_cast<long long>(size), ',',
                      static_cast<long long>(diff_sum / mpi_ctx.mpi_size()),
                      ',', static_cast<long long>(max_iter), ',',
                      static_cast<long long>(result));
      }
#endif
#if DEBUG == 1
      cout << "Result vector:                             " << x << endl;
#endif
    }
    return result;
  }
}
}  // namespace GMRES

namespace ConjugateGradient {
template <typename MPILhs, typename Rhs, typename Scalar, typename ExactSol,
          typename... Preconditioner>
int solve_MPI(MPILhs &A, Rhs b, ExactSol &e, const MPIContext mpi_ctx,
              ObjectiveContext obj_ctx = {}, bool produce_out_file = 1,
              Preconditioner... P) {
  constexpr std::size_t P_size = sizeof...(P);
  static_assert(P_size < 2, "Please specify max 1 preconditioner");

#if PRODUCE_OUT_FILE == 0
  (void)obj_ctx;
#endif

  const int size = b.size();

  Rhs x;
  x.resize(size);
  x.fill(0.0);
  int max_iter = CG_MAX_ITER(size);
  Scalar tol = CG_TOL;

  if constexpr (P_size == 0) {
    std::chrono::high_resolution_clock::time_point begin =
        std::chrono::high_resolution_clock::now();
    auto result =
        apsc::LinearAlgebra::LinearSolvers::MPI::CG_no_precon<MPILhs, Rhs, Scalar>(
            A, x, b, max_iter, tol, mpi_ctx, MPI_DOUBLE);
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    long long diff =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
            .count();
    decltype(diff) diff_sum = 0;

    MPI_Reduce(&diff, &diff_sum, 1, MPI_LONG_LONG, MPI_SUM, 0,
               mpi_ctx.mpi_comm());

    if (mpi_ctx.mpi_rank() == 0) {
      cout << "(Time spent by all processes: " << diff_sum
           << ", total processes: " << mpi_ctx.mpi_size() << ")" << std::endl;
      cout << "Mean elapsed time = " << diff_sum / mpi_ctx.mpi_size() << "[µs]"
           << endl;
      cout << "Solution with Conjugate Gradient:" << endl;
      cout << "iterations performed:                      " << max_iter << endl;
      cout << "tolerance achieved:                        " << tol << endl;
      cout << "Error norm:                                " << (x - e).norm()
           << std::endl;
      if (produce_out_file) {
        obj_ctx.write(static_cast<long long>(size), ',',
                      static_cast<long long>(diff_sum / mpi_ctx.mpi_size()),
                      ',', static_cast<long long>(max_iter), ',',
                      static_cast<long long>(result));
      }
#if DEBUG == 1
      cout << "Result vector:                             " << x << std::endl;
#endif
    }
    return result;
  } else {
    // TODO
  }
}
}  // namespace ConjugateGradient
}  // namespace Solvers
}  // namespace Utils
}  // namespace apsc::LinearAlgebra

#endif /*UTILS_HPP*/