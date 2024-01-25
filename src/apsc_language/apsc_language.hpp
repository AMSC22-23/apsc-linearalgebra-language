/**
 * @file apsc_language.hpp
 * @brief Header file containing defining apsc custom linear algebra language.
 * It aims to ease the developer experience when dealing with linear algebra
 * operations combined with parallel computing using MPI.
 * @author Kaixi Matteo Chen
 */
#include <mpi.h>

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <MPIContext.hpp>
#include <MPIMatrix.hpp>
#include <MPISparseMatrix.hpp>
#include <Matrix/Matrix.hpp>
#include <MatrixWithVecSupport.hpp>
#include <Parallel/Utilities/partitioner.hpp>
#include <Vector.hpp>
#include <algorithm>
#include <assert.hpp>
#include <cg.hpp>
#include <cg_mpi.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <utils.hpp>

namespace apsc::LinearAlgebra {
namespace Language {
/**
 * @brief Enum class defing the matrix ordering type
 */
enum class OrderingType { ROWMAJOR = 0, COLUMNMAJOR = 1 };

/**
 * @brief A class representing a sparse matrix.
 * @tparam Scalar The scalar type of the matrix elements.
 * @tparam OT The ordering type of the matrix.
 * @tparam USE_MPI Flag indicating whether MPI is used or not.
 */
template <typename Scalar, OrderingType OT = OrderingType::COLUMNMAJOR,
          int USE_MPI = 1>
class SparseMatrix {
 public:
  using Vector = Eigen::VectorX<Scalar>;
  /**
   * @brief Constructor for SparseMatrix.
   */
  SparseMatrix<Scalar, OT, USE_MPI>() {
    communicator = MPI_COMM_WORLD;
    MPI_Comm_rank(communicator, &mpi_rank);
    MPI_Comm_size(communicator, &mpi_size);
  }
  /**
   * @brief Sets up MPI communication after resizing or modifying data.
   */
  void setup_mpi() {
    if constexpr (USE_MPI) {
      eigen_sparse_matrix.makeCompressed();
      parallel_sparse_matrix.setup(eigen_sparse_matrix, MPI_COMM_WORLD);
    }
  }
  /**
   * @brief Accesses elements of the matrix for modification.
   * @param i Row index.
   * @param j Column index.
   * @return Reference to the matrix element.
   */
  Scalar& operator()(const int i, const int j) {
    if constexpr (USE_MPI) {
      if (mpi_rank == 0) {
        return eigen_sparse_matrix.coeffRef(i, j);
      } else {
        return eigen_sparse_matrix.coeffRef(0, 0);
      }
    } else {
      return eigen_sparse_matrix.coeffRef(i, j);
    }
  }
  /**
   * @brief Accesses elements of the matrix for read-only purposes.
   * @param i Row index.
   * @param j Column index.
   * @return Value of the matrix element.
   */
  Scalar operator()(const int i, const int j) const {
    if constexpr (USE_MPI) {
      if (mpi_rank == 0) {
        return eigen_sparse_matrix.coeff(i, j);
      } else {
        return static_cast<Scalar>(0);
      }
    } else {
      return eigen_sparse_matrix.coeff(i, j);
    }
  }
  /**
   * @brief Performs matrix-vector multiplication.
   * @param vec Input vector.
   * @return Resultant vector after multiplication.
   */
  Vector operator*(Vector const& vec) {
    if constexpr (USE_MPI) {
      Vector collection;
      parallel_sparse_matrix.product(vec);
      parallel_sparse_matrix.AllCollectGlobal(collection);
      return collection;
    } else {
      return eigen_sparse_matrix * vec;
    }
  }
  /**
   * @brief Resizes the matrix.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  void resize(const int rows, const int cols) {
    if constexpr (USE_MPI) {
      if (mpi_rank == 0) {
        eigen_sparse_matrix.resize(rows, cols);
      }
      MPI_Barrier(communicator);
      setup_mpi();
    } else {
      eigen_sparse_matrix.resize(rows, cols);
    }
  }
  /**
   * @brief Solves the linear system iteratively.
   * @param rhs Right-hand side vector.
   * @return Solution vector.
   */
  Vector solve_iterative(Vector rhs) {
    Vector x(rhs.size());
    x.fill(static_cast<Scalar>(0.0));
    // Maybe parametrise this
    int max_iter = CG_MAX_ITER(rhs.size());
    Scalar tol = CG_TOL;

    if constexpr (USE_MPI) {
      solver_info = ::LinearAlgebra::LinearSolvers::MPI::CG_no_precon<
          decltype(parallel_sparse_matrix), decltype(rhs), Scalar>(
          parallel_sparse_matrix, x, rhs, max_iter, tol,
          MPIContext(communicator, mpi_rank), MPI_DOUBLE);
    } else {
      // Create identity preconditioner
      auto I = Eigen::IdentityPreconditioner();
      solver_info = ::LinearAlgebra::LinearSolvers::Sequential::CG<
          decltype(eigen_sparse_matrix), decltype(rhs), decltype(I), Scalar>(
          eigen_sparse_matrix, x, rhs, I, max_iter, tol);
    }
    return x;
  }
  /**
   * @brief Solves the linear system directly.
   * @param rhs Right-hand side vector.
   * @return Solution vector.
   */
  Vector solve_direct(Vector& rhs) {
    Eigen::BiCGSTAB<decltype(eigen_sparse_matrix)> solver;
    Vector x;
    if constexpr (USE_MPI) {
      if (mpi_rank == 0) {
        solver.compute(eigen_sparse_matrix);
        x = solver.solve(rhs);
        solver_info = solver.info();
      }
    } else {
      solver.compute(eigen_sparse_matrix);
      x = solver.solve(rhs);
      solver_info = solver.info();
    }
    return x;
  }
  /**
   * @brief Loads the matrix data from a file.
   * @param file_name Path to the file containing matrix data.
   */
  void load_from_file(std::string file_name) {
    auto load = [&]() {
      ASSERT(Eigen::loadMarket(eigen_sparse_matrix, file_name),
             "Failed to load matrix from file" << std::endl);
    };

    if constexpr (USE_MPI) {
      if (mpi_rank == 0) {
        load();
      } else {
        // Needed for a reference retrurn for operator()
        eigen_sparse_matrix.resize(1, 1);
        eigen_sparse_matrix.insert(0, 0) = static_cast<Scalar>(0);
        eigen_sparse_matrix.makeCompressed();
      }
      MPI_Barrier(communicator);
      setup_mpi();
    } else {
      load();
    }
  }
  /**
   * @brief Displays the split of the matrix across MPI processes.
   */
  void show_mpi_split() {
    if constexpr (USE_MPI) {
      Utils::MPI_matrix_show<decltype(parallel_sparse_matrix)>(
          parallel_sparse_matrix, mpi_rank, mpi_size, communicator);
    }
  }
  /**
   * @brief Retrieves the number of rows in the matrix.
   * @return Number of rows.
   */
  int rows() const { return eigen_sparse_matrix.rows(); }
  /**
   * @brief Retrieves the number of columns in the matrix.
   * @return Number of columns.
   */
  int cols() const { return eigen_sparse_matrix.cols(); }
  /**
   * @brief Retrieves the number of non zero elements.
   * @return Number of non zeros.
   */
  int non_zeros() const { return eigen_sparse_matrix.nonZeros(); }
  /**
   * @brief Retrieves the solver status code.
   * @return The solver status code.
   */
  int solver_success() const { return solver_info; }
  /**
   * @brief Stream the matrix to an output stream
   * @param os A reference to an output stream.
   */
  void stream_to(std::ostream& os) const { os << eigen_sparse_matrix; }

 protected:
  /**
   * @brief MPI size.
   */
  int mpi_size;
  /**
   * @brief MPI rank.
   */
  int mpi_rank;
  /**
   * @brief MPI communicator.
   */
  MPI_Comm communicator;
  /**
   * @brief Linear solver status code.
   */
  int solver_info = 0;
  /**
   * @brief Instance of a sparse matrix.
   */
  Eigen::SparseMatrix<double, OT == OrderingType::COLUMNMAJOR ? Eigen::ColMajor
                                                              : Eigen::RowMajor>
      eigen_sparse_matrix;
  /**
   * @brief Instance of a MPI sparse matrix.
   */
  MPISparseMatrix<decltype(eigen_sparse_matrix), Vector,
                  OT == OrderingType::COLUMNMAJOR ? ORDERINGTYPE::COLUMNWISE
                                                  : ORDERINGTYPE::ROWWISE>
      parallel_sparse_matrix;
};

/**
 * @brief A class representing a full matrix.
 * @tparam Scalar The scalar type of the matrix elements.
 * @tparam OT The ordering type of the matrix.
 * @tparam USE_MPI Flag indicating whether MPI is used or not.
 */
template <typename Scalar, OrderingType OT = OrderingType::COLUMNMAJOR,
          int USE_MPI = 1>
inline std::ostream& operator<<(std::ostream& os,
                                SparseMatrix<Scalar, OT, USE_MPI> const& mat) {
  mat.stream_to(os);
  return os;
}

template <typename Scalar, OrderingType OT = OrderingType::COLUMNMAJOR,
          int USE_MPI = 1>
class FullMatrix {
 public:
  /**
   * @brief Constructor for FullMatrix.
   */
  FullMatrix() {
    if constexpr (USE_MPI) {
      communicator = MPI_COMM_WORLD;
      MPI_Comm_rank(communicator, &mpi_rank);
      MPI_Comm_size(communicator, &mpi_size);
    }
  }
  /**
   * @brief Accesses elements of the matrix for modification.
   * @param i Row index.
   * @param j Column index.
   * @return Reference to the matrix element.
   */
  Scalar& operator()(const int i, const int j) {
    if constexpr (USE_MPI) {
      if (mpi_rank == 0) {
        return full_matirx(i, j);
      } else {
        return full_matirx(0, 0);
      }
    } else {
      return full_matirx(i, j);
    }
  }
  /**
   * @brief Accesses elements of the matrix for read-only purposes.
   * @param i Row index.
   * @param j Column index.
   * @return Value of the matrix element.
   */
  Scalar operator()(const int i, const int j) const {
    if constexpr (USE_MPI) {
      if (mpi_rank == 0) {
        return full_matirx(i, j);
      } else {
        return static_cast<Scalar>(0);
      }
    } else {
      return full_matirx(i, j);
    }
  }
  /**
   * @brief Performs matrix-vector multiplication.
   * @param vec Input vector.
   * @return Resultant vector after multiplication.
   */
  Vector<Scalar> operator*(Vector<Scalar> const& vec) {
    if constexpr (USE_MPI) {
      Vector<Scalar> collection;
      paralle_full_matrix.product(vec);
      paralle_full_matrix.AllCollectGlobal(collection);
      return collection;
    } else {
      return full_matirx * vec;
    }
  }
  /**
   * @brief Fill the matrix with a scalar value. This will call the MPI setup.
   * @param value Scalar value.
   */
  void fill(const Scalar value) {
    std::fill(full_matirx.data(), full_matirx.data() + (m * n), value);
    setup_mpi();
  }
  /**
   * @brief Resizes the matrix.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  void resize(const int rows, const int cols) {
    if constexpr (USE_MPI) {
      if (mpi_rank == 0) {
        full_matirx.clear();
        full_matirx.resize(rows, cols);
        m = rows;
        n = cols;
      }
      MPI_Barrier(communicator);
      setup_mpi();
    } else {
      full_matirx.clear();
      full_matirx.resize(rows, cols);
      m = rows;
      n = cols;
    }
  }
  /**
   * @brief Solves the linear system directly.
   * @param rhs Right-hand side vector.
   * @return Solution vector.
   */
  Vector<Scalar> solve_direct(Vector<Scalar>& rhs) {
    if constexpr (USE_MPI) {
      if (mpi_rank == 0) {
        return full_matirx.solve(rhs);
      } else {
        return Vector<Scalar>();
      }
    } else {
      return full_matirx.solve(rhs);
    }
  }
  /**
   * @brief Solves the linear system iteratively.
   * @param rhs Right-hand side vector.
   * @return Solution vector.
   */
  Vector<Scalar> solve_iterative(Vector<Scalar> rhs) {
    Vector<Scalar> x(rhs.size(), static_cast<Scalar>(0.0));
    // Maybe parametrise this
    int max_iter = CG_MAX_ITER(rhs.size());
    Scalar tol = CG_TOL;

    if constexpr (USE_MPI) {
      solver_info = ::LinearAlgebra::LinearSolvers::MPI::CG_no_precon<
          decltype(paralle_full_matrix), decltype(rhs), Scalar>(
          paralle_full_matrix, x, rhs, max_iter, tol,
          MPIContext(communicator, mpi_rank), MPI_DOUBLE);
    } else {
      // Create identity preconditioner
      decltype(full_matirx) I(m, n);
      std::fill(I.data(), I.data() + (m * n), static_cast<Scalar>(0));
      for (int i = 0; i < m; i++) {
        I(i, i) = static_cast<Scalar>(1);
      }
      solver_info = ::LinearAlgebra::LinearSolvers::Sequential::CG<
          decltype(full_matirx), decltype(rhs), decltype(I), Scalar>(
          full_matirx, x, rhs, I, max_iter, tol);
    }
    return x;
  }
  /**
   * @brief Sets up MPI communication after resizing or modifying data.
   */
  void setup_mpi() {
    if constexpr (USE_MPI) {
      paralle_full_matrix.setup(full_matirx, MPI_COMM_WORLD);
    }
  }
  /**
   * @brief Loads the matrix data from a file.
   * @param file_name Path to the file containing matrix data.
   */
  void load_from_file(std::string file_name) {
    auto load = [&]() {
      ASSERT(Eigen::loadMarket(eigen_load_source_matrix, file_name),
             "Failed to load matrix from file" << std::endl);
      // Convert sparse matrix to full matrix;
      auto eigen_full_matrix =
          Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(
              eigen_load_source_matrix);
      // Resize the full_matrix container and copy data
      full_matirx.resize(eigen_full_matrix.rows(), eigen_full_matrix.cols());
      memcpy(full_matirx.data(), eigen_full_matrix.data(),
             sizeof(Scalar) *
                 (eigen_full_matrix.rows() * eigen_full_matrix.cols()));
    };

    if constexpr (USE_MPI) {
      if (mpi_rank == 0) {
        load();
        m = full_matirx.rows();
        n = full_matirx.cols();
      } else {
        // Needed for a reference retrurn for operator()
        full_matirx.resize(1, 1);
        full_matirx(0, 0) = static_cast<Scalar>(0);
      }
      MPI_Barrier(communicator);
      setup_mpi();
    } else {
      load();
      m = full_matirx.rows();
      n = full_matirx.cols();
    }
  }
  /**
   * @brief Displays the split of the matrix across MPI processes.
   */
  void show_mpi_split() {
    if constexpr (USE_MPI) {
      Utils::MPI_matrix_show<decltype(paralle_full_matrix)>(
          paralle_full_matrix, mpi_rank, mpi_size, communicator);
    }
  }
  /**
   * @brief Retrieves the number of rows in the matrix.
   * @return Number of rows.
   */
  int rows() const { return m; }
  /**
   * @brief Retrieves the number of columns in the matrix.
   * @return Number of columns.
   */
  int cols() const { return n; }
  /**
   * @brief Retrieves the solver status code.
   * @return The solver status code.
   */
  int solver_success() const { return solver_info; }
  /**
   * @brief Stream the matrix to an output stream
   * @param os A reference to an output stream.
   */
  void stream_to(std::ostream& os) const { os << full_matirx; }

 protected:
  /**
   * @brief MPI size.
   */
  int mpi_size;
  /**
   * @brief MPI rank.
   */
  int mpi_rank;
  /**
   * @brief MPI communicator.
   */
  MPI_Comm communicator;
  /**
   * @brief Matrix rows count.
   */
  int m = 0;
  /**
   * @brief Matrix columns count.
   */
  int n = 0;
  /**
   * @brief Linear solver status code.
   */
  int solver_info = 0;
  /**
   * @brief Instance of a full matrix.
   */
  MatrixWithVecSupport<double, Vector<double>,
                       OT == OrderingType::COLUMNMAJOR ? ORDERING::COLUMNMAJOR
                                                       : ORDERING::ROWMAJOR>
      full_matirx;
  /**
   * @brief Instance of a MPI full matrix.
   */
  MPIMatrix<decltype(full_matirx), Vector<Scalar>,
            OT == OrderingType::COLUMNMAJOR ? ORDERINGTYPE::COLUMNWISE
                                            : ORDERINGTYPE::ROWWISE>
      paralle_full_matrix;
  /**
   * @brief Eigen sparse matrix helper to load a matrix from file.
   */
  Eigen::SparseMatrix<Scalar> eigen_load_source_matrix;
};

template <typename Scalar, OrderingType OT = OrderingType::COLUMNMAJOR,
          int USE_MPI = 1>
inline std::ostream& operator<<(std::ostream& os,
                                FullMatrix<Scalar, OT, USE_MPI>& mat) {
  mat.stream_to(os);
  return os;
}
}  // namespace Language
}  // namespace apsc::LinearAlgebra
