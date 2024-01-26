/**
 * @file CSC.hpp
 * @brief Header file containing the CSC struct for compressed sparse column
 * (CSC) matrix representation.
 */

#ifndef CSC_HPP
#define CSC_HPP

#include <mpi.h>
#include <stdint.h>

#include <Eigen/Sparse>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "EigenStructureMap.hpp"
#include "Parallel/Utilities/mpi_utils.hpp"
#include "assert.hpp"
namespace apsc::LinearAlgebra {
/**
 * @struct CSC
 * @brief Struct representing a compressed sparse column (CSC) matrix.
 * @tparam Scalar The scalar type of the matrix.
 */
template <typename Scalar>
struct CSC {
  int m = 0;                /**< Number of rows. */
  int n = 0;                /**< Number of columns. */
  int non_zeros = 0;        /**< Number of non-zero elements. */
  int* offset = nullptr;    /**< Array of column offsets. */
  Scalar* values = nullptr; /**< Array of non-zero values. */
  int* flat_row_index =
      nullptr; /**< Array of row indices for non-zero values. */
  uint8_t initialised =
      0; /**< Flag indicating whether the struct is initialised. */
  uint8_t external_buffer =
      0; /**< Flag indicating whether an external buffer is used. */
  /**
   * @brief Default constructor.
   */
  CSC() = default;
  /**
   * @brief Destructor.
   */
  ~CSC() { destoy(); }
  /**
   * @brief Destroys the CSC struct and releases memory.
   */
  void destoy() {
    if (!external_buffer && initialised) {
      free(offset);
      offset = 0;
      free(values);
      values = 0;
      free(flat_row_index);
      flat_row_index = 0;
    }
    initialised = 0;
    m = 0;
    n = 0;
    non_zeros = 0;
    external_buffer = 0;
  }
  /**
   * @brief Allocates the buffer memory.
   * @param nnz The number of non zero elements.
   * @param rows The number of matrix rows.
   * @param cols The number of matrix columns.
   */
  void allocate(const int nnz, const int rows, const int cols) {
    m = rows, n = cols, non_zeros = nnz;
    offset = (int*)malloc(sizeof(int) * (cols + 1));
    flat_row_index = (int*)malloc(sizeof(int) * (nnz));
    values = (Scalar*)malloc(sizeof(Scalar) * (nnz));
    initialised = 1;
  }
  /**
   * @brief Converts the CSC matrix to an Eigen matrix of the specified type.
   * @tparam EigenMatrixType The Eigen matrix type.
   * @param size The size of the matrix.
   * @return Eigen matrix mapped to the CSC matrix data.
   */
  template <typename EigenMatrixType>
  auto to_eigen(const std::size_t size) const {
    return EigenStructureMap<EigenMatrixType, double, CSC<Scalar>>::create_map(
               size, size, non_zeros, offset, flat_row_index, values)
        .structure();
  }
  /**
   * @brief Solves a linear system using the CSC matrix.
   * @tparam Vector The vector type.
   * @param b The vector representing the right-hand side of the linear system.
   * @return The solution vector.
   */
  template <typename Vector>
  Vector solve(Vector& b) const {
    int size = b.size();
    ASSERT(size == n, "matvet multiplication size does not match, matrix col = "
                          << n << ", vector size = " << size << std::endl);
    using EigenVec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    EigenVec eigen_x;
    // map this CSC matrix into an Eigen type to exploit Eigen iterative solvers
    auto eigen_csc =
        EigenStructureMap<Eigen::SparseMatrix<Scalar, Eigen::ColMajor>, Scalar,
                          CSC<Scalar>>::create_map(m, n, non_zeros, offset,
                                                   flat_row_index, values)
            .structure();
    // map this b vector into an Eigen type to exploit Eigen iterative solvers
    auto eigen_b =
        EigenStructureMap<EigenVec, Scalar, Vector>::create_map(b, b.size())
            .structure();
    Eigen::BiCGSTAB<Eigen::SparseMatrix<Scalar>,
                    Eigen::DiagonalPreconditioner<Scalar>>
        solver;
    solver.compute(eigen_csc);

    // in order to maintain back compatibility map the eigen vector to template
    // Vector type
    eigen_x = solver.solve(eigen_b);
    Vector x(size);
    memcpy(x.data(), eigen_x.data(), sizeof(Scalar) * eigen_x.size());
    return x;
  }
  /**
   * @brief Solves a linear system using the CSC matrix.
   * @tparam Vector The vector type.
   * @param b The vector representing the right-hand side of the linear system.
   * @return The solution vector.
   */
  template <typename Vector>
  Vector solve(Vector& b) {
    int size = b.size();
    ASSERT(size == n, "matvet multiplication size does not match, matrix col = "
                          << n << ", vector size = " << size << std::endl);
    using EigenVec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    EigenVec eigen_x;
    // map this CSC matrix into an Eigen type to exploit Eigen iterative solvers
    auto eigen_csc =
        EigenStructureMap<Eigen::SparseMatrix<Scalar, Eigen::ColMajor>, Scalar,
                          CSC<Scalar>>::create_map(m, n, non_zeros, offset,
                                                   flat_row_index, values)
            .structure();
    // map this b vector into an Eigen type to exploit Eigen iterative solvers
    auto eigen_b =
        EigenStructureMap<EigenVec, Scalar, Vector>::create_map(b, b.size())
            .structure();
    Eigen::BiCGSTAB<Eigen::SparseMatrix<Scalar>,
                    Eigen::DiagonalPreconditioner<Scalar>>
        solver;
    solver.compute(eigen_csc);

    // in order to maintain back compatibility map the eigen vector to template
    // Vector type
    eigen_x = solver.solve(eigen_b);
    Vector x(size);
    memcpy(x.data(), eigen_x.data(), sizeof(Scalar) * eigen_x.size());
    return x;
  }
  /**
   * @brief Maps an external buffer to the CSC struct.
   * @tparam IndexType The type of indices.
   * @param offset_in Pointer to the column offsets.
   * @param flat_data_in Pointer to the non-zero values.
   * @param flat_row_index_in Pointer to the row indices for non-zero values.
   * @param m_in Number of rows.
   * @param n_in Number of columns.
   * @param nnz Number of non-zero elements.
   */
  template <typename IndexType>
  void map_external_buffer(IndexType* offset_in, Scalar* flat_data_in,
                           IndexType* flat_row_index_in, int m_in,
                           const int n_in, const int nnz) {
    // We have to protect local allocated memory if user is trying to
    // initialise the csc another time. If the memory is handled by external
    // parties, we give green light!
    if (!external_buffer) {
      ASSERT(!initialised, "CSC already initialised");
    }
    m = m_in;
    n = n_in;
    non_zeros = nnz;
    offset = offset_in;
    values = flat_data_in;
    flat_row_index = flat_row_index_in;
    initialised = 1;
    external_buffer = 1;
  }
  /**
   * @brief Creates a CSC matrix from a dense matrix.
   * @param A Pointer to the dense matrix.
   * @param m_in Number of rows.
   * @param n_in Number of columns.
   */
  void create_from_dense(Scalar* A, int m_in, int n_in) {
    m = m_in;
    n = n_in;

    int count = 0;
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        if (A[i * n + j] != 0.0) {
          count++;
        }
      }
    }
    non_zeros = count;

    offset = (int*)malloc(sizeof(int) * (n + 1));
    int scan = 0;
    for (int j = 0; j < n; j++) {
      offset[j] = scan;
      for (int i = 0; i < m; i++) {
        if (A[i * n + j] != 0.0) {
          scan++;
        }
      }
    }
    offset[n] = scan;

    values = (Scalar*)malloc(sizeof(Scalar) * non_zeros);
    int index = 0;
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < m; i++) {
        if (A[i * n + j] != 0.0) {
          values[index] = A[i * n + j];
          index++;
        }
      }
    }

    flat_row_index = (int*)malloc(sizeof(int) * non_zeros);
    index = 0;
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < m; i++) {
        if (A[i * n + j] != 0.0) {
          flat_row_index[index] = i;
          index++;
        }
      }
    }
    initialised = 1;
  }
  /**
   * @brief Creates a diagonal CSC matrix.
   * @param m_in Number of rows.
   * @param n_in Number of columns.
   * @param value The value of the diagonal elements.
   */
  void create_diagonal(int m_in, int n_in, Scalar value) {
    ASSERT(!initialised, "CSC already initialised");
    m = m_in;
    n = n_in;
    non_zeros = n;

    offset = (int*)malloc(sizeof(int) * (n + 1));
    for (int j = 0; j < n + 1; j++) {
      if (j < m) {
        offset[j] = j;
      } else {
        offset[j] = m;
      }
    }

    values = (Scalar*)malloc(sizeof(Scalar) * non_zeros);
    for (int j = 0; j < n; j++) {
      values[j] = value;
    }

    flat_row_index = (int*)malloc(sizeof(int) * non_zeros);
    for (int i = 0; i < n; i++) {
      flat_row_index[i] = i;
    }
    initialised = 1;
  }
  /**
   * @brief Updates the values of the k-th column.
   * @param new_values The new values for the column.
   * @param k The column index to update.
   * @param row_indices The row indices corresponding to the new values.
   * @param size The number of elements in the new values array.
   */
  void update_kth_column(Scalar* new_values, int k, int* row_indices,
                         int size) {
    CSC newA;
    newA.m = m;
    newA.n = n;

    // Compute the new number of nonzeros
    int deltaNonzeros = 0;
    for (int i = 0; i < size; i++) {
      if (new_values[i] != 0.0) {
        deltaNonzeros++;
      }
    }
    deltaNonzeros -= offset[k + 1] - offset[k];

    // set the new number of nonzeros
    newA.non_zeros = non_zeros + deltaNonzeros;

    // Malloc space
    newA.offset = (int*)malloc(sizeof(int) * (n + 1));
    newA.values = (Scalar*)malloc(sizeof(Scalar) * newA.non_zeros);
    newA.flat_row_index = (int*)malloc(sizeof(int) * newA.non_zeros);

    // Copy the offset values before k
    for (int i = 0; i < k + 1; i++) {
      newA.offset[i] = offset[i];
    }

    // Compute the new offset values for k and onwards
    for (int i = k + 1; i < n + 1; i++) {
      newA.offset[i] = offset[i] + deltaNonzeros;
    }

    // Copy the old values and flat_row_index values before k
    for (int i = 0; i < offset[k] + 1; i++) {
      newA.values[i] = values[i];
      newA.flat_row_index[i] = flat_row_index[i];
    }

    // insert the new values into the values and flat_row_index from k
    int index = 0;
    for (int i = 0; i < size; i++) {
      if (new_values[i] != 0.0) {
        newA.values[offset[k] + index] = new_values[i];
        newA.flat_row_index[offset[k] + index] = row_indices[i];
        index++;
      }
    }

    // Copy the old values and flat_row_index values after k
    for (int i = newA.offset[k + 1]; i < newA.non_zeros; i++) {
      newA.values[i] = values[i - deltaNonzeros];
      newA.flat_row_index[i] = flat_row_index[i - deltaNonzeros];
    }

    // swap old with new memory
    destoy();
    this->m = newA.m;
    this->n = newA.n;
    this->initialised = 1;
    this->offset = newA.offset;
    this->non_zeros = newA.non_zeros;
    this->flat_row_index = newA.flat_row_index;
    this->values = newA.values;
  }
  /**
   * @brief Converts the CSC matrix to a dense matrix.
   * @param I The row indices.
   * @param J The column indices.
   * @param n1 Number of rows.
   * @param n2 Number of columns.
   * @return Pointer to the dense matrix.
   */
  Scalar* to_dense(int* I, int* J, int n1, int n2) {
    Scalar* dense = (Scalar*)calloc(n1 * n2, sizeof(Scalar));
    for (int i = 0; i < n1; i++) {
      for (int j = 0; j < n2; j++) {
        for (int l = offset[J[j]]; l < offset[J[j] + 1]; l++) {
          if (I[i] == flat_row_index[l]) {
            dense[i * n2 + j] = values[l];
          }
        }
      }
    }
    return dense;
  }
  /**
   * @brief Prints the CSC matrix data.
   */
  void print() {
    printf("\n\n--------Printing CSC data--------\n");
    printf("m: %d\n", m);
    printf("n: %d\n", n);
    printf("non_zeros: %d\n", non_zeros);
    printf("offset: ");
    for (int i = 0; i < n + 1; i++) {
      printf("%d ", offset[i]);
    }
    printf("\n");
    printf("values: ");
    for (int i = 0; i < non_zeros; i++) {
      printf("%f ", values[i]);
    }
    printf("\n");
    printf("flat_row_index: ");
    for (int i = 0; i < non_zeros; i++) {
      printf("%d ", flat_row_index[i]);
    }
    printf("\n");
  }
};
}  // namespace apsc::LinearAlgebra

#endif
