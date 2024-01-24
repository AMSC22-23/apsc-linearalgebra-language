#ifndef CSC_HPP
#define CSC_HPP

#include <stdint.h>
#include <mpi.h>

#include <Eigen/Sparse>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "EigenStructureMap.hpp"
#include "Parallel/Utilities/mpi_utils.hpp"
#include "assert.hpp"

template <typename Scalar>
struct CSC {
  int m = 0;
  int n = 0;
  int non_zeros = 0;
  int* offset = 0;
  Scalar* values = 0;
  int* flat_row_index = 0;
  uint8_t initialised = 0;
  uint8_t external_buffer = 0;

  CSC() = default;

  ~CSC() { destoy(); }

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

  void setup_mpi(const int master_rank, int current_rank) {
    if (current_rank == master_rank) {
      ASSERT(initialised, "CSC not initialised" << std::endl);
    }
    std::cout << current_rank << ": 1\n";
    MPI_Bcast(&non_zeros, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::cout << current_rank << ": 2\n";
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::cout << "m: " << m << " n " << n << " non_zeros: " << non_zeros << std::endl;
    if (current_rank != master_rank) {
      offset = (int*)malloc((n+1) * sizeof(int)); 
      flat_row_index = (int*)malloc((non_zeros) * sizeof(int)); 
      values = (Scalar*)malloc((non_zeros) * sizeof(Scalar)); 
      ASSERT(offset, "offset malloc failed" << std::endl);
      ASSERT(flat_row_index, "flat_row_index malloc failed" << std::endl);
      ASSERT(values, "values malloc failed" << std::endl);
    }
    MPI_Bcast(offset, n+1, MPI_INT, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(flat_row_index, non_zeros, MPI_INT, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(values, non_zeros, mpi_typeof(Scalar{}), master_rank, MPI_COMM_WORLD);
    initialised = 1;
  }

  template<typename EigenMatrixType>
  auto to_eigen(const std::size_t size) const {
    return
        EigenStructureMap<EigenMatrixType, double, CSC<Scalar>>::create_map(
            size, size, non_zeros, offset, flat_row_index, values).structure();
  }

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

  // remember to use a column major ordering matrix!
  template <typename IndexType>
  void map_external_buffer(IndexType* offset_in, Scalar* flat_data_in,
                           IndexType* flat_row_index_in, int m_in,
                           const int n_in, const int nnz) {
    ASSERT(!initialised, "CSC already initialised");
    m = m_in;
    n = n_in;
    non_zeros = nnz;
    offset = offset_in;
    values = flat_data_in;
    flat_row_index = flat_row_index_in;
    initialised = 1;
    external_buffer = 1;
  }

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

  void update_kth_column(Scalar* newVaules, int k, int* J, int n2) {
    CSC newA;
    newA.m = m;
    newA.n = n;

    // Compute the new number of nonzeros
    int deltaNonzeros = 0;
    for (int i = 0; i < n2; i++) {
      if (newVaules[i] != 0.0) {
        deltaNonzeros++;
      }
    }
    deltaNonzeros -= offset[k + 1] - offset[k];

    // set the new number of nonzeros
    newA.non_zeros = non_zeros + deltaNonzeros;

    // Malloc space for the new offset array
    newA.offset = (int*)malloc(sizeof(int) * (n + 1));

    // Copy the offset values before k
    for (int i = 0; i < k + 1; i++) {
      newA.offset[i] = offset[i];
    }

    // Compute the new offset values for k and onwards
    for (int i = k + 1; i < n + 1; i++) {
      newA.offset[i] = offset[i] + deltaNonzeros;
    }

    // Malloc space
    newA.values = (Scalar*)malloc(sizeof(Scalar) * newA.non_zeros);
    newA.flat_row_index = (int*)malloc(sizeof(int) * newA.non_zeros);

    // Copy the old values and flat_row_index values before k
    for (int i = 0; i < offset[k] + 1; i++) {
      newA.values[i] = values[i];
      newA.flat_row_index[i] = flat_row_index[i];
    }

    // insert the new values into the values and flat_row_index from k
    int index = 0;
    for (int i = 0; i < n2; i++) {
      if (newVaules[i] != 0.0) {
        newA.values[offset[k] + index] = newVaules[i];
        newA.flat_row_index[offset[k] + index] = J[i];
        index++;
      }
    }

    // Copy the old values and flat_row_index values after k
    for (int i = newA.offset[k + 1]; i < newA.non_zeros; i++) {
      newA.values[i] = values[i - deltaNonzeros];
      newA.flat_row_index[i] = flat_row_index[i - deltaNonzeros];
    }

    // swap
    destoy();
    this->m = newA.m;
    this->n = newA.n;
    this->initialised = 1;
    this->offset = newA.offset;
    this->non_zeros = newA.non_zeros;
    this->flat_row_index = newA.flat_row_index;
    this->values = newA.values;
  }

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

#endif
