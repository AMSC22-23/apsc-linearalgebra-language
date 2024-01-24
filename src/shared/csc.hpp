#ifndef CSC_HPP
#define CSC_HPP

#include <stdint.h>

#include <Eigen/Sparse>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "EigenStructureMap.hpp"
#include "assert.hpp"

template <typename Scalar>
struct CSC {
  int m = 0;
  int n = 0;
  int countNonZero = 0;
  int* offset = 0;
  Scalar* flatData = 0;
  int* flatRowIndex = 0;
  uint8_t initialised = 0;
  uint8_t external_buffer = 0;

  CSC() = default;

  ~CSC() { destoy(); }

  void destoy() {
    if (!external_buffer && initialised) {
      free(offset);
      offset = 0;
      free(flatData);
      flatData = 0;
      free(flatRowIndex);
      flatRowIndex = 0;
    }
    initialised = 0;
    m = 0;
    n = 0;
    countNonZero = 0;
    external_buffer = 0;
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
                          CSC<Scalar>>::create_map(m, n, countNonZero, offset,
                                                   flatRowIndex, flatData)
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
                          CSC<Scalar>>::create_map(m, n, countNonZero, offset,
                                                   flatRowIndex, flatData)
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
    countNonZero = nnz;
    offset = offset_in;
    flatData = flat_data_in;
    flatRowIndex = flat_row_index_in;
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
    countNonZero = count;

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

    flatData = (Scalar*)malloc(sizeof(Scalar) * countNonZero);
    int index = 0;
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < m; i++) {
        if (A[i * n + j] != 0.0) {
          flatData[index] = A[i * n + j];
          index++;
        }
      }
    }

    flatRowIndex = (int*)malloc(sizeof(int) * countNonZero);
    index = 0;
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < m; i++) {
        if (A[i * n + j] != 0.0) {
          flatRowIndex[index] = i;
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
    countNonZero = n;

    offset = (int*)malloc(sizeof(int) * (n + 1));
    for (int j = 0; j < n + 1; j++) {
      if (j < m) {
        offset[j] = j;
      } else {
        offset[j] = m;
      }
    }

    flatData = (Scalar*)malloc(sizeof(Scalar) * countNonZero);
    for (int j = 0; j < n; j++) {
      flatData[j] = value;
    }

    flatRowIndex = (int*)malloc(sizeof(int) * countNonZero);
    for (int i = 0; i < n; i++) {
      flatRowIndex[i] = i;
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
    newA.countNonZero = countNonZero + deltaNonzeros;

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
    newA.flatData = (Scalar*)malloc(sizeof(Scalar) * newA.countNonZero);
    newA.flatRowIndex = (int*)malloc(sizeof(int) * newA.countNonZero);

    // Copy the old flatData and flatRowIndex values before k
    for (int i = 0; i < offset[k] + 1; i++) {
      newA.flatData[i] = flatData[i];
      newA.flatRowIndex[i] = flatRowIndex[i];
    }

    // insert the new values into the flatData and flatRowIndex from k
    int index = 0;
    for (int i = 0; i < n2; i++) {
      if (newVaules[i] != 0.0) {
        newA.flatData[offset[k] + index] = newVaules[i];
        newA.flatRowIndex[offset[k] + index] = J[i];
        index++;
      }
    }

    // Copy the old flatData and flatRowIndex values after k
    for (int i = newA.offset[k + 1]; i < newA.countNonZero; i++) {
      newA.flatData[i] = flatData[i - deltaNonzeros];
      newA.flatRowIndex[i] = flatRowIndex[i - deltaNonzeros];
    }

    // swap
    destoy();
    this->m = newA.m;
    this->n = newA.n;
    this->initialised = 1;
    this->offset = newA.offset;
    this->countNonZero = newA.countNonZero;
    this->flatRowIndex = newA.flatRowIndex;
    this->flatData = newA.flatData;
  }

  Scalar* to_dense(int* I, int* J, int n1, int n2) {
    Scalar* dense = (Scalar*)calloc(n1 * n2, sizeof(Scalar));
    for (int i = 0; i < n1; i++) {
      for (int j = 0; j < n2; j++) {
        for (int l = offset[J[j]]; l < offset[J[j] + 1]; l++) {
          if (I[i] == flatRowIndex[l]) {
            dense[i * n2 + j] = flatData[l];
          }
        }
      }
    }
    return dense;
  }

  static struct CSC* multiply_two_csc(CSC* A, CSC* B) {
    CSC* C = (CSC*)malloc(sizeof(CSC));
    C->m = A->m;
    C->n = B->n;
    C->countNonZero = 0;
    C->offset = (int*)malloc(sizeof(int) * (C->n + 1));
    C->offset[0] = 0;
    C->flatData =
        (Scalar*)malloc(sizeof(Scalar) * (A->countNonZero + B->countNonZero));
    C->flatRowIndex =
        (int*)malloc(sizeof(int) * (A->countNonZero + B->countNonZero));

    for (int j = 0; j < C->n; j++) {
      int countNonZero = 0;
      for (int i = 0; i < C->m; i++) {
        Scalar sum = 0.0;
        for (int l = A->offset[i]; l < A->offset[i + 1]; l++) {
          for (int k = B->offset[j]; k < B->offset[j + 1]; k++) {
            if (A->flatRowIndex[l] == B->flatRowIndex[k]) {
              sum += A->flatData[l] * B->flatData[k];
            }
          }
        }
        if (sum != 0.0) {
          C->flatData[C->countNonZero] = sum;
          C->flatRowIndex[C->countNonZero] = i;
          C->countNonZero++;
          countNonZero++;
        }
      }
      C->offset[j + 1] = C->offset[j] + countNonZero;
    }
    return C;
  }

  void print() {
    printf("\n\n--------Printing CSC data--------\n");
    printf("m: %d\n", m);
    printf("n: %d\n", n);
    printf("countNonZero: %d\n", countNonZero);
    printf("offset: ");
    for (int i = 0; i < n + 1; i++) {
      printf("%d ", offset[i]);
    }
    printf("\n");
    printf("flatData: ");
    for (int i = 0; i < countNonZero; i++) {
      printf("%f ", flatData[i]);
    }
    printf("\n");
    printf("flatRowIndex: ");
    for (int i = 0; i < countNonZero; i++) {
      printf("%d ", flatRowIndex[i]);
    }
    printf("\n");
  }
};

#endif
