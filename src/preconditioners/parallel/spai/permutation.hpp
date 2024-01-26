/**
 * @file permutation.hpp
 * @brief Header file containing a method to create permutation matrices for SPAI.
 * @author Kaixi Matteo Chen
 */
#ifndef PERMUTATION_H
#define PERMUTATION_H

#include <assert.hpp>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace apsc::LinearAlgebra {
namespace Preconditioners {
namespace ApproximateInverse {
namespace Utils {
/**
 * \brief Create permutation matrices based on row and column indices.
 *
 * This function creates permutation matrices based on row and column indices.
 * It generates row permutation matrix `Pr` and column permutation matrix `Pc`
 * using the given row and column indices `I` and `J`, respectively. The size
 * of the matrices `Pr` and `Pc` is determined by the dimensions `n1` and `n2`.
 *
 * \tparam Scalar The scalar type of the permutation matrices.
 *
 * \param I Pointer to the row indices.
 * \param J Pointer to the column indices.
 * \param n1 Number of rows in the permutation matrix `Pr`.
 * \param n2 Number of columns in the permutation matrix `Pc`.
 * \param Pr Pointer to store the row permutation matrix `Pr`.
 * \param Pc Pointer to store the column permutation matrix `Pc`.
 *
 * \note The pointers `I`, `J`, `Pr`, and `Pc` must be initialized before calling this function.
 */
template <typename Scalar>
void create_permutation(int* I, int* J, int n1, int n2, Scalar* Pr,
                        Scalar* Pc) {
  ASSERT(Pr != 0 && Pc != 0, "Pc and Pr pointers not initialised");
  ASSERT(I != 0 && J != 0, "I and J pointers not initialised");

  // Create normalized index of I
  int* IIndex = (int*)malloc(n1 * sizeof(int));
  int prevLowest = -1;
  for (int i = 0; i < n1; i++) {
    int currentLowest = INT_MAX;
    for (int j = 0; j < n1; j++) {
      if (I[j] > prevLowest && I[j] < currentLowest) {
        currentLowest = I[j];
        IIndex[j] = i;
      }
    }
    prevLowest = currentLowest;
  }

  // Create row permutation matrix of size n1 x n1
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n1; j++) {
      if (IIndex[j] == i) {
        Pr[i * n1 + j] = static_cast<Scalar>(1);
      } else {
        Pr[i * n1 + j] = static_cast<Scalar>(0);
      }
    }
  }

  // Create normalized index of J
  int* JIndex = (int*)malloc(n2 * sizeof(int));
  prevLowest = -1;
  for (int i = 0; i < n2; i++) {
    int currentLowest = INT_MAX;
    for (int j = 0; j < n2; j++) {
      if (J[j] > prevLowest && J[j] < currentLowest) {
        currentLowest = J[j];
        JIndex[j] = i;
      }
    }
    prevLowest = currentLowest;
  }

  // Create column permutation matrix of size n2 x n2
  for (int i = 0; i < n2; i++) {
    for (int j = 0; j < n2; j++) {
      if (JIndex[j] == i) {
        Pc[i * n2 + j] = static_cast<Scalar>(1);
      } else {
        Pc[i * n2 + j] = static_cast<Scalar>(0);
      }
    }
  }

  // Free memory
  free(IIndex);
  free(JIndex);
}
}  // namespace Utils
}  // namespace ApproximateInverse
}  // namespace Preconditioners
}  // namespace apsc::LinearAlgebra

#endif
