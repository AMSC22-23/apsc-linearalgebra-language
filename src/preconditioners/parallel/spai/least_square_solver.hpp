/**
 * @file least_square_solver.hpp
 * @brief Header file containing a least square solver.
 * @author Kaixi Matteo Chen
 */
#ifndef LEAST_SQUARE_PROBLEM_HPP
#define LEAST_SQUARE_PROBLEM_HPP

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <EigenStructureMap.hpp>

#include "CSC.hpp"
#include "assert.hpp"

namespace apsc::LinearAlgebra {
namespace Preconditioners {
namespace ApproximateInverse {
namespace Utils {
/**
 * \brief Solves a least square problem using the given QR factorization.
 *
 * This function computes the solution to a least square problem using the given QR
 * factorization. It computes the solution for the k-th column of the matrix A,
 * where A is represented in compressed sparse column (CSC) format.
 *
 * \tparam Scalar The scalar type of the matrix elements.
 * \tparam FullMatrix The Eigen-compatible matrix type representing the full matrix.
 *
 * \param A Pointer to the CSC representation of the matrix A.
 * \param Q Pointer to the matrix Q from the QR factorization.
 * \param R Pointer to the matrix R from the QR factorization.
 * \param mHat_k Pointer to the output array to store the computed solution.
 * \param residual Pointer to the residual vector.
 * \param I Array representing the row indices of non-zero elements in the matrix A.
 * \param J Array representing the column indices of non-zero elements in the matrix A.
 * \param n1 Number of rows in the matrix A.
 * \param n2 Number of columns in the matrix A.
 * \param k Index of the column for which to compute the solution.
 * \param residualNorm Pointer to store the residual norm after computation.
 */
template <typename Scalar, typename FullMatrix>
void solve_least_square(CSC<Scalar> *A, Scalar *Q, Scalar *R, Scalar **mHat_k,
                        Scalar *residual, int *I, int *J, int n1, int n2, int k,
                        Scalar *residualNorm) {
  // 5.1) Compute cHat = Q^T * ê_k
  // Make e_k and set index k to 1.0
  Scalar *e_k = (Scalar *)malloc(n1 * sizeof(Scalar));
  for (int i = 0; i < n1; i++) {
    e_k[i] = 0.0;
    if (k == I[i]) {
      e_k[i] = 1.0;
    }
  }

  // Malloc space for cHat and do matrix multiplication
  Scalar *cHat = (Scalar *)malloc(n2 * sizeof(Scalar));
  for (int j = 0; j < n2; j++) {
    cHat[j] = 0.0;
    for (int i = 0; i < n1; i++) {
      cHat[j] += Q[i * n1 + j] * e_k[i];
    }
  }

  // 5.2) Make the inverse of R of size n2 x n2
  Scalar *invR = (Scalar *)malloc(n2 * n2 * sizeof(Scalar));

  auto eigen_R = EigenStructureMap<FullMatrix, Scalar>(R, n2, n2).structure();
  FullMatrix eigen_R_inverse = eigen_R.inverse();
  ASSERT(eigen_R_inverse.allFinite(), "Failed to invert R" << std::endl
                                                           << R << std::endl);
  // eigen_R_inverse will desctructor will be called
  memcpy(invR, eigen_R_inverse.data(), sizeof(Scalar) * n2 * n2);

  // 5.3) Compute mHat_k = R^-1 * cHat
  for (int i = 0; i < n2; i++) {
    (*mHat_k)[i] = 0.0;
    for (int j = 0; j < n2; j++) {
      (*mHat_k)[i] += invR[i * n2 + j] * cHat[j];
    }
  }

  // Free memory
  free(e_k);
  free(cHat);
  free(invR);
}
}  // namespace Utils
}  // namespace ApproximateInverse
}  // namespace Preconditioners
}  // namespace apsc::LinearAlgebra

#endif
