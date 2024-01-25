#ifndef LEAST_SQUARE_PROBLEM_HPP
#define LEAST_SQUARE_PROBLEM_HPP

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <EigenStructureMap.hpp>

#include "assert.hpp"
#include "csc.hpp"

namespace LinearAlgebra {
namespace Preconditioners {
namespace ApproximateInverse {
namespace Utils {
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
}  // namespace LinearAlgebra

#endif
