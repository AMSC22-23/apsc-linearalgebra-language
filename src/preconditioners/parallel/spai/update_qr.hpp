#ifndef UPDATE_QR_HPP
#define UPDATE_QR_HPP

#include <stdlib.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "assert.hpp"
#include "csc.hpp"
#include "least_sqaure_solver.hpp"
#include "permutation.hpp"

namespace LinearAlgebra
{
namespace Preconditioners
{
namespace ApproximateInverse
{
namespace Utils
{
template <typename Scalar, typename FullMatrix>
int update_QR(struct CSC<Scalar> *A, Scalar **AHat, Scalar **Q, Scalar **R,
              int **I, int **J, int **sortedJ, int *ITilde, int *JTilde,
              int *IUnion, int *JUnion, int n1, int n2, int n1Tilde,
              int n2Tilde, int n1Union, int n2Union, Scalar **m_kOut,
              Scalar *residual, Scalar *residualNorm, int k) {
  // 13.1) Create A(I, JTilde) and A(ITilde, JTilde)
  Scalar *AIJTilde = A->to_dense((*I), JTilde, n1, n2Tilde);
  Scalar *AITildeJTilde = A->to_dense(ITilde, JTilde, n1Tilde, n2Tilde);

  // Create permutation matrices Pr and Pc
  int *Pr = (int *)malloc(n1Union * n1Union * sizeof(int));
  int *Pc = (int *)malloc(n2Union * n2Union * sizeof(int));
  create_permutation<int>(IUnion, JUnion, n1Union, n2Union, Pr, Pc);

  // 13.2) ABreve of size n1 x n2Tilde = Q^T * AIJTilde
  Scalar *ABreve = (Scalar *)malloc(n1 * n2Tilde * sizeof(Scalar));
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2Tilde; j++) {
      ABreve[i * n2Tilde + j] = 0;
      for (int k = 0; k < n1; k++) {
        ABreve[i * n2Tilde + j] += (*Q)[k * n1 + i] * AIJTilde[k * n2Tilde + j];
      }
    }
  }

  // 13.3) Compute B1 = ABreve[0 : n2, 0 : n2Tilde]
  Scalar *B1 = (Scalar *)malloc(n2 * n2Tilde * sizeof(Scalar));
  for (int i = 0; i < n2; i++) {
    for (int j = 0; j < n2Tilde; j++) {
      B1[i * n2Tilde + j] = ABreve[i * n2Tilde + j];
    }
  }

  // 13.4) Compute B2 = ABreve[n2 + 1 : n1, 0 : n2Tilde] + AITildeJTilde
  Scalar *B2;
  int B2_rows = 0, B2_cols = 0;
  if (n1 - n2 < 0) {
    B2_rows = n1Tilde;
    B2_cols = n2Tilde;
    B2 = (Scalar *)malloc(n1Tilde * n2Tilde * sizeof(Scalar));
    for (int i = 0; i < n1Tilde; i++) {
      for (int j = 0; j < n2Tilde; j++) {
        B2[i * n2Tilde + j] = AITildeJTilde[i * n2Tilde + j];
      }
    }
  } else {
    B2_rows = n1Union - n2;
    B2_cols = n2Tilde;
    B2 = (Scalar *)malloc((n1Union - n2) * n2Tilde * sizeof(Scalar));
    for (int i = 0; i < n1 - n2; i++) {
      for (int j = 0; j < n2Tilde; j++) {
        B2[i * n2Tilde + j] = ABreve[(n2 + i) * n2Tilde + j];
      }
    }

    for (int i = 0; i < n1Tilde; i++) {
      for (int j = 0; j < n2Tilde; j++) {
        B2[(n1 - n2 + i) * n2Tilde + j] = AITildeJTilde[i * n2Tilde + j];
      }
    }
  }

  // 13.5) Do QR factorization of B2
  ASSERT(B2_rows == (n1Union - n2) && B2_cols == n2Tilde,
         "13.5) B2 size error");
  Scalar *B2Q =
      (Scalar *)malloc((n1Union - n2) * (n1Union - n2) * sizeof(Scalar));
  Scalar *B2R = (Scalar *)malloc((n1Union - n2) * n2Tilde * sizeof(Scalar));
  auto eigen_B2 =
      EigenStructureMap<Eigen::MatrixXd, Scalar>(B2, B2_rows, B2_cols)
          .structure();
  // TODO: check for correct factorisation
  Eigen::HouseholderQR<FullMatrix> qr(eigen_B2);
  // TODO: use eigen dynamic
  Eigen::MatrixXd eigen_B2Q = qr.householderQ();
  Eigen::MatrixXd eigen_B2R =
      qr.matrixQR().template triangularView<Eigen::Upper>();
  // eigen_B2Q/R will be destructed
  ASSERT(
      eigen_B2Q.rows() == (n1Union - n2) && eigen_B2Q.cols() == (n1Union - n2),
      "eigen_B2Q rows and cols do not match expected");
  ASSERT(eigen_B2R.rows() == (n1Union - n2) && eigen_B2R.cols() == (n2Tilde),
         "eigen_B2R rows and cols do not match expected");
  memcpy(B2Q, eigen_B2Q.data(),
         sizeof(Scalar) * (n1Union - n2) * (n1Union - n2));
  memcpy(B2R, eigen_B2R.data(), sizeof(Scalar) * (n1Union - n2) * n2Tilde);

  // 13.6) Compute Q_B and R_B from algorithm 17
  // Make first matrix with Q in the upper left and identity in the lower right
  // of size n1Union x n1Union
  Scalar *firstMatrix = (Scalar *)malloc(n1Union * n1Union * sizeof(Scalar));
  for (int i = 0; i < n1Union; i++) {
    for (int j = 0; j < n1Union; j++) {
      firstMatrix[i * n1Union + j] = 0.0;
    }
  }
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n1; j++) {
      firstMatrix[i * n1Union + j] = (*Q)[i * n1 + j];
    }
  }
  for (int i = 0; i < n1Tilde; i++) {
    firstMatrix[(n1 + i) * n1Union + n1 + i] = 1.0;
  }

  // Make second matrix with identity in the upper left corner and B2Q in the
  // lower right corner of size n1Union x n1Union
  Scalar *secondMatrix = (Scalar *)malloc(n1Union * n1Union * sizeof(Scalar));
  for (int i = 0; i < n1Union; i++) {
    for (int j = 0; j < n1Union; j++) {
      secondMatrix[i * n1Union + j] = 0.0;
    }
  }
  for (int i = 0; i < n2; i++) {
    secondMatrix[i * n1Union + i] = 1.0;
  }
  for (int i = 0; i < n1Union - n2; i++) {
    for (int j = 0; j < n1Union - n2; j++) {
      secondMatrix[(n2 + i) * n1Union + n2 + j] = B2Q[i * (n1Union - n2) + j];
    }
  }

  // Compute unsortedQ = firstMatrix * secondMatrix
  Scalar *unsortedQ = (Scalar *)malloc(n1Union * n1Union * sizeof(Scalar));
  for (int i = 0; i < n1Union; i++) {
    for (int j = 0; j < n1Union; j++) {
      unsortedQ[i * n1Union + j] = 0.0;
      for (int k = 0; k < n1Union; k++) {
        unsortedQ[i * n1Union + j] +=
            firstMatrix[i * n1Union + k] * secondMatrix[k * n1Union + j];
      }
    }
  }

  // Make unsortedR with R in the top left corner, B1 in the top right corner
  // and B2R under B1 of size n1Union x n2Union
  Scalar *unsortedR = (Scalar *)malloc(n1Union * n2Union * sizeof(Scalar));
  for (int i = 0; i < n1Union; i++) {
    for (int j = 0; j < n2Union; j++) {
      unsortedR[i * n2Union + j] = 0.0;
    }
  }

  for (int i = 0; i < n2; i++) {
    for (int j = 0; j < n2; j++) {
      unsortedR[i * n2Union + j] = (*R)[i * n2 + j];
    }
  }

  for (int i = 0; i < n2; i++) {
    for (int j = 0; j < n2Tilde; j++) {
      unsortedR[i * n2Union + n2 + j] = B1[i * n2Tilde + j];
    }
  }

  for (int i = 0; i < n1Union - n2; i++) {
    for (int j = 0; j < n2Tilde; j++) {
      unsortedR[(n2 + i) * n2Union + n2 + j] = B2R[i * n2Tilde + j];
    }
  }

  if (m_kOut != 0) {
    free(*m_kOut);
  }
  (*m_kOut) = (Scalar *)malloc(n2Union * sizeof(Scalar));

  // 13.7) Compute the new solution m_k for the least squares problem
  solve_least_square<Scalar, FullMatrix>(A, unsortedQ, unsortedR, m_kOut,
                                         residual, IUnion, JUnion, n1Union,
                                         n2Union, k, residualNorm);

  Scalar *tempM_k = (Scalar *)malloc(n2Union * sizeof(Scalar));
  memcpy(tempM_k, (*m_kOut), n2Union * sizeof(Scalar));

  free(*m_kOut);
  (*m_kOut) = (Scalar *)malloc(n2Union * sizeof(Scalar));

  // Compute m_KOut = Pc * tempM_k
  for (int i = 0; i < n2Union; i++) {
    (*m_kOut)[i] = 0.0;
    for (int j = 0; j < n2Union; j++) {
      (*m_kOut)[i] += Pc[i * n2Union + j] * tempM_k[j];
    }
  }

  // 14) Compute residual = A * mHat_k - e_k
  // Malloc space for residual
  // Do matrix multiplication
  int *IDense = (int *)malloc(A->m * sizeof(int));
  int *JDense = (int *)malloc(A->n * sizeof(int));
  for (int i = 0; i < A->m; i++) {
    IDense[i] = i;
  }
  for (int j = 0; j < A->n; j++) {
    JDense[j] = j;
  }

  // Set I and J to IUnion and JUnion
  free(*I);
  free(*J);
  (*I) = (int *)malloc(n1Union * sizeof(int));
  (*J) = (int *)malloc(n2Union * sizeof(int));
  for (int i = 0; i < n1Union; i++) {
    (*I)[i] = IUnion[i];
  }
  for (int i = 0; i < n2Union; i++) {
    (*J)[i] = JUnion[i];
  }

  // Set sortedJ to Pc * J
  free(*sortedJ);
  (*sortedJ) = (int *)malloc(n2Union * sizeof(int));
  for (int i = 0; i < n2Union; i++) {
    (*sortedJ)[i] = 0;
    for (int j = 0; j < n2Union; j++) {
      (*sortedJ)[i] += Pc[i * n2Union + j] * JUnion[j];
    }
  }

  Scalar *ADense = A->to_dense(IDense, JDense, A->m, A->n);

  // 14) Compute residual
  for (int i = 0; i < A->m; i++) {
    residual[i] = 0.0;
    for (int j = 0; j < A->n; j++) {
      for (int h = 0; h < n2Union; h++) {
        if ((*sortedJ)[h] == j) {
          residual[i] += ADense[i * A->n + j] * (*m_kOut)[h];
        }
      }
    }
    if (i == k) {
      residual[i] -= 1.0;
    }
  }

  // Compute the norm of the residual
  *residualNorm = 0.0;
  for (int i = 0; i < A->m; i++) {
    *residualNorm += residual[i] * residual[i];
  }
  *residualNorm = sqrt(*residualNorm);

  // set Q and R to unsortedQ and unsortedR
  free(*Q);
  (*Q) = (Scalar *)malloc(n1Union * n1Union * sizeof(Scalar));
  free(*R);
  (*R) = (Scalar *)malloc(n1Union * n2Union * sizeof(Scalar));
  for (int i = 0; i < n1Union; i++) {
    for (int j = 0; j < n1Union; j++) {
      (*Q)[i * n1Union + j] = unsortedQ[i * n1Union + j];
    }
  }
  for (int i = 0; i < n1Union; i++) {
    for (int j = 0; j < n2Union; j++) {
      (*R)[i * n2Union + j] = unsortedR[i * n2Union + j];
    }
  }

  // Free memory
  free(ADense);
  free(AIJTilde);
  free(AITildeJTilde);
  free(ABreve);
  free(Pr);
  free(Pc);
  free(B1);
  free(B2);
  free(B2Q);
  free(B2R);
  free(firstMatrix);
  free(secondMatrix);
  free(unsortedQ);
  free(unsortedR);
  free(tempM_k);
  free(IDense);
  free(JDense);

  return 0;
}
}
}
}
}

#endif
