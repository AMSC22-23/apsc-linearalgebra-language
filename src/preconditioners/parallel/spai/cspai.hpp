#ifndef CSPAI_H
#define CSPAI_H

#include <mpi.h>
#include <stdlib.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>

#include "Parallel/Utilities/mpi_utils.hpp"
#include "assert.hpp"
#include "csc.hpp"
#include "least_sqaure_solver.hpp"
#include "update_qr.hpp"

namespace LinearAlgebra {
namespace Preconditioners {
namespace ApproximateInverse {
template <typename Scalar, typename FullMatrix, int DEBUG_MODE = 0>
struct CSC<Scalar> CSPAI(struct CSC<Scalar>* A, Scalar tolerance,
                         int maxIteration, int s) {
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  if constexpr (DEBUG_MODE) {
    if (mpi_rank == 0) {
      printf(
          "------------------------------ SEQUENTIAL SPAI (C) "
          "--------------------------\n");
      printf(
          "running with parameters: tolerance = %f, maxIteration = %d, s = "
          "%d\n",
          tolerance, maxIteration, s);
    }
  }

  // Initialize M and set to diagonal
  ASSERT(A->initialised, "Input matrix not initialised");
  CSC<Scalar> M;
  M.create_diagonal(A->m, A->n, static_cast<Scalar>(1));

  // m_k = column in M
  for (int k = mpi_rank; k < M.n; k += mpi_size) {
    // variables
    int n1 = 0;
    int n2 = 0;
    int iteration = 0;
    Scalar residualNorm = 0.0;

    int* J;
    int* I;
    int* sortedJ = (int*)malloc(sizeof(int) * M.n);
    Scalar* AHat;
    Scalar* Q;
    Scalar* R;
    Scalar* mHat_k;
    Scalar* residual;

    // 1) Find the initial sparsity J of m_k
    // Malloc space for the indeces from offset[k] to offset[k + 1]
    n2 = M.offset[k + 1] - M.offset[k];
    J = (int*)malloc(sizeof(int) * n2);

    // Iterate through row indeces from offset[k] to offset[k + 1] and take all
    // elements from the flat_row_index
    int h = 0;
    for (int i = M.offset[k]; i < M.offset[k + 1]; i++) {
      J[h] = M.flat_row_index[i];
      h++;
    }

    // 2) Compute the row indices I of the corresponding nonzero entries of A(i,
    // J) We initialize I to -1, and the iterate through all elements of J. Then
    // we iterate through the row indeces of A from the offset J[j] to J[j] + 1.
    // If the row index is already in I, we dont do anything, else we add it to
    // I.
    I = (int*)malloc(sizeof(int) * A->m);
    for (int i = 0; i < A->m; i++) {
      I[i] = -1;
    }

    n1 = 0;
    for (int j = 0; j < n2; j++) {
      for (int i = A->offset[J[j]]; i < A->offset[J[j] + 1]; i++) {
        int keep = 1;
        for (int h = 0; h < A->m; h++) {
          if (A->flat_row_index[i] == I[h]) {
            keep = 0;
          }
        }
        if (keep == 1) {
          I[n1] = A->flat_row_index[i];
          n1++;
        }
      }
    }

    if (n1 == 0) {
      n2 = 0;
    }

    // 3) Create Â = A(I, J)
    // We initialize AHat to zeros. Then we iterate through all indeces of J,
    // and iterate through all indeces of I. For each of the indices of I and
    // the indices in the flat_row_index, we check if they match. If they do, we
    // add that element to AHat.
    AHat = A->to_dense(I, J, n1, n2);

    // 4) Do QR decomposition of AHat
    Q = (Scalar*)malloc(sizeof(Scalar) * n1 * n1);
    R = (Scalar*)malloc(sizeof(Scalar) * n1 * n2);

    auto eigen_AHat =
        EigenStructureMap<FullMatrix, Scalar>(AHat, n1, n2).structure();
    // TODO: check for correct factorisation
    Eigen::HouseholderQR<FullMatrix> qr(eigen_AHat);
    FullMatrix eigen_AHatQ = qr.householderQ();
    FullMatrix eigen_AHatR =
        qr.matrixQR().template triangularView<Eigen::Upper>();
    // eigen_B2Q/R will be destructed
    ASSERT(eigen_AHatQ.rows() == n1 && eigen_AHatQ.cols() == n1,
           "eigen_AHatQ rows and cols do not match expected");
    ASSERT(eigen_AHatR.rows() == n1 && eigen_AHatR.cols() == n2,
           "eigen_AHatR rows and cols do not match expected");
    memcpy(Q, eigen_AHatQ.data(), sizeof(Scalar) * n1 * n1);
    memcpy(R, eigen_AHatR.data(), sizeof(Scalar) * n2 * n2);

    // Overwrite AHat
    free(AHat);
    AHat = A->to_dense(I, J, n1, n2);

    // 5) Compute the solution m_k for the least squares problem
    mHat_k = (Scalar*)malloc(n2 * sizeof(Scalar));
    residual = (Scalar*)malloc(A->m * sizeof(Scalar));

    solve_least_square<Scalar, FullMatrix>(A, Q, R, &mHat_k, residual, I, J, n1,
                                           n2, k, &residualNorm);

    // 6) Compute residual = A * mHat_k - e_k
    // Malloc space for residual
    // Do matrix multiplication
    int* IDense = (int*)malloc(A->m * sizeof(int));
    int* JDense = (int*)malloc(A->n * sizeof(int));
    for (int i = 0; i < A->m; i++) {
      IDense[i] = i;
    }
    for (int j = 0; j < A->n; j++) {
      JDense[j] = j;
    }
    Scalar* ADense = A->to_dense(IDense, JDense, A->m, A->n);

    // Compute residual
    for (int i = 0; i < A->m; i++) {
      residual[i] = 0.0;
      for (int j = 0; j < A->n; j++) {
        for (int h = 0; h < n2; h++) {
          if (J[h] == j) {
            residual[i] += ADense[i * A->n + j] * mHat_k[h];
          }
        }
      }
      if (i == k) {
        residual[i] -= 1.0;
      }
    }

    // Compute the norm of the residual
    residualNorm = 0.0;
    for (int i = 0; i < A->m; i++) {
      residualNorm += residual[i] * residual[i];
    }
    residualNorm = sqrt(residualNorm);

    int somethingToBeDone = 1;

    // While norm of residual > tolerance do
    while (residualNorm > tolerance && maxIteration > iteration &&
           somethingToBeDone) {
      iteration++;

      // Variables
      int n1Tilde = 0;
      int n2Tilde = 0;
      int n1Union = 0;
      int n2Union = 0;
      int l = 0;
      int kNotInI = 0;

      Scalar* rhoSq;
      int* ITilde;
      int* IUnion;
      int* JTilde;
      int* JUnion;
      int* L;
      int* keepArray;
      int* smallestIndices;
      int* smallestJTilde;

      // 7) Set L to the set of indices where r(l) != 0
      // Count the numbers of nonzeros in residual
      for (int i = 0; i < A->m; i++) {
        if (residual[i] != 0.0) {
          l++;
        } else if (k == i) {
          kNotInI = 1;
        }
      }

      // Check if k is in I
      for (int i = 0; i < n1; i++) {
        if (k == I[i]) {
          kNotInI = 0;
        }
      }

      // increment l if k is not in I
      if (kNotInI) {
        l++;
      }

      // Malloc space for L and fill it with the indices
      L = (int*)malloc(sizeof(int) * l);

      int index = 0;
      for (int i = 0; i < A->m; i++) {
        if (residual[i] != 0.0 || (kNotInI && i == k)) {
          L[index] = i;
          index++;
        }
      }

      // 8) Set JTilde to the set of columns of A corresponding to the indices
      // in L that are not already in J Check what indeces we should keep
      keepArray = (int*)malloc(A->n * sizeof(int));
      // set all to 0
      for (int i = 0; i < A->n; i++) {
        keepArray[i] = 0;
      }

      for (int i = 0; i < A->n; i++) {
        for (int j = 0; j < l; j++) {
          for (int h = A->offset[i]; h < A->offset[i + 1]; h++) {
            if (L[j] == A->flat_row_index[h]) {
              keepArray[i] = 1;
            }
          }
        }
      }

      // Remove the indeces that are already in J
      for (int i = 0; i < n2; i++) {
        keepArray[J[i]] = 0;
      }

      // Compute the length of JTilde
      n2Tilde = 0;
      for (int i = 0; i < A->n; i++) {
        if (keepArray[i] == 1) {
          n2Tilde++;
        }
      }

      // Malloc space for JTilde
      JTilde = (int*)malloc(sizeof(int) * n2Tilde);

      // Fill JTilde
      index = 0;
      for (int i = 0; i < A->n; i++) {
        if (keepArray[i] == 1) {
          JTilde[index] = i;
          index++;
        }
      }

      // 9) For each j in JTilde, solve the minimization problem
      // Malloc space for rhoSq
      rhoSq = (Scalar*)malloc(sizeof(Scalar) * n2Tilde);
      for (int i = 0; i < n2Tilde; i++) {
        Scalar rTAe_j = 0.0;  // r^T * A(.,j)
        for (int j = A->offset[JTilde[i]]; j < A->offset[JTilde[i] + 1]; j++) {
          rTAe_j += A->values[j] * residual[A->flat_row_index[j]];
        }

        Scalar Ae_jNorm = 0.0;
        for (int j = A->offset[JTilde[i]]; j < A->offset[JTilde[i] + 1]; j++) {
          Ae_jNorm += A->values[j] * A->values[j];
        }
        Ae_jNorm = sqrt(Ae_jNorm);

        rhoSq[i] = residualNorm * residualNorm -
                   (rTAe_j * rTAe_j) / (Ae_jNorm * Ae_jNorm);
      }

      // 10) Find the s indeces of the column with the smallest rhoSq
      int newN2Tilde = n2Tilde;
      if (s < newN2Tilde) {
        newN2Tilde = s;
      }
      smallestIndices = (int*)malloc(sizeof(int) * newN2Tilde);

      for (int i = 0; i < newN2Tilde; i++) {
        smallestIndices[i] = -1;
      }

      // We iterate through rhoSq and find the smallest indeces.
      // First, we set the first s indeces to the first s indeces of JTilde
      // then if we find a smaller rhoSq, we shift the indeces to the right
      // we insert the index of JTilde with the rhoSq smaller than the current
      // smallest elements smallestIndices then contain the indeces of JTIlde
      // corresponding to the smallest values of rhoSq
      for (int i = 0; i < n2Tilde; i++) {
        for (int j = 0; j < newN2Tilde; j++) {
          if (smallestIndices[j] == -1) {
            smallestIndices[j] = i;
            break;
          } else if (rhoSq[i] < rhoSq[smallestIndices[j]]) {
            for (int h = newN2Tilde - 1; h > j; h--) {
              smallestIndices[h] = smallestIndices[h - 1];
            }

            smallestIndices[j] = i;
            break;
          }
        }
      }

      smallestJTilde = (int*)malloc(sizeof(int) * newN2Tilde);
      for (int i = 0; i < newN2Tilde; i++) {
        smallestJTilde[i] = JTilde[smallestIndices[i]];
      }

      free(JTilde);
      JTilde = (int*)malloc(sizeof(int) * newN2Tilde);
      for (int i = 0; i < newN2Tilde; i++) {
        JTilde[i] = smallestJTilde[i];
      }

      // 11) Determine the new indices Î
      // Denote by ITilde the new rows, which corresponds to the nonzero rows of
      // A(:, J union JTilde) not contained in I yet
      n2Tilde = newN2Tilde;
      n2Union = n2 + n2Tilde;
      JUnion = (int*)malloc(sizeof(int) * n2Union);
      for (int i = 0; i < n2; i++) {
        JUnion[i] = J[i];
      }
      for (int i = 0; i < n2Tilde; i++) {
        JUnion[n2 + i] = JTilde[i];
      }

      ITilde = (int*)malloc(sizeof(int) * A->m);
      for (int i = 0; i < A->m; i++) {
        ITilde[i] = -1;
      }

      n1Tilde = 0;
      for (int j = 0; j < n2Union; j++) {
        for (int i = A->offset[JUnion[j]]; i < A->offset[JUnion[j] + 1]; i++) {
          int keep = 1;
          for (int h = 0; h < n1; h++) {
            if (A->flat_row_index[i] == I[h] || A->flat_row_index[i] == ITilde[h]) {
              keep = 0;
            }
          }
          if (keep == 1) {
            ITilde[n1Tilde] = A->flat_row_index[i];
            n1Tilde++;
          }
        }
      }

      // 12) Make I U ITilde and J U JTilde
      // Make union of I and ITilde
      n1Union = n1 + n1Tilde;
      IUnion = (int*)malloc(sizeof(int) * (n1 + n1Tilde));
      for (int i = 0; i < n1; i++) {
        IUnion[i] = I[i];
      }
      for (int i = 0; i < n1Tilde; i++) {
        IUnion[n1 + i] = ITilde[i];
      }

      // 13) Update the QR factorization of A(IUnion, JUnion)
      update_QR<Scalar, FullMatrix>(A, &AHat, &Q, &R, &I, &J, &sortedJ, ITilde,
                                    JTilde, IUnion, JUnion, n1, n2, n1Tilde,
                                    n2Tilde, n1Union, n2Union, &mHat_k,
                                    residual, &residualNorm, k);

      n1 = n1Union;
      n2 = n2Union;

      // free memory
      free(ITilde);
      free(IUnion);
      free(JTilde);
      free(JUnion);
      free(L);
      free(keepArray);
      free(rhoSq);
      free(smallestJTilde);
      free(smallestIndices);
    }

    // 16) Set m_k(J) = mHat_k
    // Update kth column of M and columns computed by siblings
    if (mpi_rank == 0) {
      M.update_kth_column(mHat_k, k, sortedJ, n2);
      if constexpr (DEBUG_MODE) {
        printf("SPAI: updated %d column\n", k);
      }
      for (int i = 1; i < mpi_size; i++) {
        int recv_col = k + i;
        if (recv_col >= A->n) {
          break;
        }
        // receive mHat_k
        MPI_Status status;
        MPI_Probe(i, (100 * i), MPI_COMM_WORLD, &status);
        int recv_n2;
        MPI_Get_count(&status, MPI_DOUBLE, &recv_n2);
        Scalar* recv_mHat_k = (Scalar*)calloc(recv_n2, sizeof(Scalar));
        int* recv_sortedJ = (int*)calloc(recv_n2, sizeof(int));
        MPI_Recv(recv_mHat_k, recv_n2, mpi_typeof(Scalar{}), status.MPI_SOURCE,
                 (100 * i), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(recv_sortedJ, recv_n2, MPI_INT, status.MPI_SOURCE, (101 * i),
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        M.update_kth_column(recv_mHat_k, recv_col, recv_sortedJ, recv_n2);
        if constexpr (DEBUG_MODE) {
          printf("SPAI: updated %d column\n", recv_col);
        }
        free(recv_mHat_k);
        free(recv_sortedJ);
      }
    } else {
      MPI_Send(mHat_k, n2, MPI_DOUBLE, 0, 100 * mpi_rank, MPI_COMM_WORLD);
      MPI_Send(sortedJ, n2, MPI_INT, 0, 101 * mpi_rank, MPI_COMM_WORLD);
    }

    // Free memory
    free(I);
    free(J);
    free(sortedJ);
    free(AHat);
    free(Q);
    free(R);
    free(mHat_k);
    free(residual);
    free(ADense);
    free(IDense);
    free(JDense);
  }
  return M;
}
}  // namespace ApproximateInverse
}  // namespace Preconditioners
}  // namespace LinearAlgebra

#endif
