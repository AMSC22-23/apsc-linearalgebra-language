#ifndef SPAI_HPP
#define SPAI_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <assert.hpp>
#include <climits>
#include <cmath>
#include <numeric>
#include <vector>

#include "utils.hpp"
namespace LinearAlgebra {
namespace Preconditioner {

template <typename Scalar, typename SparseMatrix, typename FullMatrix,
          typename Vector>
class SPAI {
 public:
  SPAI() {}

  SPAI(SparseMatrix &A) { init(A); }

  SparseMatrix &get_m() { return M; }

  void init(SparseMatrix &A) {
    ASSERT(A.rows() == A.cols(), "A must be a square matrix");
    M.resize(A.rows(), A.cols());

    // set M to identity as initial sparsity pattern
    for (int i = 0; i < M.rows(); i++) M.insert(i, i) = static_cast<Scalar>(1);

    // if Eigen type, make it compressed
    if constexpr (std::is_base_of_v<Eigen::SparseCompressedBase<SparseMatrix>,
                                    SparseMatrix>) {
      M.makeCompressed();
    }

    // auto db_vec = [](std::vector<int>& v, std::string s) {
    //   std::cout << s << std::endl;
    //   for (auto n : v) std::cout << n << " ";
    //   std::cout << std::endl;
    // };

    // index of the column
    int k = 0;
    for (; k < M.cols(); k++) {
      // define variables
      int n1 = 0;
      int n2 = 0;
      double residual_norm = 0.0;
      std::vector<int> J;
      std::vector<int> I;
      std::vector<int> sortedJ(M.cols());
      FullMatrix AHat;
      Vector mHat_k;
      Vector residual;
      FullMatrix Q;
      FullMatrix R;

      // define vector J
      n2 = M.outerIndexPtr()[k + 1] - M.outerIndexPtr()[k];
      J = std::vector<int>(n2);
      int h = 0;
      for (int i = M.outerIndexPtr()[k]; i < M.outerIndexPtr()[k + 1]; i++) {
        J[h] = M.innerIndexPtr()[i];
        h++;
      }

      // 2) Compute the row indices I of the corresponding nonzero entries of
      // A(i, J) We initialize I to -1, and the iterate through all elements of
      // J. Then we iterate through the row indeces of A from the offset J[j] to
      // J[j] + 1. If the row index is already in I, we dont do anything, else
      // we add it to I.
      I = std::vector<int>(A.rows(), -1);
      n1 = 0;
      for (int j = 0; j < n2; j++) {
        for (int i = A.outerIndexPtr()[J[j]]; i < A.outerIndexPtr()[J[j] + 1];
             i++) {
          int keep = 1;
          // TODO: can be removed
          for (int h = 0; h < A.rows(); h++) {
            if (A.innerIndexPtr()[i] == I[h]) {
              std::cout << "Point 2): NOT KEEPING" << std::endl;
              keep = 0;
            }
          }
          if (keep == 1) {
            I[n1++] = A.innerIndexPtr()[i];
          }
        }
      }

      if (n1 == 0) {
        n2 = 0;
      }

      // 3) Create Â = A(I, J)
      // We initialize AHat to zeros. Then we iterate through all indeces of J,
      // and iterate through all indeces of I. For each of the indices of I and
      // the indices in the flatRowIndex, we check if they match. If they do, we
      // add that element to AHat.
      sparse_to_dense_mat(A, AHat, I, J, n1, n2);
      // create e_k
      Vector e_k(n1);
      for (int i = 0; i < n1; i++) {
        if (k == I[i]) {
          e_k[i] = 1.0;
        }
      }

      // 4-5) solve least square problem of AHat
      solve_least_square(AHat, mHat_k, e_k, Q, R);

      // 6) compute residual = A * mHat_k - e_k
      // create A dense
      std::vector<int> IDense(A.rows());
      std::iota(IDense.begin(), IDense.end(), 0);
      std::vector<int> JDense(A.cols());
      std::iota(JDense.begin(), JDense.end(), 0);
      FullMatrix ADense;
      sparse_to_dense_mat(A, ADense, IDense, JDense, A.rows(), A.cols());

      // Compute residual = (A * mHat_k) - e_k, need hardcoded loops as we have
      // custom branches to match sizes
      residual.resize(A.rows());
      for (int i = 0; i < A.rows(); i++) {
        residual[i] = 0.0;
        for (int j = 0; j < A.cols(); j++) {
          for (int h = 0; h < n2; h++) {
            if (J[h] == j) {
              residual[i] += ADense(i, j) * mHat_k[h];
            }
          }
        }
        if (i == k) {
          residual[i] -= static_cast<Scalar>(-1);
        }
      }
      // Compute the norm of the residual
      residual_norm = apsc::LinearAlgebra::Utils::vector_norm_2(residual);

      std::size_t iter = 0;
      while (residual_norm > tollerance && iter < max_iter) {
        iter++;

        int n1Tilde = 0;
        int n2Tilde = 0;
        int n1Union = 0;
        int n2Union = 0;
        int l = 0;
        int kNotInI = 0;

        std::vector<int> L;
        std::vector<int> keepArray(
            A.cols(),
            0);  // std::reduce is applied on this vector hence the container
                 // scalar type should match the reduce output type
        std::vector<int> JTilde;
        std::vector<int> ITilde(A.rows(), -1);
        std::vector<int> IUnion;
        std::vector<int> JUnion;
        std::vector<Scalar> rhoSq;
        std::vector<int> smallestIndices;
        std::vector<int> smallestJTilde;

        // 7) Set L to the set of indices where r(l) != 0
        // Count the numbers of nonzeros in residual
        for (int i = 0; i < A.rows(); i++) {
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
        // resize vector L to avoid pushbacks
        L.resize(l);
        std::fill(L.begin(), L.end(), -1);
        int index = 0;
        for (int i = 0; i < A.rows(); i++) {
          if (residual[i] != 0.0 || (kNotInI && i == k)) {
            L[index++] = i;
          }
        }

        // 8) Set JTilde to the set of columns of A corresponding to the indices
        // in L that are not already in J Check what indeces we should keep
        for (int i = 0; i < A.cols(); i++) {
          for (int j = 0; j < l; j++) {
            for (int h = A.outerIndexPtr()[i]; h < A.outerIndexPtr()[i + 1];
                 h++) {
              if (L[j] == A.innerIndexPtr()[h]) {
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
        n2Tilde = std::reduce(keepArray.begin(), keepArray.end());
        // resize vector
        JTilde.resize(n2Tilde);
        // Fill JTilde
        std::fill(JTilde.begin(), JTilde.end(), -1);
        index = 0;
        for (int i = 0; i < A.cols(); i++) {
          if (keepArray[i] == 1) {
            JTilde[index++] = i;
          }
        }

        // 9) For each j in JTilde, solve the minimization problem
        // resize rhoSq
        rhoSq.resize(n2Tilde);
        for (int i = 0; i < n2Tilde; i++) {
          double rTAe_j = 0.0;  // r^T * A(.,j)
          for (int j = A.outerIndexPtr()[JTilde[i]];
               j < A.outerIndexPtr()[JTilde[i] + 1]; j++) {
            rTAe_j += A.valuePtr()[j] * residual[A.innerIndexPtr()[j]];
          }

          double Ae_jNorm = 0.0;
          for (int j = A.outerIndexPtr()[JTilde[i]];
               j < A.outerIndexPtr()[JTilde[i] + 1]; j++) {
            Ae_jNorm += A.valuePtr()[j] * A.valuePtr()[j];
          }
          Ae_jNorm = std::sqrt(Ae_jNorm);
          rhoSq[i] = residual_norm * residual_norm -
                     (rTAe_j * rTAe_j) / (Ae_jNorm * Ae_jNorm);
        }

        // TODO:
        //  10) Find the s indeces of the column with the smallest rhoSq
        //  int newN2Tilde = std::min(s, n2Tilde);
        //  smallestIndices.resize(newN2Tilde);
        //  std::fill(smallestIndices.begin(), smallestIndices.end(), -1);
        //  // We iterate through rhoSq and find the smallest indeces.
        //  // First, we set the first s indeces to the first s indeces of
        //  JTilde
        //  // then if we find a smaller rhoSq, we shift the indeces to the
        //  right
        //  // we insert the index of JTilde with the rhoSq smaller than the
        //  current smallest elements
        //  // smallestIndices then contain the indeces of JTIlde corresponding
        //  to the smallest values of rhoSq for (int i=0; i<n2Tilde; i++) {
        //    for (int j=0; j<newN2Tilde; j++) {
        //      if (smallestIndices[j] == -1) {
        //        smallestIndices[j] = i;
        //        break;
        //      } else if (rhoSq[i] < rhoSq[smallestIndices[j]]) {
        //        for (int h=newN2Tilde-1; h>j; h--) {
        //          smallestIndices[h] = smallestIndices[h - 1];
        //        }
        //        smallestIndices[j] = i;
        //        break;
        //      }
        //    }
        //  }
        //  smallestJTilde.resize(newN2Tilde);
        //  for (int i = 0; i < newN2Tilde; i++) {
        //    smallestJTilde[i] = JTilde[smallestIndices[i]];
        //  }
        //  JTilde.resize(newN2Tilde);
        //  for (int i=0; i<newN2Tilde; i++) {
        //    JTilde[i] = smallestJTilde[i];
        //  }
        //  n2Tilde = newN2Tilde;

        // 11) Determine the new indices Î
        // Denote by ITilde the new rows, which corresponds to the nonzero rows
        // of A(:, J union JTilde) not contained in I yet
        // Make union of J and JTilde
        n2Union = n2 + n2Tilde;
        JUnion.resize(n2Union);
        for (int i = 0; i < n2; i++) {
          JUnion[i] = J[i];
        }
        for (int i = 0; i < n2Tilde; i++) {
          JUnion[n2 + i] = JTilde[i];
        }
        n1Tilde = 0;
        for (int j = 0; j < n2Union; j++) {
          for (int i = A.outerIndexPtr()[JUnion[j]];
               i < A.outerIndexPtr()[JUnion[j] + 1]; i++) {
            int keep = 1;
            for (int h = 0; h < n1; h++) {
              if (A.innerIndexPtr()[i] == I[h] ||
                  A.innerIndexPtr()[i] == ITilde[h]) {
                keep = 0;
              }
            }
            if (keep == 1) {
              ITilde[n1Tilde++] = A.innerIndexPtr()[i];
            }
          }
        }

        // 12) Make I U ITilde and J U JTilde
        // Make union of I and ITilde
        n1Union = n1 + n1Tilde;
        IUnion.resize(n1Union);
        for (int i = 0; i < n1; i++) {
          IUnion[i] = I[i];
        }
        for (int i = 0; i < n1Tilde; i++) {
          IUnion[n1 + i] = ITilde[i];
        }

        // update qr factorisation without computing the whole factorisation
        update_qr(A, AHat, ADense, Q, R, I, J, sortedJ, ITilde, JTilde, IUnion,
                  JUnion, n1, n2, n1Tilde, n2Tilde, n1Union, n2Union, mHat_k,
                  residual, residual_norm, k);

        n1 = n1Union;
        n2 = n2Union;
      }
      // 16) Set m_k(J) = mHat_k
      // Update kth column of M
      for (int i = 0; i < M.rows(); i++) {
        M.coeffRef(i, k) = mHat_k[i];
      }
    }
  }

  void set_tollerance(double tol) { tollerance = tol; }

  void set_max_inter(std::size_t iter) { max_iter = iter; }

  void set_s(int s_in) { s = s_in; }

 protected:
  SparseMatrix M;
  double tollerance = 1e-2;
  std::size_t max_iter = 30;
  int s = 1;

  void sparse_to_dense_mat(SparseMatrix &m, FullMatrix &dest,
                           std::vector<int> &I, std::vector<int> &J,
                           const int n1, const int n2) {
    dest.resize(n1, n2);
    for (int i = 0; i < n1; i++) {
      for (int j = 0; j < n2; j++) {
        dest(i, j) = static_cast<Scalar>(0);
        for (int l = m.outerIndexPtr()[J[j]]; l < m.outerIndexPtr()[J[j] + 1];
             l++) {
          if (I[i] == m.innerIndexPtr()[l]) {
            dest(i, j) = m.valuePtr()[l];
          }
        }
      }
    }
  }

  void solve_manual_least_square(SparseMatrix &A, Vector &b, FullMatrix &Q,
                                 FullMatrix &R, Vector &x, int row_size,
                                 int k) {
    // 5.1) Compute t = Q^T * b
    Vector t = Q.transpose() * b;

    // 5.2) Make the inverse of R of size n2 x n2
    FullMatrix invR = R.block(0, 0, row_size, row_size);
    // TODO: use a liner solver
    invR = invR.inverse();

    // 5.3) Compute x = R^-1 * cHat
    x = invR * t;
  }

  void solve_least_square(FullMatrix &m, Vector &dest, Vector &b, FullMatrix &Q,
                          FullMatrix &R) {
    if constexpr (std::is_base_of_v<Eigen::MatrixBase<FullMatrix>,
                                    FullMatrix>) {
      Eigen::HouseholderQR<FullMatrix> qr(m);
      Q = qr.householderQ();
      R = qr.matrixQR().template triangularView<Eigen::Upper>();
      dest = qr.solve(b);
    } else {
      std::cerr << "###### FAILURE: least sqaure problem solver not "
                   "implemented for the template FullMatrix #####"
                << std::endl;
    }
  }

  void create_permutation_matrices(std::vector<int> &I, std::vector<int> &J,
                                   int n1, int n2, FullMatrix &Pr,
                                   FullMatrix &Pc) {
    const Scalar one = static_cast<Scalar>(1);
    const Scalar zero = static_cast<Scalar>(0);

    // Create normalized index of I
    std::vector<int> IIndex(n1);
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
    Pr.resize(n1, n1);
    for (int i = 0; i < n1; i++) {
      for (int j = 0; j < n1; j++) {
        Pr(i, j) = IIndex[j] == i ? one : zero;
      }
    }

    // Create normalized index of J
    std::vector<int> JIndex(n2);
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
    Pc.resize(n2, n2);
    for (int i = 0; i < n2; i++) {
      for (int j = 0; j < n2; j++) {
        Pc(i, j) = JIndex[j] == i ? one : zero;
      }
    }
  }

  /* Function for updating the QR decomposition
  A = the input CSC matrix
  AHat = the submatrix of size n1 x n2
  ADense = the input dense matrix
  Q = the Q matrix
  R = the R matrix
  I = the row indices of AHat
  J = the column indices of AHat
  ITilde = the row indices to potentioally add to AHat
  JTilde = the column indices to potentially add to AHat
  IUnion = the union of I and ITilde
  JUnion = the union of J and JTilde
  n1 = the lentgh of I
  n2 = the length of J
  n1Tilde = the length of ITilde
  n2Tilde = the length of JTilde
  n1Union = the length of IUnion
  n2Union = the length of JUnion
  m_kOut = the output of the LS problem
  residual = the residual vector
  residual_norm = the norm of the residual vector
  k = the current iteration */
  void update_qr(SparseMatrix &A, FullMatrix &AHat, FullMatrix &ADense,
                 FullMatrix &Q, FullMatrix &R, std::vector<int> &I,
                 std::vector<int> &J, std::vector<int> &sortedJ,
                 std::vector<int> &ITilde, std::vector<int> &JTilde,
                 std::vector<int> &IUnion, std::vector<int> &JUnion, int n1,
                 int n2, int n1Tilde, int n2Tilde, int n1Union, int n2Union,
                 Vector &m_kOut, Vector &residual, double &residual_norm,
                 int k) {
    // 13.1) Create A(I, JTilde) and A(ITilde, JTilde)
    FullMatrix AIJTilde, AITildeJTilde;
    sparse_to_dense_mat(A, AIJTilde, I, JTilde, n1, n2Tilde);
    sparse_to_dense_mat(A, AITildeJTilde, ITilde, JTilde, n1Tilde, n2Tilde);
    // Create permutation matrices Pr and Pc
    FullMatrix Pr, Pc;
    create_permutation_matrices(IUnion, JUnion, n1Union, n2Union, Pr, Pc);

    // 13.2) ABreve of size n1 x n2Tilde = Q^T * AIJTilde
    FullMatrix ABreve = Q.transpose() * AIJTilde;

    // 13.3) Compute B1 = ABreve[0 : n2, 0 : n2Tilde]
    FullMatrix B1(n2, n2Tilde);
    for (int i = 0; i < n2; i++) {
      for (int j = 0; j < n2Tilde; j++) {
        B1(i, j) = ABreve(i, j);
      }
    }

    // 13.4) Compute B2 = ABreve[n2 + 1 : n1, 0 : n2Tilde] + AITildeJTilde
    FullMatrix B2;
    if (n1 - n2 < 0) {
      B2.resize(n1Tilde, n2Tilde);
      for (int i = 0; i < n1Tilde; i++) {
        for (int j = 0; j < n2Tilde; j++) {
          B2(i, j) = AITildeJTilde(i, j);
        }
      }
    } else {
      B2.resize(n1Union - n2, n2Tilde);
      for (int i = 0; i < n1 - n2; i++) {
        for (int j = 0; j < n2Tilde; j++) {
          B2(i, j) = ABreve((n2 + i), j);
        }
      }
      for (int i = 0; i < n1Tilde; i++) {
        for (int j = 0; j < n2Tilde; j++) {
          B2((n1 - n2 + i), j) = AITildeJTilde(i, j);
        }
      }
    }

    // 13.5) Do QR factorization of B2
    Eigen::HouseholderQR<decltype(B2)> B2_qr(B2);
    FullMatrix B2Q = B2_qr.householderQ();
    FullMatrix B2R = B2_qr.matrixQR().template triangularView<Eigen::Upper>();

    // 13.6) Compute Q_B and R_B from algorithm 17
    // Make first matrix with Q in the upper left and identity in the lower
    // right of size n1Union x n1Union
    FullMatrix firstMatrix(n1Union, n1Union);
    // we are not sure that the template FullMatrix initialise matrix to zero
    for (int i = 0; i < n1Union; i++) {
      for (int j = 0; j < n1Union; j++) {
        firstMatrix(i, j) = static_cast<Scalar>(0);
      }
    }
    for (int i = 0; i < n1; i++) {
      for (int j = 0; j < n1; j++) {
        firstMatrix(i, j) = Q(i, j);
      }
    }
    for (int i = 0; i < n1Tilde; i++) {
      firstMatrix(n1 + i, n1 + i) = static_cast<Scalar>(1);
    }

    // Make second matrix with identity in the upper left corner and B2Q in the
    // lower right corner of size n1Union x n1Union
    FullMatrix secondMatrix(n1Union, n1Union);
    for (int i = 0; i < n1Union; i++) {
      for (int j = 0; j < n1Union; j++) {
        secondMatrix(i, j) = static_cast<Scalar>(0);
      }
    }
    for (int i = 0; i < n2; i++) {
      secondMatrix(i, i) = static_cast<Scalar>(1);
    }
    for (int i = 0; i < n1Union - n2; i++) {
      for (int j = 0; j < n1Union - n2; j++) {
        secondMatrix((n2 + i), n2 + j) = B2Q(i, j);
      }
    }

    // Compute unsortedQ = firstMatrix * secondMatrix
    FullMatrix unsortedQ = firstMatrix * secondMatrix;

    // Make unsortedR with R in the top left corner, B1 in the top right corner
    // and B2R under B1 of size n1Union x n2Union
    FullMatrix unsortedR(n1Union, n2Union);
    for (int i = 0; i < n1Union; i++) {
      for (int j = 0; j < n2Union; j++) {
        unsortedR(i, j) = static_cast<Scalar>(0);
      }
    }
    for (int i = 0; i < n2; i++) {
      for (int j = 0; j < n2; j++) {
        unsortedR(i, j) = R(i, j);
      }
    }
    for (int i = 0; i < n2; i++) {
      for (int j = 0; j < n2Tilde; j++) {
        unsortedR(i, n2 + j) = B1(i, j);
      }
    }
    for (int i = 0; i < n1Union - n2; i++) {
      for (int j = 0; j < n2Tilde; j++) {
        unsortedR((n2 + i), n2 + j) = B2R(i, j);
      }
    }

    // 13.7) Compute the new solution m_k for the least squares problem
    Vector e_k(n1Union);
    for (int i = 0; i < n1Union; i++) {
      if (k == IUnion[i]) {
        e_k[i] = static_cast<Scalar>(1);
      }
    }
    solve_manual_least_square(A, e_k, unsortedQ, unsortedR, m_kOut, n2Union, k);
    // Compute m_KOut = Pc * tempM_k
    // TODO: do we need to allocate a copy?
    Vector tempM_k = m_kOut;
    m_kOut = Pc * tempM_k;

    // 14) Compute residual = A * mHat_k - e_k
    // update I, J, sortedJ
    I = std::vector<int>(IUnion);
    J = std::vector<int>(JUnion);
    sortedJ = std::vector<int>(n2Union, 0);
    for (int i = 0; i < n2Union; ++i) {
      for (int j = 0; j < n2Union; ++j) {
        sortedJ[i] += static_cast<int>(Pc(i, j)) * JUnion[j];
      }
    }
    // create dense A
    //  std::vector<int> IDense(A.rows());
    //  std::iota(IDense.begin(), IDense.end(), 0);
    //  std::vector<int> JDense(A.cols());
    //  std::iota(JDense.begin(), JDense.end(), 0);
    //  FullMatrix ADense;
    //  sparse_to_dense_mat(A, ADense, IDense, JDense, A.rows(), A.cols());

    // compute residual = (AHat * m_kOut) - e_k
    residual.resize(A.rows());
    for (int i = 0; i < A.rows(); i++) {
      residual[i] = 0.0;
      for (int j = 0; j < A.cols(); j++) {
        for (int h = 0; h < n2Union; h++) {
          if (sortedJ[h] == j) {
            residual[i] += ADense(i, j) * m_kOut[h];
          }
        }
      }
      if (i == k) {
        residual[i] -= 1.0;
      }
    }
    // Compute the norm of the residual
    residual_norm = apsc::LinearAlgebra::Utils::vector_norm_2(residual);

    // set Q and R to unsortedQ and unsortedR
    Q = unsortedQ;
    R = unsortedR;
  }
};

}  // namespace Preconditioner
}  // namespace LinearAlgebra

#endif
