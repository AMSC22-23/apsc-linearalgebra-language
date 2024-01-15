#ifndef SPAI_HPP
#define SPAI_HPP

#include "assert.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
namespace LinearAlgebra
{
namespace Preconditioner
{

template <typename Scalar, typename SparseMatrix, typename FullMatrix, typename Vector>
class SPAI
{
public:
  SPAI() {

  }

  SPAI(SparseMatrix A) {
    init(A);
  }

  void init(SparseMatrix A) {
    ASSERT(A.rows() == A.cols(), "A must be a square matrix");
    M.resize(A.rows(), A.cols());

    //set M to identity
    for (int i=0; i<M.rows(); i++)
      M.coeffRef(i, i) = static_cast<Scalar>(1);

    //if Eigen type, make it compressed
    if constexpr (std::is_base_of_v<Eigen::SparseCompressedBase<SparseMatrix>,
                                     SparseMatrix>) {
      M.makeCompressed();
    }

    // std::cout << "Initial M" << std::endl << M << std::endl;
    // auto M_outer_index_ptr = M.outerIndexPtr();
    // std::cout << "M offset: " << std::endl;
    // for (int i=0; i<=M.outerSize(); i++) {
    //   std:: cout << M_outer_index_ptr[i] << " ";
    // }
    // std::cout << std::endl;

    auto db_vec = [](std::vector<int>& v, std::string s) {
      std::cout << s << std::endl;
      for (auto n : v) std::cout << n << " ";
      std::cout << std::endl;
    };

    //index of the column
    int k = 0;
    for (; k<M.cols(); k++) {
      //define variables
      int n1 = 0;
      int n2 = 0;
      int iteration = 0;
      double residual_norm = 0.0;
      std::vector<int> J;
      std::vector<int> I;
      std::vector<int> sortedJ(M.cols());
      FullMatrix AHat;
      Vector mHat_k;
      Vector residual;

      //CONVERSION
      //flatRowIndex = innerIndexPtr
      //offset = outerIndexPtr

      std::cout << "COL: " << k << std::endl;

      //define vector J
      n2 = M.outerIndexPtr()[k+1] - M.outerIndexPtr()[k];
      std::cout << "n2 = " << n2 << std::endl;
      J = std::vector<int>(n2);
      int h = 0;
      for (int i=M.outerIndexPtr()[k]; i<M.outerIndexPtr()[k+1]; i++) {
        J[h] = M.innerIndexPtr()[i];
        h++;
      }

      db_vec(J, "J");

      // 2) Compute the row indices I of the corresponding nonzero entries of
      // A(i, J) We initialize I to -1, and the iterate through all elements of
      // J. Then we iterate through the row indeces of A from the offset J[j] to
      // J[j] + 1. If the row index is already in I, we dont do anything, else
      // we add it to I.
      I = std::vector<int>(A.rows(), -1);

      n1 = 0;
      for (int j = 0; j < n2; j++) {
        for (int i = A.outerIndexPtr()[J[j]]; i < A.outerIndexPtr()[J[j] + 1]; i++) {
          int keep = 1;
          for (int h = 0; h < A.rows(); h++) {
            if (A.innerIndexPtr()[i] == I[h]) {
              keep = 0;
            }
          }
          if (keep == 1) {
            I[n1++] = A.innerIndexPtr()[i];
          }
        }
      }
      std::cout << "n1 = " << n1 << std::endl;
      db_vec(I, "I");

      if (n1 == 0) {
        n2 = 0;
      }

      // 3) Create Â = A(I, J)
      // We initialize AHat to zeros. Then we iterate through all indeces of J,
      // and iterate through all indeces of I. For each of the indices of I and
      // the indices in the flatRowIndex, we check if they match. If they do, we
      // add that element to AHat.
      sparse_to_dense_mat(A, AHat, I, J, n1, n2);
      std::cout << "AHat\n" << AHat << std::endl;
      //create e_k
      Vector e_k(n1);
      for (int i = 0; i < n1; i++) {
        e_k[i] = 0.0;
        if (k == I[i]) {
            e_k[i] = 1.0;
        }
      }
      std::cout << "e_k\n" << e_k << std::endl;
      //4-5) solve least square problem
      solve_least_square(AHat, mHat_k, e_k);
      std::cout << "LSP sol\n" << mHat_k << std::endl;
      
      //6) compute residual
      residual = (A * mHat_k) - e_k;
      residual_norm = residual.norm();
      std::cout << "column " << k << "LSP residual norm " << residual_norm << std::endl;
    }
  }

protected:
  SparseMatrix M;

  void sparse_to_dense_mat(SparseMatrix& m, FullMatrix& dest, std::vector<int>& I,
                             std::vector<int>& J, const int n1, const int n2) {
    dest.resize(n1, n2);
    for (int i = 0; i < n1; i++) {
      for(int j = 0; j < n2; j++) {
        for (int l = m.outerIndexPtr()[J[j]]; l < m.outerIndexPtr()[J[j] + 1]; l++) {
          if (I[i] == m.innerIndexPtr()[l]) {
            dest(i*n2, j) = m.valuePtr()[l];
          }
        }
      }
    }
  }
  
  void solve_least_square(FullMatrix& m, Vector& dest, Vector& b) {
    dest = m.colPivHouseholderQr().solve(b);
  }
};


}
}


#endif 
