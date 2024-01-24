/*
 * MatrixWithVecSupport.hpp
 *
 *  Created on: Nov 17, 2023
 *      Author: Kaixi Matteo Chen
 */

#ifndef MATRIX_WITH_VEC_SUPPORT_HPP
#define MATRIX_WITH_VEC_SUPPORT_HPP
#include <Eigen/Dense>
#include <EigenStructureMap.hpp>
#include <Matrix/Matrix.hpp>
#include <assert.hpp>
#include <cassert>
#include <cstddef>
#include <type_traits>
// To avoid stupid warnings if I do not use openmp
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
namespace apsc::LinearAlgebra {

/*!
 * A full matrix with vector multiplication support for
 * custom Vector class.
 *
 * The template Vector param type is used only for internal temporary buffers
 * and return types. Each computation method can be use with different Vector
 * classes (hence different from the class template Vector).
 *
 * @tparam Scalar The type of the element
 * @tparam Vector The type of the Vector to be used in computation
 * @tparam ORDER The Storage order (default row-wise)
 */
template <typename SCALAR, typename Vector, ORDERING ORDER = ORDERING::ROWMAJOR>
class MatrixWithVecSupport : public Matrix<SCALAR, ORDER> {
 public:
  using Scalar = SCALAR;

  /*!
   * Constructor may take number of rows and columns
   * @param nrows Number of rows
   * @param ncols Number of columns
   */
  MatrixWithVecSupport(std::size_t nrows = 0, std::size_t ncols = 0)
      : Matrix<SCALAR, ORDER>(nrows, ncols) {}

  /*!
   * Multiplication with a custom InputVectorType
   * The output vector has the matrix scalar type.
   *
   * Numerical faults will occur if InputVectorType::Scalar and Scalar
   * are different.
   *
   * @tparam InputVectorType the input vector type
   * @param v a InputVectorType to be multiplicated with
   * @return The result of A*v
   */
  template <typename InputVectorType>
  Vector operator*(InputVectorType const &v) const {
    ASSERT((Matrix<Scalar, ORDER>::nCols == v.size()),
           "MatVetMul: Matrix columns != Vector rows");

    Vector res(Matrix<Scalar, ORDER>::nRows, static_cast<Scalar>(0));

    if constexpr (ORDER == ORDERING::ROWMAJOR) {
      // loop over rows
      for (std::size_t i = 0; i < Matrix<Scalar, ORDER>::nRows; ++i) {
        Scalar r{0};
        auto offset = i * Matrix<Scalar, ORDER>::nCols;
        // loop over columns (can be multithreaded)
#pragma omp parallel for shared(i, res, offset) reduction(+ : r)
        for (std::size_t j = 0; j < Matrix<Scalar, ORDER>::nCols; ++j) {
          r += Matrix<Scalar, ORDER>::buffer[offset + j] * v[j];
        }
        res[i] = r;
      }
    } else {
      // loop over the columns
      for (std::size_t j = 0; j < Matrix<Scalar, ORDER>::nCols; ++j) {
        auto c = v[j];
        auto offset = j * Matrix<Scalar, ORDER>::nRows;
        // loop over rows (can be multithreaded)
#pragma omp parallel for shared(j, res, c, offset)
        for (std::size_t i = 0; i < Matrix<Scalar, ORDER>::nRows; ++i) {
          res[i] += c * Matrix<Scalar, ORDER>::buffer[offset + i];
        }
      }
    }
    return res;
  }

  /*!
   * Solve a linear system.
   *
   * Numerical faults will occur if InputVectorType::Scalar and Scalar
   * are different.
   *
   * @tparam InputVectorType the input vector type
   * @param v a InputVectorType representing the know data in the linear system
   * @return The result of Ax=b
   */
  template <typename InputVectorType>
  Vector solve(InputVectorType const &v) const {
    Vector x(Matrix<SCALAR, ORDER>::nCols, static_cast<Scalar>(0.0));

    ASSERT(this->cols() == v.size(), "Matrix col size != vector size");

    // map vector eigen interface
    auto eigen_vec =
        EigenStructureMap<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>, Scalar,
                          decltype(v)>::create_map(v, this->cols())
            .structure();

    // map matrix to eigen interface
    if constexpr (ORDER == ORDERING::ROWMAJOR) {
      auto eigen_mat =
          EigenStructureMap<Eigen::Matrix<Scalar, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>,
                            Scalar, decltype(*this)>::create_map(*this,
                                                                 this->rows(),
                                                                 this->cols())
              .structure();

      // TODO: consider using ldlt for SPD
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> res =
          eigen_mat.colPivHouseholderQr().solve(eigen_vec);
      const Scalar *buff = res.data();
      return Vector(buff, res.size());
    } else {
      auto eigen_mat =
          EigenStructureMap<
              Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>, Scalar,
              decltype(*this)>::create_map(*this, this->rows(), this->cols())
              .structure();

      // TODO: consider using ldlt for SPD
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> res =
          eigen_mat.colPivHouseholderQr().solve(eigen_vec);
      const Scalar *buff = res.data();
      return Vector(buff, res.size());
    }
  }
};

}  // namespace apsc::LinearAlgebra
#pragma GCC diagnostic pop

#endif /* MATRIX_WITH_VEC_SUPPORT_HPP */
