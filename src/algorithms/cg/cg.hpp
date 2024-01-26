#ifndef HH_GC___HH
#define HH_GC___HH
//*****************************************************************
// Iterative template routine -- CG
//
// CG solves the symmetric positive definite linear
// system Ax=b using the Conjugate Gradient method.
//
// CG follows the algorithm described on p. 15 in the
// SIAM Templates book.
//
// The return value indicates convergence within max_iter (input)
// iterations (0), or no convergence within max_iter iterations (1).
//
// Upon successful return, output arguments have the following values:
//
//        x  --  approximate solution to Ax = b
// max_iter  --  the number of iterations performed before the
//               tolerance was reached
//      tol  --  the residual after the final iteration
//
//*****************************************************************

#include <cstddef>
#include <type_traits>

#include "Matrix/Matrix.hpp"
#include "FullMatrix.hpp"
#include "Vector.hpp"

namespace apsc::LinearAlgebra {
namespace LinearSolvers {
namespace Sequential {
template <class Matrix, class Vector, class Preconditioner, typename Scalar>
int CG(const Matrix &A, Vector &x, const Vector &b, const Preconditioner &M,
       int &max_iter, typename Vector::Scalar &tol) {
  static_assert(
      (std::is_base_of_v<
           apsc::LinearAlgebra::FullMatrix<
               Scalar, Vector, apsc::LinearAlgebra::ORDERING::COLUMNMAJOR>,
           Preconditioner> ||
       std::is_base_of_v<
           apsc::LinearAlgebra::FullMatrix<
               Scalar, Vector, apsc::LinearAlgebra::ORDERING::ROWMAJOR>,
           Preconditioner>),
      "The input Preconditioner class does not derive from "
      "FullMatrix");

  using Real = typename Matrix::Scalar;
  Real resid;
  Vector p(b.size());
  Vector z(b.size());
  Vector q(b.size());
  Real alpha, beta, rho;
  Real rho_1(0.0);

  Real normb = b.norm();
  Vector r = b - A * x;

  if (normb == 0.0) normb = 1;

  if ((resid = r.norm() / normb) <= tol) {
    tol = resid;
    max_iter = 0;
    return 0;
  }

  for (int i = 1; i <= max_iter; i++) {
    z = M.template solve<decltype(r)>(r);
    rho = r.dot(z);

    if (i == 1)
      p = z;
    else {
      beta = rho / rho_1;
      p = z + (p * beta);
    }

    q = A * p;
    alpha = rho / p.dot(q);

    x = x + (p * alpha);
    r = r - (q * alpha);

    if ((resid = r.norm() / normb) <= tol) {
      tol = resid;
      max_iter = i;
      return 0;
    }

    rho_1 = rho;
  }

  tol = resid;
  return 1;
}

}  // namespace Sequential
}  // namespace LinearSolvers
}  // namespace LinearAlgebra
#endif
