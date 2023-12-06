#ifndef HH_GC_MPI__HH
#define HH_GC_MPI__HH
//*****************************************************************
// Iterative template routine -- CG supporting MPI.
// Each MPI process must run this functions as the undeline
// matrix `A` takes a sub part of the original matrix.
//
// In other words: each process computes the CG algorithm
// exploiting the data parallel setting of the most costly operation:
// Matrix Vector multiplication O(n^2).
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
#include <MPIContext.hpp>
#include <MatrixWithVecSupport.hpp>
#include <mpi.h>

namespace LinearAlgebra {
template <class Matrix, class Vector, class Preconditioner, std::size_t Size,
          typename Scalar>
int CG(Matrix &A, Vector &x, const Vector &b, const Preconditioner &M,
       int &max_iter, typename Vector::Scalar &tol, const MPIContext mpi_ctx,
       MPI_Datatype mpi_datatype) {
#warning                                                                       \
    "If the precondioner does not exploit MPI this method may perform worse that the iterative solution"
  using Real = typename Matrix::Scalar;

  static_assert(
      (std::is_base_of_v<
           apsc::LinearAlgebra::MatrixWithVecSupport<
               Scalar, Vector, apsc::LinearAlgebra::ORDERING::COLUMNMAJOR>,
           Preconditioner> ||
       std::is_base_of_v<
           apsc::LinearAlgebra::MatrixWithVecSupport<
               Scalar, Vector, apsc::LinearAlgebra::ORDERING::ROWMAJOR>,
           Preconditioner>),
      "The input Preconditioner class does not derive from "
      "MatrixWithVecSupport");

  const int mpi_rank = mpi_ctx.mpi_rank();
  const MPI_Comm *mpi_comm = mpi_ctx.mpi_comm();

  Real resid;
  Vector p(b.size());
  Vector z(b.size());
  Vector q(b.size());
  Real alpha, beta, rho;
  Real rho_1(0.0);
  Real normb = b.norm();

  // MPI section start
  A.product(x);
  Vector AxX;
  A.template collectGlobal<Vector>(AxX);
  A.AllCollectGlobal(AxX);
  // MPI section end

  Vector r = b - AxX;

  if (normb == 0.0)
    normb = 1;

  if ((resid = r.norm() / normb) <= tol) {
    tol = resid;
    max_iter = 0;
    return 0;
  }

  for (int i = 1; i <= max_iter; i++) {
    // Computing this linear system in only one process and then broadcast the
    // result does not gain performance, hence it is computed in all processes
    // (`M` is not a MPI compatible matrix right now). If any data parallel
    // logic is introduced, the `Preconditioner` class must support MPI!
    z = M.template solve<decltype(r), Size>(r);
    rho = r.dot(z);

    if (i == 1)
      p = z;
    else {
      beta = rho / rho_1;
      p = z + (p * beta);
    }

    // MPI part start
    A.product(p);
    A.AllCollectGlobal(q);
    // MPI part end

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

template <class Matrix, class Vector, std::size_t Size, typename Scalar>
int CG_no_precon(Matrix &A, Vector &x, const Vector &b, int &max_iter,
                 typename Vector::Scalar &tol, const MPIContext mpi_ctx,
                 MPI_Datatype mpi_datatype) {
  using Real = typename Matrix::Scalar;

  const int mpi_rank = mpi_ctx.mpi_rank();
  const MPI_Comm *mpi_comm = mpi_ctx.mpi_comm();

  Real alpha, beta, resid, alpha_num, alpha_den, beta_num, beta_den;
  Real normb = b.norm();
  Vector A_prod_d;
  Vector d(b.size());
  Vector r(b.size());

  // MPI section start
  A.product(x);
  Vector AxX;
  A.template collectGlobal<Vector>(AxX);
  A.AllCollectGlobal(AxX);
  // MPI section end

  // Initialise residual and direction
  r = b - AxX;
  d = r;

  if (normb == 0.0) {
    normb = 1.0;
  }

  if ((resid = r.norm() / normb) <= tol) {
    tol = resid;
    max_iter = 0;
    return 0;
  }

  for (int i = 1; i <= max_iter; i++) {
    // alpha numerator
    alpha_num = d.dot(r);
    // alpha denominator
    A.product(d);
    A.AllCollectGlobal(A_prod_d);
    alpha_den = d.dot(A_prod_d);
    // alpha
    alpha = alpha_num / alpha_den;

    // update x
    x = x + (d * alpha);
    // update r
    r = r - (A_prod_d * alpha);

    // stop criteria
    if ((resid = r.norm() / normb) <= tol) {
      tol = resid;
      max_iter = i;
      return 0;
    }

    // beta numerator
    beta_num = A_prod_d.dot(r);
    // beta denominator
    beta_den = A_prod_d.dot(d);
    // beta
    beta = beta_num / beta_den;

    // update d
    d = r - (d * beta);
  }

  tol = resid;
  return 1;
}
} // namespace LinearAlgebra
#endif
