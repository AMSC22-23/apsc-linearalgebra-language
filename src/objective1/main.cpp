#include <MatrixWithVecSupport.hpp>
#include <Vector.hpp>
#include <cg.hpp>
#include <cstddef>
#include <iostream>

using std::cout;
using std::endl;

template <typename Lhs, typename Rhs, typename Scalar, typename ExactSol>
int cg_solve(Lhs &a, Rhs b, ExactSol &e) {
  const std::size_t system_size = b.size();
  // result vector
  apsc::LinearAlgebra::Vector<Scalar> x(system_size, static_cast<Scalar>(0.0));
  int max_iter = 10000;
  Scalar tol = 1e-18;

  // using no preconditioner (identity)
  Lhs P(system_size, system_size);
  Scalar one = static_cast<Scalar>(1.0);
  for (unsigned i = 0; i < system_size; i++) {
    P(i, i) = one;
  }

  auto result =
      LinearAlgebra::LinearSolvers::Sequential::CG<Lhs, Rhs, Lhs, Scalar>(
          a, x, b, P, max_iter, tol);

  cout << "Solution with Conjugate Gradient:" << endl;
  cout << "iterations performed:                      " << max_iter << endl;
  cout << "tolerance achieved:                        " << tol << endl;
  cout << "Error norm:                                " << (x - e).norm()
       << std::endl;

  return result;
}

int main(int argc, char *argv[]) {
  using namespace apsc::LinearAlgebra;

  constexpr unsigned size = 10;
  cout << "Creating a test matrix..." << endl;
  MatrixWithVecSupport<double, Vector<double>,
                       apsc::LinearAlgebra::ORDERING::COLUMNMAJOR>
      A(size, size);
  Utils::default_spd_fill<
      MatrixWithVecSupport<double, Vector<double>, ORDERING::COLUMNMAJOR>,
      double>(A);

  cout << "Creating a test rhs..." << endl;
  Vector<double> e(size, 1.0);
  Vector<double> b = A * e;

  return cg_solve<decltype(A), decltype(b), double, decltype(e)>(A, b, e);
}
