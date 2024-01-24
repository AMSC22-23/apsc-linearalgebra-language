#include <MatrixWithVecSupport.hpp>
#include <Vector.hpp>
#include <cstddef>
#include <iostream>

using std::cout;
using std::endl;

int main(int argc, char *argv[]) {
  using namespace apsc::LinearAlgebra;

  constexpr std::size_t size = 10;
  MatrixWithVecSupport<double, Vector<double>,
                       apsc::LinearAlgebra::ORDERING::COLUMNMAJOR>
      A(size, size);
  Utils::default_spd_fill<
      MatrixWithVecSupport<double, Vector<double>, ORDERING::COLUMNMAJOR>,
      double>(A);

  Vector<double> e(size, 1.0);
  Vector<double> b = A * e;

  auto x = A.solve<Vector<double>>(b);

  cout << "Solution of " << endl
       << A << endl
       << "X" << endl
       << b << endl
       << "=" << endl
       << x << endl;

  return 0;
}
