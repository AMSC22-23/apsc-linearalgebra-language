#include <MatrixWithVecSupport.hpp>
#include <Vector.hpp>
#include <iostream>
#include <string>

using std::cout;
using std::endl;

int main(int argc, char *argv[]) {
  using namespace apsc::LinearAlgebra;

  constexpr unsigned size = 10;
  Vector<double> e(size, 1.0);
  MatrixWithVecSupport<double, decltype(e),
                       apsc::LinearAlgebra::ORDERING::ROWMAJOR>
      A(size, size);
  Utils::default_spd_fill<
      MatrixWithVecSupport<double, decltype(e), ORDERING::ROWMAJOR>, double>(A);

  Vector<double> b = A * e;

  const std::string tab = ":\t";

  cout << "test copyctor of " << b;
  Vector<double> copy(b);
  cout << tab << copy << endl;

  cout << "test addition of " << b << " + " << copy << tab << b + copy << endl;

  cout << "test subtraction of " << b << " - " << copy << tab << b - copy
       << endl;

  cout << "test scalar prod of " << b << " *= " << 2.0 << tab;
  b *= 2.0;
  cout << b << endl;

  cout << "test norm of " << b << tab << b.norm() << endl;

  cout << "test norm of " << b << " - " << b << tab << (b - b).norm() << endl;

  return 0;
}
