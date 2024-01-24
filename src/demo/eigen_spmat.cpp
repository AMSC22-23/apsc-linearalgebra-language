#include <Eigen/Sparse>
#include <cstddef>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>

using namespace std;
using namespace Eigen;

int main(int argc, char **argv) {
  int n = 5;
  SparseMatrix<double, Eigen::RowMajor> colmaj(n, n);  // define matrix
  for (int i = 0; i < n; i++) {
    colmaj.coeffRef(i, i) = 2.0;
    if (i > 0) colmaj.coeffRef(i, i - 1) = -1.0;
    if (i < n - 1) colmaj.coeffRef(i, i + 1) = -1.0;
  }
  cout << colmaj << endl;

  cout << "outer size: " << colmaj.outerSize() << endl;
  cout << "inner size: " << colmaj.innerSize() << endl;
  for (int k = 0; k < colmaj.outerSize(); ++k) {
    int i = 0;
    for (decltype(colmaj)::InnerIterator it(colmaj, k); it; ++it, i++) {
      std::cout << "(" << it.row() << ",";  // row index
      std::cout << it.col() << ")\t";       // col index (here it is equal to k)
    }
    cout << "i: " << i << endl;
  }

  SparseMatrix<double, Eigen::RowMajor> rowmaj(n, n);  // define matrix
  for (int i = 0; i < n; i++) {
    rowmaj.coeffRef(i, i) = 2.0;
    if (i > 0) rowmaj.coeffRef(i, i - 1) = -1.0;
    if (i < n - 1) colmaj.coeffRef(i, i + 1) = -1.0;
  }
  cout << rowmaj << endl;

  cout << "outer size: " << rowmaj.outerSize() << endl;
  cout << "inner size: " << rowmaj.innerSize() << endl;
  for (int k = 0; k < rowmaj.outerSize(); ++k) {
    int i = 0;
    for (decltype(rowmaj)::InnerIterator it(rowmaj, k); it; ++it, i++) {
      std::cout << "(" << it.row() << ",";  // row index
      std::cout << it.col() << ")\t";       // col index (here it is equal to k)
    }
    cout << "i: " << i << endl;
  }

  return 0;
}
