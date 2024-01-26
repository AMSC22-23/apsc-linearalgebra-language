#include <stdlib.h>

#include <CSC.hpp>
#include <Eigen/Sparse>
#include <iostream>

#include "EigenStructureMap.hpp"

using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using SpVec = Eigen::VectorXd;

int main(int argc, char* argv[]) {
  constexpr int size = 100;

  // cpp implementation
  SpMat A;
  A.resize(size, size);
  // Create tridiagonal
  for (int i = 0; i < size; i++) {
    A.insert(i, i) = 2.0;
    if (i > 0) {
      A.insert(i, i - 1) = -1.0;
    }
    if (i < size - 1) {
      A.insert(i, i + 1) = -1.0;
    }
  }
  A.makeCompressed();
  auto I = Eigen::MatrixXd::Identity(size, size);

  {
    SpVec e(size);
    e.fill(1.0);
    SpVec b = A * e;
    apsc::LinearAlgebra::CSC<double> CSC_A;
    CSC_A.map_external_buffer(A.outerIndexPtr(), A.valuePtr(),
                              A.innerIndexPtr(), A.rows(), A.cols(),
                              A.nonZeros());
    auto x = CSC_A.template solve<SpVec>(b);

    std::cout << "Solver norm = " << (x - e).norm() << std::endl;

    CSC_A.destoy();
  }

  return 0;
}
