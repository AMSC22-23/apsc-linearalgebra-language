#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <EigenStructureMap.hpp>
#include <FullMatrix.hpp>
#include <Vector.hpp>
#include <cmath>
#include <iostream>
#include <string>

#include "assert.hpp"

using std::cout;
using std::endl;

int main(int argc, char* argv[]) {
  using namespace apsc::LinearAlgebra;

  {
    constexpr unsigned size = 5;
    FullMatrix<double, Vector<double>, apsc::LinearAlgebra::ORDERING::ROWMAJOR>
        A(size, size);
    Utils::default_spd_fill<
        FullMatrix<double, Vector<double>, ORDERING::ROWMAJOR>, double>(A);

    auto mapped_A = EigenStructureMap<Eigen::Matrix<double, size, size>, double,
                                      decltype(A)>::create_map(A, size, size);
    cout << "Original matrix" << endl << A << endl;
    cout << "Mapped matrix" << endl << mapped_A.structure() << endl;
    mapped_A.structure() += mapped_A.structure();
    cout << "Modifying..." << endl;
    cout << "Original matrix" << endl << A << endl;
    cout << "Mapped matrix" << endl << mapped_A.structure() << endl;

    Vector<double> b(size, 1.0);
    auto mapped_b = EigenStructureMap<Eigen::Matrix<double, size, 1>, double,
                                      decltype(b)>::create_map(b, size);
    cout << "Original vector" << endl << b << endl;
    cout << "Mapped vector" << endl << mapped_b.structure() << endl;
    mapped_b.structure() *= 2.0;
    cout << "Modifying..." << endl;
    cout << "Original vector" << endl << b << endl;
    cout << "Mapped vector" << endl << mapped_b.structure() << endl;
  }

  // test sparse matrix
  {
    double values[] = {1.0, 2.0, 3.0};
    int innerIndices[] = {0, 1, 2};
    int outerIndices[] = {0, 1, 2, 3};
    // Number of rows and columns
    int rows = 3;
    int cols = 3;

    auto sparse_map = Eigen::Map<Eigen::SparseMatrix<double>>(
        rows, cols, 3, outerIndices, innerIndices, values);
    std::cout << "Sparse matrix map:" << std::endl << sparse_map << std::endl;
  }

  // test inverse matrix;
  {
    int n = 2;
    double values[] = {2.0, 0.0, 0.0, 2.0};
    auto eigen_m =
        EigenStructureMap<Eigen::MatrixXd, double>(values, n, n).structure();
    std::cout << "Eigen full matrix" << std::endl << eigen_m << std::endl;
    Eigen::MatrixXd inverse_m = eigen_m.inverse();
    ASSERT(inverse_m.allFinite(), "Failed to invert");
    std::cout << "Eigen inverse matrix" << std::endl << inverse_m << std::endl;
    double* inverse_data = inverse_m.data();
    std::cout << "inverse buffer:";
    for (int i = 0; i < n * n; i++) {
      std::cout << inverse_data[i] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}
