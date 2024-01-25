#include <apsc_language.hpp>
#include <Vector.hpp>
#include <utils.hpp>
#include <mpi.h>

int main(int argc, char* argv[]) {
  // Declare the MPI runner that will offer MPI context and handle the finalisation
  apsc::LinearAlgebra::Utils::MPIUtils::MPIRunner mpi_runner(&argc, &argv);

  {
    if (mpi_runner.mpi_rank == 0) {
      std::cout << " ===================================================================" << std::endl;
      std::cout << " =                        FULL MATRIX DEMO                         =" << std::endl; 
      std::cout << " ===================================================================" << std::endl;
    }

    apsc::LinearAlgebra::Language::FullMatrix<double, apsc::LinearAlgebra::Language::OrderingType::COLUMNMAJOR, 1> M;
    M.load_from_file("../inputs/tridiagonal_M10_N10.mtx");
    // Test full matrix
    if (mpi_runner.mpi_rank == 0) {
      std::cout << "=============== Testing file load and MPI split ===============" << std::endl;
      std::cout << "loaded full matrix:" << std::endl << M << std::endl;
      std::cout << std::endl << std::endl;
      std::cout << "split matrix over MPI processes:" << std::endl;
    }
    // Test MPI split
    M.show_mpi_split();

    if (mpi_runner.mpi_rank == 0) {
      std::cout << std::endl << std::endl << std::endl;
    }

    // Test modificaton
    M(0, 0) = 10.0;
    if (mpi_runner.mpi_rank == 0) {
      std::cout << "=============== Testing modification ===============" << std::endl;
      std::cout << "modified full matrix:" << std::endl << M << std::endl;
      std::cout << std::endl << std::endl;
      std::cout << "split matrix over MPI processes:" << std::endl;
    }
    // Forward the modification to MPI
    M.setup_mpi();
    M.show_mpi_split();

    if (mpi_runner.mpi_rank == 0) {
      std::cout << std::endl << std::endl << std::endl;
    }

    // Test resize
    M.resize(M.rows() + 1, M.cols() + 1);
    for (int i=0; i<M.rows(); i++) {
      M(i, i) = 1.0;
    }
    if (mpi_runner.mpi_rank == 0) {
      std::cout << "=============== Testing resize ===============" << std::endl;
      std::cout << "modified full matrix:" << std::endl << M << std::endl;
      std::cout << std::endl << std::endl;
      std::cout << "split matrix over MPI processes:" << std::endl;
    }
    // Forward the modification to MPI
    M.setup_mpi();
    M.show_mpi_split();

    if (mpi_runner.mpi_rank == 0) {
      std::cout << std::endl << std::endl << std::endl;
    }

    // Algebra operation
    M(0,0) = 2.0;
    M(M.rows()-1, M.cols()-1) = 2.0;
    M.setup_mpi();
    int size = M.rows();
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    apsc::LinearAlgebra::Vector<double> b(size);
    b.fill(1.0);
    // Done in parallel!
    auto matmul = M * b;
    if (mpi_runner.mpi_rank == 0) {
      std::cout << "=============== Testing matrix vector multiplication ===============" << std::endl;
      std::cout << "Matmul between:" << std::endl << M << "and" << std::endl << b << std::endl << "=" << std::endl << matmul << std::endl;
    }

    if (mpi_runner.mpi_rank == 0) {
      std::cout << std::endl << std::endl << std::endl;
    }

    MPI_Barrier(mpi_runner.communicator);

    // Iterative solver
    //create exact solution vector
    apsc::LinearAlgebra::Vector<double> e(size, 1.0);
    // Done in parallel!
    b = M * e;
    // Iterative solvers runs in parallel!
    auto x = M.solve_iterative(b);
    if (mpi_runner.mpi_rank == 0) {
      std::cout << "=============== Testing iterative linear solver ===============" << std::endl;
      std::cout << "Linear solver between:" << std::endl << M << "and" << std::endl << b << std::endl << "=" << std::endl << x << std::endl;
      std::cout << "Error norm: " << (x - e).norm() << std::endl;
      std::cout << "Solver return code: " << M.solver_success() << std::endl;
    }

    if (mpi_runner.mpi_rank == 0) {
      std::cout << std::endl << std::endl << std::endl;
    }

    MPI_Barrier(mpi_runner.communicator);

    // Direct solvers runs is sequential!
    x = M.solve_direct(b);
    if (mpi_runner.mpi_rank == 0) {
      std::cout << "=============== Testing direct linear solver ===============" << std::endl;
      std::cout << "Linear solver between:" << std::endl << M << "and" << std::endl << b << std::endl << "=" << std::endl << x << std::endl;
      std::cout << "Error norm: " << (x - e).norm() << std::endl;
      std::cout << "Solver return code: " << M.solver_success() << std::endl;
    }

    if (mpi_runner.mpi_rank == 0) {
      std::cout << std::endl << std::endl << std::endl;
    }

    MPI_Barrier(mpi_runner.communicator);

    // Test a larger problem
    const int test_size = 200;
    M.resize(test_size, test_size);
    M.fill(0.0);
    for (int i=0; i<test_size; i++) {
      M(i, i) = 2.0;
      if (i > 0) {
        M(i, i-1) = -1.0;
      }
      if (i < test_size-1) {
        M(i, i+1) = -1.0;
      }
    }
    M.setup_mpi();
    e.resize(test_size);
    e.fill(1.0);
    b = M * e; 
    x = M.solve_iterative(b);
    if (mpi_runner.mpi_rank == 0) {
      std::cout << "=============== Testing iterative linear solver ===============" << std::endl;
      std::cout << "Problem size: " << test_size << std::endl;  
      std::cout << "Error norm: " << (x - e).norm() << std::endl;
      std::cout << "Solver return code: " << M.solver_success() << std::endl;
    }
  }


  {
    if (mpi_runner.mpi_rank == 0) {
      std::cout << std::endl << std::endl;
      std::cout << " ===================================================================" << std::endl;
      std::cout << " =                        SPARSE MATRIX DEMO                       =" << std::endl; 
      std::cout << " ===================================================================" << std::endl;
    }

    apsc::LinearAlgebra::Language::SparseMatrix<double, apsc::LinearAlgebra::Language::OrderingType::COLUMNMAJOR, 1> M;
    M.load_from_file("../inputs/tridiagonal_M10_N10.mtx");
    if (mpi_runner.mpi_rank == 0) {
      std::cout << "=============== Testing file load and MPI split ===============" << std::endl;
      std::cout << "loaded full matrix:" << std::endl << M << std::endl;
      std::cout << std::endl << std::endl;
      std::cout << "split matrix over MPI processes:" << std::endl;
    }
    M.show_mpi_split();

    if (mpi_runner.mpi_rank == 0) {
      std::cout << std::endl << std::endl << std::endl;
    }

    // Test modificaton
    M(0, 0) = 10.0;
    if (mpi_runner.mpi_rank == 0) {
      std::cout << "=============== Testing modification ===============" << std::endl;
      std::cout << "modified full matrix:" << std::endl << M << std::endl;
      std::cout << std::endl << std::endl;
      std::cout << "split matrix over MPI processes:" << std::endl;
    }
    // Forward the modification to MPI
    M.setup_mpi();
    M.show_mpi_split();

    if (mpi_runner.mpi_rank == 0) {
      std::cout << std::endl << std::endl << std::endl;
    }

    // Test resize
    M.resize(M.rows() + 1, M.cols() + 1);
    for (int i=0; i<M.rows(); i++) {
      M(i, i) = 1.0;
    }
    if (mpi_runner.mpi_rank == 0) {
      std::cout << "=============== Testing resize ===============" << std::endl;
      std::cout << "modified full matrix (" << M.rows() << " x " << M.cols() << "):" << std::endl << M << std::endl;
      std::cout << std::endl << std::endl;
      std::cout << "split matrix over MPI processes:" << std::endl;
    }
    // Forward the modification to MPI
    M.setup_mpi();
    M.show_mpi_split();

    if (mpi_runner.mpi_rank == 0) {
      std::cout << std::endl << std::endl << std::endl;
    }

    // Algebra operation
    M(0,0) = 2.0;
    M(M.rows()-1, M.cols()-1) = 2.0;
    M.setup_mpi();
    int size = M.rows();
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Here we need to use a compatible vector with the sparse matrix class in order to perform algbraic operations
    apsc::LinearAlgebra::Language::SparseMatrix<double>::Vector b(size);
    b.fill(1.0);
    // Done in parallel!
    auto matmul = M * b;
    if (mpi_runner.mpi_rank == 0) {
      std::cout << "=============== Testing matrix vector multiplication ===============" << std::endl;
      std::cout << "Matmul between:" << std::endl << M << "and" << std::endl << b << std::endl << "=" << std::endl << matmul << std::endl;
    }

    if (mpi_runner.mpi_rank == 0) {
      std::cout << std::endl << std::endl << std::endl;
    }

    MPI_Barrier(mpi_runner.communicator);

    // Iterative solver
    //create exact solution vector
    apsc::LinearAlgebra::Language::SparseMatrix<double>::Vector e(size);
    e.fill(1.0);
    // Done in parallel!
    b = M * e;
    // Iterative solvers runs in parallel!
    auto x = M.solve_iterative(b);
    if (mpi_runner.mpi_rank == 0) {
      std::cout << "=============== Testing iterative linear solver ===============" << std::endl;
      std::cout << "Linear solver between:" << std::endl << M << "and" << std::endl << b << std::endl << "=" << std::endl << x << std::endl;
      std::cout << "Error norm: " << (x - e).norm() << std::endl;
      std::cout << "Solver return code: " << M.solver_success() << std::endl;
    }

    if (mpi_runner.mpi_rank == 0) {
      std::cout << std::endl << std::endl << std::endl;
    }

    MPI_Barrier(mpi_runner.communicator);

    // Direct solvers runs is sequential!
    x = M.solve_direct(b);
    if (mpi_runner.mpi_rank == 0) {
      std::cout << "=============== Testing direct linear solver ===============" << std::endl;
      std::cout << "Linear solver between:" << std::endl << M << "and" << std::endl << b << std::endl << "=" << std::endl << x << std::endl;
      std::cout << "Error norm: " << (x - e).norm() << std::endl;
      std::cout << "Solver return code: " << M.solver_success() << std::endl;
    }

    if (mpi_runner.mpi_rank == 0) {
      std::cout << std::endl << std::endl << std::endl;
    }

    MPI_Barrier(mpi_runner.communicator);

    // Test a larger problem
    const int test_size = 2000;
    M.resize(test_size, test_size);
    for (int i=0; i<test_size; i++) {
      M(i, i) = 2.0;
      if (i > 0) {
        M(i, i-1) = -1.0;
      }
      if (i < test_size-1) {
        M(i, i+1) = -1.0;
      }
    }
    M.setup_mpi();
    e.resize(test_size);
    e.fill(1.0);
    b = M * e; 
    x = M.solve_iterative(b);
    if (mpi_runner.mpi_rank == 0) {
      std::cout << "=============== Testing iterative linear solver ===============" << std::endl;
      std::cout << "Problem size: " << test_size << std::endl;  
      std::cout << "Error norm: " << (x - e).norm() << std::endl;
      std::cout << "Solver return code: " << M.solver_success() << std::endl;
    }
  }

  return 0;
}
