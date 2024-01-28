# apsc Linear Algebra Language

This is a C++ header-only library designed for efficient manipulation of full
and sparse matrices using MPI (Message Passing Interface) techniques. This
library abstracts away the complexities of MPI programming, providing users
with a simple yet powerful interface for performing parallel matrix operations.

The following algebraic objects are available:
- [x] Full matrix
- [x] Sparse matrix
- [x] Full vector

The following algebraic operations are available on `FullMatrix`:
- [x] Addition, subtraciton, multiplication
- [x] Frobenius norm
- [x] Multiplication with a compatible vector

The following algebraic operations are available on `SparseMatrix`:
- [x] Addition, subtraciton, multiplication
- [x] Frobenius norm
- [x] Multiplication with a compatible vector

The following iterative linear solvers are available for `FullMatrix`
- [x] Conjugate gradient (with no preconditioner)

The following direct linear solvers (decompositions) are available for `FullMatrix`
- [x] QR

The following iterative linear solvers are available for `SparseMatrix`
- [x] Conjugate gradient (with no preconditioner)
- [x] GMRES (with no preconditioner)
- [x] GMRES + SPAI preconditioner
- [x] BiCGSTAB (with no preconditioner)
- [x] BiCGSTAB + SPAI preconditioner

Currently, the following direct linear solvers (decompositions) are available for `SparseMatrix`
- [x] QR

Let's introduce its usage by walking in an example! We will be using a sparse
matrix to show case the language features, but the same concepts can be applied
to full matrices too.

## Installation
This library is based on the following external parties:
- `MPI`
- `Eigen` >= `3.39`

Please be sure to have them installed.

The following docker container is recommended: `pcafrica/mk`, see
[here](https://github.com/HPC-Courses/AMSC-Labs/tree/main/Labs/2023-24/lab00-setup)

This library depends on the following submodules:
- [AMSC Code Examples](https://github.com/HPC-Courses/AMSC-CodeExamples)
You can initialise it by running:
```
git submodule update --init --recursive
```

## Project setup
You can create a directory named `your_project` under `src` and place your `cpp` and `hpp` file inside.
Then you can create the `CMake` configuration file by following this template `CMakeLists.txt` for the compilation step with `make`:
```
cmake_minimum_required(VERSION 3.12.0)
project(demo LANGUAGES CXX C)

include(./cmake_shared/cmake-common.cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find MPI package
find_package(MPI REQUIRED)

SET(EIGEN3_DIR_LOCAL $ENV{EIGEN3_INCLUDE_DIR})        #local installation
SET(EIGEN3_DIR_PCAFRICA $ENV{mkEigenInc})             #pcafrica/mk module

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/AMSC-CodeExamples/Examples/src/
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/ ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/algorithms/cg/
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/algorithms/gmres/
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/algorithms/bicg/
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/preconditioners/parallel/spai/
  ${EIGEN3_DIR_LOCAL} ${EIGEN3_DIR_PCAFRICA})
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

#Define executables
add_executable(main demo/main.cpp)
target_link_libraries(main MPI::MPI_CXX)
```

Note that you have to export the following variable path if you are using a local enviroenment:
```
export EIGEN3_INCLUDE_DIR=installation/path/include
```

## Demo
Let's walk into a demo to showcase the language capabilities!

First define a `main.cpp` file in the directory `src/demo` and include the header library:
```
#include <apsc_language.hpp>

int main(int argc, char* argv[]) {
    return 0;
}
```

If `MPI` is used, declare the MPI handling object that will take care of it initialisation and finalisation: 
```
  apsc::LinearAlgebra::Utils::MPIUtils::MPIRunner mpi_runner(&argc, &argv);
```

Let's create a sparse, MPI parallel matrix:
```
    apsc::LinearAlgebra::Language::SparseMatrix<
        double, 
        apsc::LinearAlgebra::Language::OrderingType::COLUMNMAJOR,
        1> M;
```

load a `mtx` matrix from file:
```
    M.load_from_file("matrix.mtx");
```
Note that the relative path should be from the directory where the binary is executed.

Let's see how the matrix is split between MPI processes in an impicit way:
```
    if (mpi_runner.mpi_rank == 0) {
      std::cout
          << "=============== Testing file load and MPI split ==============="
          << std::endl;
      std::cout << "loaded full matrix:" << std::endl << M << std::endl;
      std::cout << std::endl << std::endl;
      std::cout << "split matrix over MPI processes:" << std::endl;
    }
    M.show_mpi_split();
```

To modify or resize the matrix, the `operator()` and `resize()` method can be used:
```
    M(0, 0) = 10.0;
    M.resize(M.rows() + 1, M.cols() + 1);
```
If any changes are made to the matrix, remember to update the `MPI` configuration by calling:
```
    M.setup_mpi();
```
as for efficiency reasons, the split is not called automatically at every single change.

We are ready to test a common linear algebra operation, the matrix vector multiplication.
Let's define a compatible vector type with the matrix type we are using, it can be retrieved automatically with:
```
    apsc::LinearAlgebra::Language::SparseMatrix<double>::VectorX b(size);
```
then:
```
    b.fill(1.0);
    auto matmul = M * b;
```
***Note***: the vector must be allocated in all the `MPI` processes, this means that
the vector size should be broadcasted to all the processes.

One of the most common operations, Iterative linear solvers:
```
    auto x = M.solve_iterative<apsc::LinearAlgebra::Language::IterativeSolverType::CONJUGATE_GRADIENT>(b);
    x = M.solve_iterative<apsc::LinearAlgebra::Language::IterativeSolverType::GMRES>(b);
    x = M.solve_iterative<apsc::LinearAlgebra::Language::IterativeSolverType::BiCGSTAB>(b);
    x = M.solve_iterative<apsc::LinearAlgebra::Language::IterativeSolverType::SPAI_GMRES>(b);
    x = M.solve_iterative<apsc::LinearAlgebra::Language::IterativeSolverType::SPAI_BiCGSTAB>(b);
```
or direct solvers:
```
    x = M.solve_direct(b);
```

The language user doesn't have to think about `MPI` synchronisation processes while performing operations over `FullMatrix` and `SparseMatrix`.
All the operations between matrices and vectors are made in parallel while possible (`mpi_size` >= 2).
Iterative linear systems are solved in parallel too. 

The parallel speedup that can be achieved is dependent from the class type
(`FullMatrix` vs `SparseMatrix`). In particular when using a sparse format, the
non zero elements and the matrix size are relevant information when choosing
the parallelization setup.

A few benchmarks can be found under `src/logs`.

Please note that in some cases, only the master `mpi_rank` will receive the
correct vector or matrix hence do not take as exact any matrix or vector
created by the library in non master ranks (please see `apsc_language.hpp` documentation for more).
If you need any information in a non master rank, manual data passing must be done.

Now we are ready to compile and run:
```
cd src/demo
mkdir build
cd build
cmake ..
make
mpirun -n 1 main
```

# MPI docker errors
You might experience a strange error when launching `MPI` inside the suggested docker container image:
```
Read -1, expected <someNumber>, errno =1
```
Please refer to [this](https://github.com/feelpp/docker/issues/26).

# apsc::LinearAlgebra::Language
Now let's navigate inside this namespace to understand how the language is built.

## apsc::LinearAlgebra::Language::FullMatrix
This implicit parallel full matrix implementation leverages two main components: `apsc::LinearAlgebra::FullMatrix` and `apsc::LinearAlgebra::MPIFullMatrix`.
`apsc::LinearAlgebra::FullMatrix` is initalized only in the master rank (hence only the master rank will have the matrix data) while `apsc::LinearAlgebra::MPIFullMatrix` is initialised in each `MPI` process.
By calling the `setup_mpi()` method, the underlying classes will perform the required split.

When a methd is called on the `apsc::LinearAlgebra::Language::FullMatrix`, it automatically choose if `MPI` is used or not hence the library undertands the right matrix to use in each case.

At each matrix modification, by calling the same method as before, all the updates will be shared with all `MPI` processes (if used).

This full matrix class offeres two linear solvers, one is with a direct method (by using QR factorisation) and the second one is an iterative Conjugate Gradient method.
The first uses the solver mehtod of `apsc::LinearAlgebra::FullMatrix` wich leverages `Eigen` library and no parallelisation are available.
The latter uses a paralell Conjuagte Gradient method hence this enhancement is implicit by using the language library.
Currently no parallel preconditioners are available.

More specific information can be found inside the source file.

## apsc::LinearAlgebra::Language::SparseMatrix
This implicit parallel sparse matrix implementation leverages two main components: `Eigen::SparseMatrix` and `apsc::LinearAlgebra::MPISparseMatrix`.
`Eigen::SparseMatrix` is initalized only in the master rank (hence only the master rank will have the matrix data) while `apsc::LinearAlgebra::MPISparseMatrix` is initialised in each `MPI` process.
By calling the `setup_mpi()` method, the underlying classes will perform the required split.

When a methd is called on the `apsc::LinearAlgebra::Language::SparseMatrix`, it automatically choose if `MPI` is used or not hence the library undertands the right matrix to use in each case.

At each matrix modification, by calling the same method as before, all the updates will be shared with all `MPI` processes (if used).

This sparse matrix class offeres one direct linear solver:
- `Eigen::SolverQR<>`
and five different types of iterative solvers:
- Conjugate Gradient
- GMRES
- GMRES + SPAI preconditioner
- BiCGSTAB
- BiCGSTAB + SPAI preconditioner

The `SPAI` preconditioner setup is not very trivial and we highly suggest to go through the source file in order to understand more.

More specific information can be found inside the source file.

# Benchmarks

## MPI parallelisation
The parallelisation speed up analysis for full and sparse matices can be found
by running the python scripts under `src/logs`.

The take home message is that while full matrices can leverage a multiprocess
system in a very good way, sparse matrices operations deeply depends on the
matix non zero elements. The number of processes, hence how big local data
structures are as local CPU cache and communication prices are a few big actors
in retrieving the speedup amount.

## SPAI preconditioner
Currently this preconditioner has been used by changing the original linear system from:
\[
Ax = b
\]
to
\[
AMy = b
\]
\[
x = My
\]

The preconditioner setup follow the work in (https://epubs.siam.org/doi/10.1137/S1064827594276552).

If `MPI` is used, the setup process is done in parallel by splitting the matrix column work.

Below some benchmarks can be found by using different `epsilon` values and the GMRES algorithm:

`orsirr_1`
| Epsilon | GMRES |
|---------|-------|
| 0.6     | iter  |
| 0.3     | iters |
| 0.1     | iters |

`orsirr_2`
| Epsilon | GMRES |
|---------|-------|
| 0.6     | iter  |
| 0.3     | iters |
| 0.1     | iters |

`orsreg_1`
| Epsilon | GMRES |
|---------|-------|
| 0.6     | iter  |
| 0.3     | iters |
| 0.1     | iters |


# Section for library maintainers

## File format
In order to maintain a consistent format please format your files with
```
clang-format -style=Google --sort-includes -i path/to/file
```

## A note on Eigen usage
In order to maintain back compability with the `Eigen` version inside the offical
supported docker image (`pcafrica/mk`), `Eigen 3.4` or above features must not
be used.

## Valgrind
Compile the binary with the debug flag `-g3`, and then:
```
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-out.txt [your_executable]
```
