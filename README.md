# matvetiterative-chen

Iterative linerar solvers based on distribuited matrix vector multiplication (MPI)

A parallel MPI implementation of different iterative schemes.

## Supported Iterative schemes
- [x] Conjugate Gradient

## Supported preconditioners

## Setup
Initialise submodules

```
git submodule update --init --recursive
```

## Compilation
This repository is intended to be used inside the `pcafrica/mk` docker image
env. See
[here](https://github.com/HPC-Courses/AMSC-Labs/tree/main/Labs/2023-24/lab00-setup)
for the configuration. If so, you don't need to take additional steps.

In case you want to run the project locally, you have to:
- Export the `Eigen` include directory path under the env variable
  `$EIGEN3_INCLUDE_DIR`.
- Have `MPI` installed (Note: we had no issued using version `3.1` of `MPI`,
  newer versions might not compatible)

Then:

```
- cd src
- mkdir build
- cd build
- cmake ..
- make [-j n]
```

## Run `objective1`
```
./your_executable
```

## Run `objective2`
```
mpirun -n [your_core_number] objective2 [problem_size]
```
For example:
```
mpirun -n 2 objective2 1000
```

## Run `objective3`
```
mpirun -n [your_core_number] objective3 [input_file]
```
For example (launched from `build` directory):
```
mpirun -n 2 objective3 ../inputs/0_05fill_size9604.mtx 
```

## File format
In order to maintain a consistent format please format your files with
```
clang-format -style=Google --sort-includes -i path/to/file
```

## MPI docker errors
You might experience a strange error when launching `MPI` inside the suggested docker container image:
```
Read -1, expected <someNumber>, errno =1
```

Please refer to [this](https://github.com/feelpp/docker/issues/26).

## A note on Eigen usage
In order to maintain back compability with the `Eigen` version inside the offical
supported docker image (`pcafrica/mk`), `Eigen 3.4` or above features must not
be used.

## Valgrind
```
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-out.txt [your_executable]
```
