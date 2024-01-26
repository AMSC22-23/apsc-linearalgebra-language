# Custom Classes for Linear Algebra Operations

## `EigenStructureMap.hpp`

`EigenStructureMap` is a helper class designed to facilitate the usage of
`Eigen` matrix class methods on non-`Eigen` matrices, such as
`MatrixWithVecSupport`. This class allows you to leverage the powerful
functionalities of `Eigen` without the need for memory movements when operating
on non-`Eigen` data.

For comprehensive details and usage examples, please refer to the official
[Eigen documentation on Mapping External
Structures](http://www.eigen.tuxfamily.org/dox/group__TutorialMapClass.html).
The key concept behind `EigenStructureMap` is to seamlessly apply `Eigen`
methods to non-`Eigen` data, ensuring efficient and memory-friendly operations.

## `MatrixWithVecSupport.hpp`

`MatrixWithVecSupport` extends the functionality of
`AMSC-CodeExamples/Examples/src/Matrix/Matrix.hpp`. This full matrix class
introduces support for the `operator*` to facilitate matrix-vector
multiplication with `Vector.hpp` objects. Additionally, it includes a `solve`
method, providing a straightforward approach to solving linear systems using
direct solvers.

## `Vector.hpp`

`Vector` is a comprehensive vector class definition inspired by
`AMSC-CodeExamples/Examples/src/Matrix/Matrix.hpp`. It currently supports
multiplication operations, specifically when performing matrix-vector
multiplications between instances of `MatrixWithVecSupport.hpp` and
`Vector.hpp`.

## `MPIMatrix.hpp`
This `MPI` compatible matrix class is inspired by `AMSC-CodeExamples/Examples/src/Parallel/MPI/PMatirx.hpp`.
This updated class can use a custom `Vector` class as local solution buffer and can collect the result in
a custom `Vector` class.

The only restrictions are:
- `Vector` must implement `data()` method in order to expose a pointer to its buffer.
- `Vector` must implement `resize()` method in order to be resized when needed.

## Demo Files

You can find them in `matvetcg-chen-zhang-guo/src/demo`.

A set of demonstration `cpp` files is provided to showcase the usage of these
objects. Feel free to explore the demos to gain a better understanding of how
these custom classes can be effectively employed in your linear algebra
operations.
