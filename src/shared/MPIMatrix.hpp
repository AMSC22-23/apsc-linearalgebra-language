/*
 * MPIMatrix.hpp
 *
 *  Created on: Oct 15, 2022
 *      Author: forma
 *  Modified on: Nov 29 2023
 *      Author: Kaixi Matteo Chen
 */

#ifndef MPIMATRIX_HPP
#define MPIMATRIX_HPP

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#pragma GCC diagnostic ignored "-Wcast-function-type"
#pragma GCC diagnostic pop
#include <mpi.h>

#include <Eigen/Dense>
#include <EigenStructureMap.hpp>
#include <Parallel/Utilities/mpi_utils.hpp>  // for MPI_SIZE_T and mpi_typeof()
#include <Parallel/Utilities/partitioner.hpp>
#include <Vector.hpp>
#include <vector>
namespace apsc {
/*!
 * A class for parallel matrix product
 * @tparam Matrix A matrix compliant with that in Matrix.hpp
 * @tparam Vector The vector type used for the local solution and internal
 * usages. It must have the following methods:
 * - data(): Returns a pointer to the data buffer
 * - resize(std::size_t): Resizes the vector with the requested length
 * It must have also a range constructor as std::vector
 */
template <class Matrix, class Vector, ORDERINGTYPE ORDER_TYPE>
class MPIMatrix {
 public:
  /*!
   * We assume that Matrix defines a type equal to that of the contained element
   */
  using Scalar = typename Matrix::Scalar;
  /*!
   * All processes will call setup but only the managing process
   * (with number 0) will actually pass a non-empty global matrix
   *
   * @param gMat The global matrix
   * @param communic The MPI communicator
   *
   * @note This is not the only way to create a parallel matrix. Alternatively,
   * all processes may directly build the local matrix, for instance reading
   * from a file. In this case, for the setup we need just the number of rows
   * and columns of the global matrix
   */
  void setup(Matrix const &gMat, MPI_Comm communic) {
    mpi_comm = communic;
    MPI_Comm_rank(mpi_comm, &mpi_rank);
    MPI_Comm_size(mpi_comm, &mpi_size);
    using namespace apsc;
    // This will contain the number of row and columns of all the local matrices
    std::array<std::vector<std::size_t>, 2> localRandC;
    localRandC[0].resize(
        mpi_size);  // get the right size to avoid messing up things
    localRandC[1].resize(mpi_size);
    // We need the tools to split the matrix data buffer
    counts.resize(mpi_size);
    displacements.resize(mpi_size);
    if (mpi_rank == manager) {
      // I am the boss
      global_nRows = gMat.rows();
      global_nCols = gMat.cols();
    }
    // it would be more efficient to pack stuff to be broadcasted
    MPI_Bcast(&global_nRows, 1, MPI_SIZE_T, manager, mpi_comm);
    MPI_Bcast(&global_nCols, 1, MPI_SIZE_T, manager, mpi_comm);

    // I let all tasks compute the partition data, alternative
    // is have it computed only by the master rank and then
    // broadcast. But remember that communication is costly.

    MatrixPartitioner<apsc::DistributedPartitioner, ORDER_TYPE> partitioner(
        global_nRows, global_nCols, mpi_size);  // the partitioner

    auto countAndDisp = apsc::counts_and_displacements(partitioner);
    counts = countAndDisp[0];
    displacements = countAndDisp[1];
    localRandC = partitioner.getLocalRowsAndCols(mpi_size);
    local_nRows = localRandC[0][mpi_rank];
    local_nCols = localRandC[1][mpi_rank];

    // Now get the local matrix!
    localMatrix.resize(local_nRows, local_nCols);
    int matrixSize = local_nRows * local_nCols;
    MPI_Scatterv(gMat.data(), counts.data(), displacements.data(),
                 MPI_Scalar_Type, localMatrix.data(), matrixSize,
                 MPI_Scalar_Type, manager, mpi_comm);
  }
  /*!
   * Performs the local matrix times vector product.
   * For simplicity we do not partition the input vector, which is indeed
   * a global vector.
   *
   * @tparam InputVectorType The input vector type. It must be complian with the
   * localMatrix type in order to compute the product.
   * @param x A global vector
   */
  template <typename InputVectorType>
  void product(InputVectorType const &x) {
    using namespace apsc;
    if constexpr (ORDER_TYPE == ORDERINGTYPE::ROWWISE) {
      // this is the simplest case. The matrix has all column
      // (but not all rows!)
      this->localProduct = this->localMatrix * x;
    } else {
      //  This case is much trickier. The local matrix has fewer columns than
      //  the rows of x,
      // so I need to reduce the global vector in input x
      // I need to get the portion of x corresponding to the columns in the
      // global matrix
      auto startcol = displacements[mpi_rank] / global_nRows;
      auto endcol = (displacements[mpi_rank] + counts[mpi_rank]) / global_nRows;

      // I copy the relevant portion of the global vector in a local vector
      // exploiting a constructor that takes a range.
      LinearAlgebra::Vector<Scalar> y(x.data() + startcol, x.data() + endcol);

      // TODO: In order to remove this check we need to transform the `y`'s type
      // to a template type, where an Eigen vector can be specified
      if constexpr (std::is_base_of_v<Eigen::MatrixBase<Matrix>, Matrix>) {
        auto mapped_y = Eigen::Map<Eigen::VectorXd>(y.data(), y.size());
        this->localProduct = this->localMatrix * mapped_y;
      } else {
        // If you use a non standard Matrix class (hence not compatible with
        // LinearAlgebra::Vector<Scalar> you have to do the required mappings as
        // done for Eigen)
        this->localProduct = this->localMatrix * y;
      }
    }
  }
  /*!
   * Gets the global solution. All processes call it but just process 0
   * (manager) gets a non empty vector equal to the result.
   * The template CollectionVector type must implemen `data()` and `resize()`
   * methods as defined in std::vector.
   *
   * @tparam CollectionVector the result vector type
   * @return the global solution of the matrix product (only process 0), in v.
   */
  template <typename CollectionVector>
  void collectGlobal(CollectionVector &v) const {
    using namespace apsc;
    if (mpi_rank == manager) {
      v.resize(global_nRows);
    }
    if constexpr (ORDER_TYPE == ORDERINGTYPE::ROWWISE) {
      // I need to gather the contribution, but first I need to have
      // find the counts and displacements for the vector
      std::vector<int> vec_counts(mpi_size);
      std::vector<int> vec_displacements(mpi_size);
      for (int i = 0; i < mpi_size; ++i) {
        vec_counts[i] = counts[i] / global_nCols;
        vec_displacements[i] = displacements[i] / global_nCols;
      }
      // note: vec_counts[i] should be equal to the number of local matrix rows.
      MPI_Gatherv(localProduct.data(), localProduct.size(), MPI_Scalar_Type,
                  v.data(), vec_counts.data(), vec_displacements.data(),
                  MPI_Scalar_Type, manager, mpi_comm);
    } else {
      // I need to do a reduction
      // The local vectors are of the richt size, but contain only
      // a partial result. I need to sum up.
      /*
       * int MPI_Reduce(const void* send_buffer,
               void* receive_buffer,
               int count,
               MPI_Datatype datatype,
               MPI_Op operation,
               int root,
               MPI_Comm communicator);
       */
      MPI_Reduce(localProduct.data(), v.data(), global_nRows, MPI_Scalar_Type,
                 MPI_SUM, manager, mpi_comm);
    }
  }
  /*!
   * Gets the global solution. All processes call it and get the
   * result.
   * The template CollectionVector type must implemen `data()` and `resize()`
   * methods as defined in std::vector.
   *
   * @tparam CollectionVector the result vector type
   *
   * @return the global solution of the matrix product, in v.
   */
  template <typename CollectionVector>
  void AllCollectGlobal(CollectionVector &v) const {
    using namespace apsc;
    v.resize(global_nRows);
    if constexpr (ORDER_TYPE == ORDERINGTYPE::ROWWISE) {
      // I need to gather the contribution, but first I need to have
      // find the counts and displacements for the vector
      std::vector<int> vec_counts(mpi_size);
      std::vector<int> vec_displacements(mpi_size);
      for (int i = 0; i < mpi_size; ++i) {
        vec_counts[i] = counts[i] / global_nCols;
        vec_displacements[i] = displacements[i] / global_nCols;
      }
      MPI_Allgatherv(localProduct.data(), localProduct.size(), MPI_Scalar_Type,
                     v.data(), vec_counts.data(), vec_displacements.data(),
                     MPI_Scalar_Type, mpi_comm);
    } else {
      //  This case is trickier. I need to do a reduction
      MPI_Allreduce(localProduct.data(), v.data(), global_nRows,
                    MPI_Scalar_Type, MPI_SUM, mpi_comm);
    }
  }
  /*!
   * Returns the local matrix assigned to the processor
   * @return The local matrix
   */
  auto const &getLocalMatrix() const { return localMatrix; }
  static constexpr int manager = 0;

 protected:
  MPI_Comm mpi_comm;
  int mpi_rank;                    // my rank
  int mpi_size;                    // the number of processes
  std::vector<int> counts;         // The vector used for gathering/scattering
  std::vector<int> displacements;  // The vector used for gathering/scattering
  Matrix localMatrix;              // The local portion of the matrix
  Vector
      localProduct;  // The place where to store the result of the local mult.
  std::size_t local_nRows = 0u;
  std::size_t local_nCols = 0u;
  std::size_t global_nRows = 0u;
  std::size_t global_nCols = 0u;
  // std::size_t offset_Row=0u;
  // std::size_t offset_Col=0u;
  // I use mpi_typeof() in mpi_util.h to recover genericity. Note that to
  // activate the overload I need to create an object of type Scalar. I use the
  // default constructor, withScalar{}
  MPI_Datatype MPI_Scalar_Type = mpi_typeof(Scalar{});
};
}  // end namespace apsc

#endif /* MPIMATRIX_HPP */
