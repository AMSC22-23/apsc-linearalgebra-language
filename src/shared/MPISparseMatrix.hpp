/*
 * MPIMatrix.hpp
 *
 *  Created on: Dec 3 2023
 *      Author: Kaixi Matteo Chen
 */

#ifndef MPISPARSEMATRIX_HPP
#define MPISPARSEMATRIX_HPP

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#pragma GCC diagnostic ignored "-Wcast-function-type"
#pragma GCC diagnostic pop
#include <mpi.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Parallel/Utilities/mpi_utils.hpp>  // for MPI_SIZE_T and mpi_typeof()
#include <Parallel/Utilities/partitioner.hpp>
#include <Vector.hpp>
#include <array>
#include <vector>
namespace apsc {
/*!
 * A class for parallel sparse matrix product
 * @tparam Matrix A sparse matrix class
 * @tparam Vector The vector type used for the local solution and internal
 * usages. It must have the following methods:
 * - data(): Returns a pointer to the data buffer
 * - resize(std::size_t): Resizes the vector with the requested length
 */
template <class Matrix, class Vector, ORDERINGTYPE ORDER_TYPE>
class MPISparseMatrix {
 public:
  /*!
   * We assume that Matrix defines a type equal to that of the contained
   * element
   */
  using Scalar = typename Matrix::Scalar;
  /*!
   * All processes will call setup but only the managing process
   * (with number 0) will actually pass a non-empty global matrix.
   * Currently only Eigen::SparseMatrix class is supported.
   *
   * @tparam ORDER The input sparse matrix ordering type
   * @param compressed_global_sparse_matrix The global matrix in compressed mode
   * (https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))
   * @param communicator The MPI communicator
   *
   * @note This is not the only way to create a parallel matrix. Alternatively,
   * all processes may directly build the local matrix, for instance reading
   * from a file. In this case, for the setup we need just the number of rows
   * and columns of the global matrix
   */
  void setup(Matrix const &compressed_global_sparse_matrix,
             MPI_Comm communicator) {
    mpi_comm = communicator;
    MPI_Comm_rank(mpi_comm, &mpi_rank);
    MPI_Comm_size(mpi_comm, &mpi_size);
    using namespace apsc;

    if constexpr (!std::is_base_of_v<Eigen::SparseCompressedBase<Matrix>,
                                     Matrix>) {
      throw std::invalid_argument(
          "Only Eigen::SparseMatrix class is supported");
    }

    // Retrive safely the Eigen index type, by default it is int
    using EigenIndexType = std::remove_const_t<typename std::remove_reference<
        decltype(compressed_global_sparse_matrix.innerIndexPtr()[0])>::type>;

    // Accept only Eigen compressed sparse matrix
    bool dead_signal = 0;
    if constexpr (std::is_base_of_v<Eigen::SparseCompressedBase<Matrix>,
                                    Matrix>) {
      if (mpi_rank == 0 && !compressed_global_sparse_matrix.isCompressed()) {
        dead_signal = 1;
      }
      MPI_Bcast(&dead_signal, 1, mpi_typeof(decltype(dead_signal){}), 0,
                mpi_comm);
      if (dead_signal) {
        throw std::invalid_argument(
            "Received a not compressed Eigen::SparseMatrix");
      }
      MPI_Barrier(mpi_comm);
    }

    // This will contain the number of row and columns of all the local matrices
    std::array<std::vector<std::size_t>, 2> localRandC;
    localRandC[0].resize(
        mpi_size);  // get the right size to avoid messing up things
    localRandC[1].resize(mpi_size);
    // We need the tools to split the matrix data buffer
    counts.resize(mpi_size);
    displacements.resize(mpi_size);

    if (mpi_rank == manager) {
      global_num_rows = static_cast<decltype(global_num_rows)>(
          compressed_global_sparse_matrix.rows());
      global_num_cols = static_cast<decltype(global_num_cols)>(
          compressed_global_sparse_matrix.cols());
      global_non_zero = static_cast<decltype(global_non_zero)>(
          compressed_global_sparse_matrix.nonZeros());
      global_outer_size = static_cast<decltype(global_outer_size)>(
          compressed_global_sparse_matrix.outerSize() + 1);
    }
    // it would be more efficient to pack stuff to be broadcasted
    MPI_Bcast(&global_num_rows, 1, MPI_SIZE_T, manager, mpi_comm);
    MPI_Bcast(&global_num_cols, 1, MPI_SIZE_T, manager, mpi_comm);
    MPI_Bcast(&global_non_zero, 1, MPI_SIZE_T, manager, mpi_comm);
    MPI_Bcast(&global_outer_size, 1, MPI_SIZE_T, manager, mpi_comm);

    MatrixPartitioner<apsc::DistributedPartitioner, ORDER_TYPE> partitioner(
        global_num_rows, global_num_cols, mpi_size);

    auto countAndDisp = apsc::counts_and_displacements(partitioner);
    counts = countAndDisp[0];
    displacements = countAndDisp[1];
    localRandC = partitioner.getLocalRowsAndCols(mpi_size);
    local_num_rows = localRandC[0][mpi_rank];
    local_num_cols = localRandC[1][mpi_rank];

    local_matrix.resize(local_num_rows, local_num_cols);

    std::vector<Scalar> global_mat_values_ptr(global_non_zero);
    std::vector<EigenIndexType> global_mat_inner_index_ptr(global_non_zero);
    std::vector<EigenIndexType> global_mat_outer_index_ptr(global_outer_size);
    if (mpi_rank == 0) {
      global_mat_values_ptr.assign(
          compressed_global_sparse_matrix.valuePtr(),
          compressed_global_sparse_matrix.valuePtr() + global_non_zero);
      global_mat_inner_index_ptr.assign(
          compressed_global_sparse_matrix.innerIndexPtr(),
          compressed_global_sparse_matrix.innerIndexPtr() + global_non_zero);
      global_mat_outer_index_ptr.assign(
          compressed_global_sparse_matrix.outerIndexPtr(),
          compressed_global_sparse_matrix.outerIndexPtr() + global_outer_size);
    }
    MPI_Bcast(global_mat_values_ptr.data(), global_non_zero,
              mpi_typeof(Scalar{}), 0, MPI_COMM_WORLD);
    MPI_Bcast(global_mat_inner_index_ptr.data(), global_non_zero, MPI_INT, 0,
              MPI_COMM_WORLD);
    MPI_Bcast(global_mat_outer_index_ptr.data(), global_outer_size, MPI_INT, 0,
              MPI_COMM_WORLD);

    const EigenIndexType it_start = displacements[mpi_rank] / global_num_cols;
    const EigenIndexType it_end =
        it_start + (counts[mpi_rank] / global_num_cols);
    for (EigenIndexType i = it_start, local_i_start = 0; i < it_end;
         i++, ++local_i_start) {
      const EigenIndexType k_start = global_mat_outer_index_ptr[i];
      const EigenIndexType k_end = global_mat_outer_index_ptr[i + 1];
      for (EigenIndexType k = k_start; k < k_end; k++) {
        EigenIndexType j = global_mat_inner_index_ptr[k];
        Scalar v = global_mat_values_ptr[k];
        if constexpr (ORDER_TYPE == ORDERINGTYPE::ROWWISE) {
          local_matrix.coeffRef(local_i_start, j) = v;
        } else {
          local_matrix.coeffRef(j, local_i_start) = v;
        }
      }
    }
  }
  /*!
   * Performs the local matrix times vector product.
   * For simplicity we do not partition the input vector, which is indeed
   * a global vector.
   *
   * @tparam InputVectorType The input vector type. It must be complian with the
   * local_matrix type in order to compute the product.
   * @param x A global vector of type InputVectorType.
   */
  template <typename InputVectorType>
  void product(InputVectorType const &x) {
    using namespace apsc;
    if constexpr (ORDER_TYPE == ORDERINGTYPE::ROWWISE) {
      // this is the simplest case. The matrix has all column
      // (but not all rows!)
      this->local_product = this->local_matrix * x;
    } else {
      //  This case is much trickier. The local matrix has fewer columns than
      //  the rows of x,
      // so I need to reduce the global vector in input x
      // I need to get the portion of x corresponding to the columns in the
      // global matrix
      auto startcol = displacements[mpi_rank] / global_num_rows;
      auto endcol =
          (displacements[mpi_rank] + counts[mpi_rank]) / global_num_rows;

      // I copy the relevant portion of the global vector in a local vector
      // exploiting a constructor that takes a range.
      LinearAlgebra::Vector<Scalar> y(x.data() + startcol, x.data() + endcol);

      // TODO: In order to remove this check we need to transform the `y`'s type
      // to a template type, where an Eigen vector can be specified
      if constexpr (std::is_base_of_v<Eigen::SparseCompressedBase<Matrix>,
                                      Matrix>) {
        auto mapped_y = Eigen::Map<Eigen::VectorXd>(y.data(), y.size());
        this->local_product = this->local_matrix * mapped_y;
      } else {
        // If you use a non standard Matrix class (hence not compatible with
        // LinearAlgebra::Vector<Scalar> you have to do the required mappings as
        // done for Eigen)
        this->local_product = this->local_matrix * y;
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
      v.resize(global_num_rows);
    }
    if constexpr (ORDER_TYPE == ORDERINGTYPE::ROWWISE) {
      // I need to gather the contribution, but first I need to have
      // find the counts and displacements for the vector
      std::vector<int> vec_counts(mpi_size);
      std::vector<int> vec_displacements(mpi_size);
      for (int i = 0; i < mpi_size; ++i) {
        vec_counts[i] = counts[i] / global_num_cols;
        vec_displacements[i] = displacements[i] / global_num_cols;
      }
      // note: vec_counts[i] should be equal to the number of local matrix rows.
      MPI_Gatherv(local_product.data(), local_product.size(), MPI_Scalar_Type,
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
      MPI_Reduce(local_product.data(), v.data(), global_num_rows,
                 MPI_Scalar_Type, MPI_SUM, manager, mpi_comm);
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
    v.resize(global_num_rows);
    if constexpr (ORDER_TYPE == ORDERINGTYPE::ROWWISE) {
      // I need to gather the contribution, but first I need to have
      // find the counts and displacements for the vector
      std::vector<int> vec_counts(mpi_size);
      std::vector<int> vec_displacements(mpi_size);
      for (int i = 0; i < mpi_size; ++i) {
        vec_counts[i] = counts[i] / global_num_cols;
        vec_displacements[i] = displacements[i] / global_num_cols;
      }
      MPI_Allgatherv(local_product.data(), local_product.size(),
                     MPI_Scalar_Type, v.data(), vec_counts.data(),
                     vec_displacements.data(), MPI_Scalar_Type, mpi_comm);
    } else {
      //  This case is trickier. I need to do a reduction
      MPI_Allreduce(local_product.data(), v.data(), global_num_rows,
                    MPI_Scalar_Type, MPI_SUM, mpi_comm);
    }
  }
  /*!
   * Returns the local matrix assigned to the processor
   * @return The local matrix
   */
  auto const &getLocalMatrix() const { return local_matrix; }
  static constexpr int manager = 0;

 protected:
  MPI_Comm mpi_comm;
  int mpi_rank;                    // my rank
  int mpi_size;                    // the number of processes
  std::vector<int> counts;         // The vector used for gathering/scattering
  std::vector<int> displacements;  // The vector used for gathering/scattering
  Matrix local_matrix;             // The local portion of the matrix
  Vector
      local_product;  // The place where to store the result of the local mult.
  std::size_t local_num_rows = 0u;
  std::size_t local_num_cols = 0u;
  std::size_t global_num_rows = 0u;
  std::size_t global_num_cols = 0u;
  std::size_t global_non_zero = 0u;    // Eigen::SparseMatrix::nonZeros()
  std::size_t global_outer_size = 0u;  // Eigen::SparseMatrix::outerSize()
  // std::size_t offset_Row=0u;
  // std::size_t offset_Col=0u;
  // I use mpi_typeof() in mpi_util.h to recover genericity. Note that to
  // activate the overload I need to create an object of type Scalar. I use the
  // default constructor, with Scalar{}
  MPI_Datatype MPI_Scalar_Type = mpi_typeof(Scalar{});
};
}  // end namespace apsc

#endif /* MPISPARSEMATRIX_HPP */
