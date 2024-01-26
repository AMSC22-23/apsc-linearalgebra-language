/**
 * @file MPIContext.hpp
 * @brief Header file containing the MPIContext class for encapsulating MPI
 * context information.
 * @author Kaixi Matteo Chen
 */

#ifndef MPICONTEXT_HPP
#define MPICONTEXT_HPP

#include <mpi.h>
/**
 * @class MPIContext
 * @brief Class for encapsulating MPI context information.
 */
class MPIContext {
 public:
  /**
   * @brief Constructor.
   * @param mpi_comm The MPI communicator.
   * @param mpi_rank The rank of the MPI process.
   * @param mpi_size The size of the MPI communicator (optional, default is 0).
   */
  MPIContext(MPI_Comm mpi_comm, const int mpi_rank, const int mpi_size = 0)
      : m_mpi_comm(mpi_comm), m_mpi_rank(mpi_rank), m_mpi_size(mpi_size) {}
  /**
   * @brief Get the MPI communicator.
   * @return The MPI communicator.
   */
  MPI_Comm mpi_comm() const { return m_mpi_comm; }
  /**
   * @brief Get the rank of the MPI process.
   * @return The rank of the MPI process.
   */
  int mpi_rank() const { return m_mpi_rank; }
  /**
   * @brief Get the size of the MPI communicator.
   * @return The size of the MPI communicator.
   */
  int mpi_size() const { return m_mpi_size; }

 private:
  MPI_Comm m_mpi_comm; /**< MPI communicator. */
  int m_mpi_rank;      /**< Rank of the MPI process. */
  int m_mpi_size;      /**< Size of the MPI communicator. */
};

#endif  // MPICONTEXT_HPP
