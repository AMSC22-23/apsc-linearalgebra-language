/*
 * MPIContext.hpp
 *
 *  Created on: Nov 30, 2023
 *      Author: Kaixi Matteo Chen
 */

#ifndef MPICONTEXT_HPP
#define MPICONTEXT_HPP

#include <mpi.h>

class MPIContext {
 public:
  MPIContext(MPI_Comm mpi_comm, const int mpi_rank, const int mpi_size = 0)
      : m_mpi_comm(mpi_comm), m_mpi_rank(mpi_rank), m_mpi_size(mpi_size) {}

  MPI_Comm mpi_comm() const { return m_mpi_comm; }

  int mpi_rank() const { return m_mpi_rank; }

  int mpi_size() const { return m_mpi_size; }

 private:
  MPI_Comm m_mpi_comm;
  int m_mpi_rank;
  int m_mpi_size;
};

#endif  // MPICONTEXT_HPP
