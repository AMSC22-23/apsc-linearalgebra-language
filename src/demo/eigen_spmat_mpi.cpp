#include <mpi.h>

#include <Eigen/Sparse>
#include <iostream>

void MPI_row_partition() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    std::cout << "Row partition\n";
  }

  // Create a sparse matrix on the root process
  Eigen::SparseMatrix<double> sparseMatrix;
  if (rank == 0) {
    // Initialize the sparse matrix with data on the root process
    // For example, you can use the insert() method to add non-zero elements
    sparseMatrix.resize(5, 5);
    sparseMatrix.insert(0, 0) = 1.0;
    sparseMatrix.insert(1, 1) = 2.0;
    sparseMatrix.insert(2, 2) = 3.0;
    sparseMatrix.insert(3, 3) = 4.0;
    sparseMatrix.insert(4, 4) = 5.0;
  }

  // Broadcast the size of the sparse matrix
  int rows, cols, nnz;
  if (rank == 0) {
    rows = sparseMatrix.rows();
    cols = sparseMatrix.cols();
    nnz = sparseMatrix.nonZeros();
  }
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Resize the sparse matrix on all processes
  sparseMatrix.resize(rows, cols);

  // Determine the range of rows each process is responsible for
  int localRowCount = rows / size;
  int startRow = rank * localRowCount;
  int endRow = (rank == size - 1) ? rows : startRow + localRowCount;

  // Broadcast only the portion of the matrix that each process is responsible
  // for
  Eigen::SparseMatrix<double> localMatrix(endRow - startRow, cols);
  std::vector<double> values(nnz);
  std::vector<int> innerIndices(nnz);

  if (rank == 0) {
    values.assign(sparseMatrix.valuePtr(), sparseMatrix.valuePtr() + nnz);
    innerIndices.assign(sparseMatrix.innerIndexPtr(),
                        sparseMatrix.innerIndexPtr() + nnz);
  }

  MPI_Bcast(values.data(), nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(innerIndices.data(), nnz, MPI_INT, 0, MPI_COMM_WORLD);

  // Populate the local portion of the sparse matrix on all processes
  for (int k = 0; k < nnz; ++k) {
    if (innerIndices[k] >= startRow && innerIndices[k] < endRow) {
      localMatrix.insert(innerIndices[k] - startRow, k % cols) = values[k];
    }
  }

  // Perform operations on the local portion of the sparse matrix
  for (int i = 0; i < size; i++) {
    if (rank == i) {
      std::cout << localMatrix << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

void MPI_col_partition() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    std::cout << "Col partition\n";
  }

  // Create a sparse matrix on the root process
  Eigen::SparseMatrix<double> sparseMatrix;
  if (rank == 0) {
    // Initialize the sparse matrix with data on the root process
    // For example, you can use the insert() method to add non-zero elements
    sparseMatrix.resize(5, 5);
    sparseMatrix.insert(0, 0) = 1.0;
    sparseMatrix.insert(1, 1) = 2.0;
    sparseMatrix.insert(2, 2) = 3.0;
    sparseMatrix.insert(3, 3) = 4.0;
    sparseMatrix.insert(4, 4) = 5.0;
  }

  // Broadcast the size of the sparse matrix
  int rows, cols, nnz;
  if (rank == 0) {
    rows = sparseMatrix.rows();
    cols = sparseMatrix.cols();
    nnz = sparseMatrix.nonZeros();
  }
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Resize the sparse matrix on all processes
  sparseMatrix.resize(rows, cols);

  // Determine the range of columns each process is responsible for
  int localColCount = cols / size;
  int startCol = rank * localColCount;
  int endCol = (rank == size - 1) ? cols : startCol + localColCount;

  // Broadcast only the portion of the matrix that each process is responsible
  // for
  Eigen::SparseMatrix<double> localMatrix(rows, endCol - startCol);
  std::vector<double> values(nnz);
  std::vector<int> innerIndices(nnz);

  if (rank == 0) {
    values.assign(sparseMatrix.valuePtr(), sparseMatrix.valuePtr() + nnz);
    innerIndices.assign(sparseMatrix.innerIndexPtr(),
                        sparseMatrix.innerIndexPtr() + nnz);
  }

  MPI_Bcast(values.data(), nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(innerIndices.data(), nnz, MPI_INT, 0, MPI_COMM_WORLD);

  // Populate the local portion of the sparse matrix on all processes
  for (int k = 0; k < nnz; ++k) {
    if (innerIndices[k] >= startCol && innerIndices[k] < endCol) {
      localMatrix.insert(innerIndices[k] % rows, innerIndices[k] - startCol) =
          values[k];
    }
  }

  // Perform operations on the local portion of the sparse matrix
  for (int i = 0; i < size; i++) {
    if (rank == i) {
      std::cout << localMatrix << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

int main(int argc, char **argv) {
  MPI_Init(0, 0);
  MPI_row_partition();
  MPI_col_partition();
  MPI_Finalize();
  return 0;
}
