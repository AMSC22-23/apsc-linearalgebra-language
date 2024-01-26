/**
 * @file EigenStructureMap.hpp
 * @brief Header file containing the EigenStructureMap class for mapping custom data to Eigen structures.
 */

#ifndef EIGEN_MATRIX_MAP_HPP
#define EIGEN_MATRIX_MAP_HPP

#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace apsc::LinearAlgebra
{
/**
 * @class EigenStructureMap
 * @brief A full Eigen-compatible matrix class with custom-handled buffer data.
 * This class is meant to call Eigen methods on data not owned by Eigen, in
 * order to avoid memory movements.
 * @tparam EigenStructure The mapped Eigen type (e.g., MatrixX<>, VectorX<>)
 * @tparam Scalar The scalar type
 * @tparam MappedMatrix The custom matrix type that owns the data buffer
 */
template <typename EigenStructure, typename Scalar,
          typename MappedMatrix = Scalar *>
class EigenStructureMap {
 public:
  /**
   * @brief Creates an EigenStructureMap object from the given data buffer and size.
   * @param m The data buffer.
   * @param size The size of the buffer.
   * @return An EigenStructureMap object mapped to the given buffer.
   */
  static EigenStructureMap<EigenStructure, Scalar, MappedMatrix> create_map(
      MappedMatrix const &m, const std::size_t size) {
    Scalar *data =
        const_cast<Scalar *>(m.data());  // const versioon is called, why?

    static_assert(std::is_same_v<decltype(data), Scalar *>,
                  "Mapping different scalar types");
    return EigenStructureMap<EigenStructure, Scalar, MappedMatrix>(data, size);
  }
  /**
   * @brief Creates an EigenStructureMap object from the given data buffer, rows, and columns.
   * @param m The data buffer.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @return An EigenStructureMap object mapped to the given buffer.
   */
  static EigenStructureMap<EigenStructure, Scalar, MappedMatrix> create_map(
      MappedMatrix const &m, const std::size_t rows, const std::size_t cols) {
    Scalar *data =
        const_cast<Scalar *>(m.data());  // const versioon is called, why?

    static_assert(std::is_same_v<decltype(data), Scalar *>,
                  "Mapping different scalar types");
    return EigenStructureMap<EigenStructure, Scalar, MappedMatrix>(data, rows,
                                                                   cols);
  }
  /**
   * @brief Creates an EigenStructureMap object from the given data pointers and matrix dimensions.
   * @tparam IndexType The type of indices.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param nnz Number of non-zero elements.
   * @param outer_size_ptr Pointer to the outer size (row size for compressed matrices).
   * @param inner_size_ptr Pointer to the inner size (column size for compressed matrices).
   * @param value_ptr Pointer to the values of non-zero elements.
   * @return An EigenStructureMap object mapped to the given data.
   */
  template <typename IndexType>
  static EigenStructureMap<EigenStructure, Scalar, MappedMatrix> create_map(
      const std::size_t rows, const std::size_t cols, std::size_t nnz,
      IndexType *outer_size_ptr, IndexType *inner_size_ptr, Scalar *value_ptr) {
    return EigenStructureMap<EigenStructure, Scalar, MappedMatrix>(
        rows, cols, nnz, outer_size_ptr, inner_size_ptr, value_ptr);
  }
  /**
   * @brief Constructs an EigenStructureMap object from the given data buffer and size.
   * @param data The data buffer.
   * @param size The size of the buffer.
   */
  EigenStructureMap(Scalar *data, const std::size_t size)
      : structure_map(data, size) {}
  /**
   * @brief Constructs an EigenStructureMap object from the given data buffer, rows, and columns.
   * @param data The data buffer.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  EigenStructureMap(Scalar *data, const std::size_t rows,
                    const std::size_t cols)
      : structure_map(data, rows, cols) {}
  /**
   * @brief Accessor for the mapped Eigen structure.
   * @return Reference to the mapped Eigen structure.
   */
  auto structure() { return structure_map; }

 protected:
  Eigen::Map<EigenStructure> structure_map;

  template <typename IndexType>
  EigenStructureMap(uint32_t rows, uint32_t cols, uint32_t nnz,
                    IndexType *outer_size_ptr, IndexType *inner_size_ptr,
                    Scalar *value_ptr)
      : structure_map(rows, cols, nnz, outer_size_ptr, inner_size_ptr,
                      value_ptr) {}
};

}

#endif  // EIGEN_MATRIX_MAP_HPP
