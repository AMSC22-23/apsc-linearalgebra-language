/*
 * EigenStructureMap.hpp
 *
 *  Created on: Nov 18, 2023
 *      Author: Kaixi Matteo Chen
 */

#ifndef EIGEN_MATRIX_MAP_HPP
#define EIGEN_MATRIX_MAP_HPP

#include <Eigen/Dense>
#include <cstddef>
#include <type_traits>

/*!
 * A full Eigen compatible matrix class with custom handled buffer data.
 * This class is meant to call Eigen methods on data now owned by Eigen, in
 * order to avoid memory movements. Refer to
 * http://www.eigen.tuxfamily.org/dox/group__TutorialMapClass.html
 *
 * @tparam EigenStructure The mapped Eigen type (MatrixX<>, VectorX<>, ...)
 * @tparam Scalar The scalar type
 * @tparam MappedMatrix The custom matrix type who owns the data buffer
 */
template <typename EigenStructure, typename Scalar,
          typename MappedMatrix = Scalar *>
class EigenStructureMap {
 public:
  // TODO: consider using cpp concempts for MappedMatrix type
  static EigenStructureMap<EigenStructure, Scalar, MappedMatrix> create_map(
      MappedMatrix const &m, const std::size_t size) {
    Scalar *data =
        const_cast<Scalar *>(m.data());  // const versioon is called, why?

    static_assert(std::is_same_v<decltype(data), Scalar *>,
                  "Mapping different scalar types");
    return EigenStructureMap<EigenStructure, Scalar, MappedMatrix>(data, size);
  }

  static EigenStructureMap<EigenStructure, Scalar, MappedMatrix> create_map(
      MappedMatrix const &m, const std::size_t rows, const std::size_t cols) {
    Scalar *data =
        const_cast<Scalar *>(m.data());  // const versioon is called, why?

    static_assert(std::is_same_v<decltype(data), Scalar *>,
                  "Mapping different scalar types");
    return EigenStructureMap<EigenStructure, Scalar, MappedMatrix>(data, rows,
                                                                   cols);
  }

  EigenStructureMap(Scalar *data, const std::size_t size)
      : structure_map(data, size) {}

  EigenStructureMap(Scalar *data, const std::size_t rows,
                    const std::size_t cols)
      : structure_map(data, rows, cols) {}

  auto structure() { return structure_map; }

 protected:
  Eigen::Map<EigenStructure> structure_map;
};

#endif  // EIGEN_MATRIX_MAP_HPP
