/*
 * Vector.hpp
 *
 *  Created on: Nov 17 2023
 *      Author: Kaixi Matteo Chen
 */

#ifndef VECTOR_HPP
#define VECTOR_HPP
#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#include "utils.hpp"
// To avoid stupid warnings if I do not use openmp
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
namespace apsc::LinearAlgebra {

/*!
 * A full vector
 * @tparam Scalar The type of the element
 */
template <typename SCALAR>
class Vector {
 public:
  using Scalar = SCALAR;

  /*!
   * Constructor may take the size number
   * @param size Length of the vector
   */
  Vector(std::size_t size = 0) : vector_size{size} {
    buffer.resize(size);
    fill(static_cast<Scalar>(0));
  }

  /*!
   * Constructor may take the size number and the initial fill value
   * @param size Length of the vector
   * @param value The initial fill value
   */
  Vector(std::size_t size, Scalar value) : vector_size{size} {
    buffer.resize(size);
    fill(value);
  }

  /*!
   * Copy constructor.
   * @param v The source vector
   */
  Vector(Vector<Scalar> const &v) : vector_size(v.size()) {
    buffer.resize(v.size());
    std::copy(v.buffer.begin(), v.buffer.end(), buffer.begin());
  }

  /*!
   * Move constructor.
   * @param v The source vector
   */
  Vector(const Scalar *begin, const std::size_t length) : vector_size(length) {
    buffer = std::vector(std::make_move_iterator(begin),
                         std::make_move_iterator(begin + length));
  }

  /*!
   * Range constructor.
   * @param begin The pointer to the first buffer element (included)
   * @param end The pointer to the last buffer element (excluded)
   */
  Vector(const Scalar *start, const Scalar *end) {
    buffer = std::vector(start, end);
    vector_size = buffer.size();
  }

  /*!
   * Sets size value
   * @param new_size
   */
  void resize(std::size_t size) {
    vector_size = size;
    buffer.resize(vector_size);
  }

  /*!
   * Fill the buffer with the given value.
   * @param value The value to be filled
   */
  void fill(Scalar value) { std::fill(buffer.begin(), buffer.end(), value); }

  /*!
   * Get element v(i). Const version
   * @param i
   * @return The value
   */
  auto operator()(std::size_t i) const { return buffer[i]; }

  /*!
   * Get element v[i]. Const version
   * @param i
   * @return The value
   */
  auto operator[](std::size_t i) const { return buffer[i]; }

  /*!
   * Get element v(i). Non const version
   *
   * It allows to change the element.
   *
   * @param i
   * @return The value
   */

  auto &operator()(std::size_t i) { return buffer[i]; }

  /*!
   * Get element v[i]. Non const version
   *
   * It allows to change the element.
   *
   * @param i
   * @return The value
   */

  auto &operator[](std::size_t i) { return buffer[i]; }

  /*!
   * Vector size
   * @return the vector size
   */
  auto size() const { return vector_size; };

  /*!
   * Returns the buffer containing the elements of the vector non const
   * version)
   *
   * @note I use decltype(auto) because I want it to return exactly what
   * std::vector::data() returns.
   *
   * @return A pointer to the buffer
   */
  decltype(auto)  // is Scalar *
  data() noexcept {
    return buffer.data();
  }

  /*!
   * Returns the buffer containing the elements of the vector (const version)
   *
   * @note I use decltype(auto) becouse I want it to return exactly what
   * std::vector::data() returns.
   *
   * @return A pointer to the buffer
   */
  decltype(auto)  // is Scalar const *
  data() const noexcept {
    return buffer.data();
  }

  /*!
   * Returns the begin iterator associated to the managed std::vector
   *
   * @return The std::vector begin iterator
   */
  auto begin() const noexcept { return buffer.begin(); }

  /*!
   * Returns the end iterator associated to the managed std::vector
   *
   * @return The std::vector end iterator
   */
  auto end() const noexcept { return buffer.end(); }

  /*!
   * Dot product with a apsc::LinearAlgebra::Vector.
   * The return type is coerent with the lhs vector type.
   *
   * @param v a apsc::LinearAlgebra::Vector vector
   * @return The result of v*v
   */
  template <typename InputVectorScalar>
  Scalar dot(Vector<InputVectorScalar> const &v) const {
    ASSERT(size() == v.size(), "DotProd: Vector sizes does not match");
    // TODO: check how to integrate MPI
    return std::inner_product(buffer.begin(), buffer.end(), v.buffer.begin(),
                              static_cast<Scalar>(0));
  }

  /*!
   * Multiplication (dot product) with a apsc::LinearAlgebra::Vector.
   * The return type is coerent with the lhs vector type.
   *
   * @param v a apsc::LinearAlgebra::Vector vector
   * @return The result of v*v
   */
  template <typename InputVectorScalar>
  Scalar operator*(Vector<InputVectorScalar> const &v) const {
    return dot(v);
  }

  /*!
   * Multiplication by a scalar.
   *
   * @param value A scalar value.
   * @return The result of scalar*v
   */
  Vector<Scalar> operator*(Scalar value) const {
    // TODO: check how to integrate MPI
    Vector<Scalar> res(*this);
    std::transform(res.buffer.begin(), res.buffer.end(), res.buffer.begin(),
                   [&value](auto &c) { return c * value; });
    return res;
  }

  /*!
   * Addition with a apsc::LinearAlgebra::Vector.
   * The return type is coerent with the lhs vector type.
   *
   * @param v a apsc::LinearAlgebra::Vector vector
   * @return The result of v1+v2
   */
  template <typename InputVectorScalar>
  Vector<Scalar> operator+(Vector<InputVectorScalar> const &v) const {
    // TODO: check how to integrate MPI
    Vector<Scalar> res(*this);
    ASSERT(res.size() >= v.size(),
           "Destination vector size is less that lhs vector size");
    std::transform(res.buffer.begin(), res.buffer.end(), v.buffer.begin(),
                   res.buffer.begin(), std::plus<Scalar>());
    return res;
  }

  /*!
   * Subtraction with a apsc::LinearAlgebra::Vector.
   * The return type is coerent with the lhs vector type.
   *
   * @param v a apsc::LinearAlgebra::Vector vector
   * @return The result of v1-v2
   */
  template <typename InputVectorScalar>
  Vector<Scalar> operator-(Vector<InputVectorScalar> const &v) const {
    // TODO: check how to integrate MPI
    Vector<Scalar> res(*this);
    ASSERT(res.size() >= v.size(),
           "Destination vector size is less that lhs vector size");
    std::transform(res.buffer.begin(), res.buffer.end(), v.buffer.begin(),
                   res.buffer.begin(), std::minus<Scalar>());
    return res;
  }

  /*!
   * Multiplication with a Scalar.
   *
   * @param value a scalar value.
   */
  void operator*=(Scalar value) {
    // TODO: check how to integrate MPI
    std::transform(buffer.begin(), buffer.end(), buffer.begin(),
                   [&value](auto &c) { return c * value; });
  }

  /*!
   * Assignment operator.
   *
   * @param v The input vector.
   */
  void operator=(Vector<Scalar> const &v) {
    // TODO: check how to integrate MPI
    vector_size = v.size();
    buffer.resize(vector_size);
    std::copy(v.buffer.begin(), v.buffer.end(), buffer.begin());
  }

  /*!
   * Euclidean norm.
   *
   * @return The computed euclidean norm.
   */
  Scalar norm() const {
    return static_cast<Scalar>(std::sqrt(std::inner_product(
        buffer.begin(), buffer.end(), buffer.begin(), static_cast<Scalar>(0))));
  }

  /*!
   * To read from a file (or any input stream)
   * The input format is
   * nrows ncols
   * a_00 a_01 a_02 ..
   * a_10 a_11 ...
   *
   * @note that carriage returns are in fact ignored. Tokens may be separated by
   * any blank character, as usual in text streams.
   * @param input The input stream to read from
   */
  void readFromStream(std::istream &input);

  /*!
   * The size of the buffer
   * @return the size
   */
  auto bufferSize() const { return buffer.size(); }

  /*!
   * Clears the vector completely and frees memory
   */
  void clear() {
    vector_size = 0u;
    buffer.clear();
    buffer.shrink_to_fit();
  }

 protected:
  std::size_t vector_size = 0u;
  std::vector<Scalar> buffer;
};

/*!
 * To write the vector the output stream
 * @tparam Scalar
 * @param out
 * @param vec
 * @return
 */
template <typename Scalar>
std::ostream &operator<<(std::ostream &out, Vector<Scalar> const &vec);

/*
 * ***************************************************************************
 * Definitions
 * ***************************************************************************
 */

template <typename Scalar>
void Vector<Scalar>::readFromStream(std::istream &input) {
  input >> vector_size;
  resize(vector_size);
  for (std::size_t i = 0u; i < vector_size; ++i) {
    input >> this->operator()(i);
  }

  if (!input.good()) {
    throw std::runtime_error("ERROR: problems while reading from stream\n");
  }
}

template <typename Scalar>
std::ostream &operator<<(std::ostream &out, Vector<Scalar> const &v) {
  if (v.size() < 1) {
    return out;
  }
  out << "[";
  for (std::size_t i = 0u; i < v.size(); ++i) {
    if (i == v.size() - 1) {
      out << v(i);
    } else {
      out << v(i) << ", ";
    }
  }
  out << "]";

  return out;
}

}  // namespace apsc::LinearAlgebra
#pragma GCC diagnostic pop

#endif /* VECTOR_HPP */
