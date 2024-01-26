/**
 * @file assert.hpp
 * @brief Header file containing a definition of an assertion macro.
 * @author Kaixi Matteo Chen
 */
#ifndef ASSERT_HPP
#define ASSERT_HPP

#include <iostream>

#define ASSERT(condition, message)                                       \
  do {                                                                   \
    if (!(condition)) {                                                  \
      std::cerr << "Assertion `" #condition "` failed in " << __FILE__   \
                << " line " << __LINE__ << ": " << message << std::endl; \
      std::terminate();                                                  \
    }                                                                    \
  } while (false)

#endif  // ASSERT_HPP
