/**
 * @file ObjectiveContext.hpp
 * @brief Header file containing the objective context class for managing
 * objective-related information and reporting.
 * @author Kaixi Matteo Chen
 */

#ifndef OBJECTIVE_CONTEXT_HPP
#define OBJECTIVE_CONTEXT_HPP

#include <stdint.h>

#include <fstream>
#include <iostream>

/**
 * @class ObjectiveContext
 * @brief Class for managing objective-related information and reporting.
 */
class ObjectiveContext {
 private:
  uint8_t m_objective_number;     /**< The objective number. */
  uint8_t m_mpi_sie;              /**< The MPI size. */
  std::string m_report_file_name; /**< The name of the report file. */
  std::string m_problem_name;     /**< The name of the problem. */
  std::ofstream m_report_file;    /**< Output file stream for reporting. */

  /**
   * @brief Check if the problem name is provided.
   * @return True if the problem name is provided, false otherwise.
   */
  bool show_problem_name() { return m_problem_name.length(); }

 public:
  /**
   * @brief Default constructor.
   */
  ObjectiveContext() = default;
  /**
   * @brief Constructor.
   * @param objective_number The objective number.
   * @param mpi_size The MPI size.
   * @param report_file_name The name of the report file.
   * @param problem_name The name of the problem (optional, default is an empty
   * string).
   */
  ObjectiveContext(const uint8_t objective_number, const uint8_t mpi_size,
                   const std::string report_file_name,
                   const std::string problem_name = "")
      : m_objective_number(objective_number),
        m_mpi_sie(mpi_size),
        m_report_file_name(report_file_name),
        m_problem_name(problem_name) {}
  /**
   * @brief Write data to the report file.
   * @tparam Var Variable argument types.
   * @param vars Variable arguments to write to the report file.
   */
  template <typename... Var>
  void write(Var... vars) {
    std::streamsize report_file_size;
    std::ifstream report_file(m_report_file_name, std::ios::ate);
    if (report_file.is_open()) {
      report_file_size = report_file.tellg();
      report_file.close();
    } else {
      report_file_size = 0;
      // std::cerr << "Failed to open output file" << std::endl;
      // return;
    }
    m_report_file.open(m_report_file_name, std::ios::app);
    if (!m_report_file.is_open()) {
      std::cerr << "Failed to open output file" << std::endl;
      return;
    }

    // write headers
    if (report_file_size == 0) {
      if (show_problem_name()) {
        m_report_file << "PROBLEM_NAME,";
      }
      m_report_file << "SIZE,TIME(microseconds),ITERATIONS,FLAG" << std::endl;
    }

    // write content
    if (show_problem_name()) {
      m_report_file << m_problem_name << ',';
    }
    ((m_report_file << vars), ...) << std::endl;
    m_report_file.close();
  }
};

#endif  // OBJECTIVE_CONTEXT_HPP
