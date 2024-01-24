#ifndef OBJECTIVE_CONTEXT_HPP
#define OBJECTIVE_CONTEXT_HPP

#include <stdint.h>

#include <fstream>
#include <iostream>

class objective_context {
 private:
  uint8_t m_objective_number;
  uint8_t m_mpi_sie;
  std::string m_report_file_name;
  std::string m_problem_name;
  std::ofstream m_report_file;

  bool show_problem_name() { return m_problem_name.length(); }

 public:
  objective_context(const uint8_t objective_number, const uint8_t mpi_size,
                    const std::string report_file_name,
                    const std::string problem_name = "")
      : m_objective_number(objective_number),
        m_mpi_sie(mpi_size),
        m_report_file_name(report_file_name),
        m_problem_name(problem_name) {}

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
