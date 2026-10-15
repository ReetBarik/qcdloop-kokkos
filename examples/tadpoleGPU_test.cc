//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//          Stefan Hoeche   : shoeche@fnal.gov
//          Max Knobbe      : mknobbe@fnal.gov

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include "qcdloop/timer.h"
#include "qcdloop/tadpoleGPU.h"
#include "quad_inputs.h"
#include "ql_launch.h"

using std::vector;
using std::cout;
using std::endl;
using complex = Kokkos::complex<double>;

template<typename T>
std::string arrayToCSV(const T* arr, size_t size) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < size; ++i) {
        if (i > 0) ss << ",";
        ss << arr[i];
    }
    ss << "]";
    return ss.str();
}

std::string complexToCSV(const complex& c) {
    std::stringstream ss;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.16e", c.real());
    ss << "(" << buf << ",";
    std::snprintf(buf, sizeof(buf), "%.16e", c.imag());
    ss << buf << ")";
    return ss.str();
}

double r(double min, double max) {
    return dd_rand(min, max).hi;
}


int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {

      // Parse command line arguments
      int mode = 1; // default value
      int batch_size = 1000000; // default value

      if (argc < 2) {
          std::cout << "Usage: " << argv[0] << " <mode> [batch_size]" << std::endl;
          std::cout << "  mode: 0 for performance benchmark, 1 for accuracy test (required)" << std::endl;
          std::cout << "  batch_size: Number of batch iterations (default: 1000000)" << std::endl;
          Kokkos::finalize();
          return 1;
      }

      // Parse mode (required)
      try {
          mode = std::stoi(argv[1]);
          if (mode != 0 && mode != 1) {
              std::cout << "Error: mode must be 0 or 1. Using default value of 1." << std::endl;
              mode = 1;
          }
      } catch (const std::exception& e) {
          std::cout << "Error: Invalid argument for mode. Using default value of 1." << std::endl;
          mode = 1;
      }

      // Parse batch_size (optional)
      if (argc > 2) {
          try {
              batch_size = std::stoi(argv[2]);
              if (batch_size <= 0) {
                  std::cout << "Error: batch_size must be a positive integer. Using default value of 1000000." << std::endl;
                  batch_size = 1000000;
              }
          } catch (const std::exception& e) {
              std::cout << "Error: Invalid argument for batch_size. Using default value of 1000000." << std::endl;
              batch_size = 1000000;
          }
      }

      if (argc > 3) {
          std::cout << "Usage: " << argv[0] << " <mode> [batch_size]" << std::endl;
          std::cout << "  mode: 0 for performance benchmark, 1 for accuracy test (required)" << std::endl;
          std::cout << "  batch_size: Number of batch iterations (default: 1000000)" << std::endl;
      }

      std::cout << "Running with mode = " << mode << std::endl;
      std::cout << "Running with batch_size = " << batch_size << std::endl;

      if (mode == 0) {
          // Print CSV header for performance benchmark mode
          std::cout << "Target Integral,Batch size,Time" << std::endl;
      } else if (mode == 1) {
          // Print CSV header for accuracy test mode
          std::cout << "Target Integral,Test ID,mu2,ms,ps,Coeff 1,Coeff 2,Coeff 3" << std::endl;
      }

      ql::Timer tt;

      // Create Kokkos Views for batch processing
      Kokkos::View<double*> mu2_d("mu2", batch_size);
      Kokkos::View<double* [1]> m_d("m", batch_size);
      Kokkos::View<double*> p_d("p", 0);  // Empty view for API consistency
      Kokkos::View<complex* [3]> res_d("res", batch_size);

      auto mu2_h = Kokkos::create_mirror_view(mu2_d);
      auto m_h = Kokkos::create_mirror_view(m_d);
      auto p_h = Kokkos::create_mirror_view(p_d);
      auto res_h = Kokkos::create_mirror_view(res_d);

      // Initialize mu2
      for (size_t i = 0; i < batch_size; ++i) {
          mu2_h(i) = dd_mu2().hi;
      }

      ql::LaunchPolicy policy(0, batch_size);

      // TIN0 - zero mass: TP returns all zeros
      for (size_t i(0); i<batch_size; ++i) {
          m_h(i, 0) = 0.;
      }
      Kokkos::deep_copy(mu2_d, mu2_h);
      Kokkos::deep_copy(m_d, m_h);
      tt.start();
      Kokkos::parallel_for("Tadpole TIN0", policy, KOKKOS_LAMBDA(const int& i) {
          ql::TP<complex, double, double>(res_d, mu2_d, m_d, p_d, i);
      });
      Kokkos::fence();
      double elapsed = tt.stop();
      Kokkos::deep_copy(res_h, res_d);
      if (mode == 0) {
          std::cout << "TIN0," << batch_size << "," << elapsed << std::endl;
      } else if (mode == 1) {
          for (size_t i = 0; i < batch_size; ++i) {
              double m_arr[1] = {m_h(i, 0)};
              std::cout << "TIN0," << (i+1) << "," << mu2_h(i) << ","
                        << arrayToCSV(m_arr, 1) << ",[],"
                        << complexToCSV(res_h(i, 0)) << ","
                        << complexToCSV(res_h(i, 1)) << ","
                        << complexToCSV(res_h(i, 2)) << std::endl;
          }
      }

      // TIN1 - one mass: TP0, I(m^2) = m^2 * (mu^2/m^2)^eps * [1/eps + 1]
      // Randomize m per element so each batch element exercises a different mass value.
      std::srand(12345);
      for (size_t i(0); i<batch_size; ++i) {
          m_h(i, 0) = r(100, 1000000);
      }
      Kokkos::deep_copy(mu2_d, mu2_h);
      Kokkos::deep_copy(m_d, m_h);
      tt.start();
      Kokkos::parallel_for("Tadpole TIN1", policy, KOKKOS_LAMBDA(const int& i) {
          ql::TP<complex, double, double>(res_d, mu2_d, m_d, p_d, i);
      });
      Kokkos::fence();
      elapsed = tt.stop();
      Kokkos::deep_copy(res_h, res_d);
      if (mode == 0) {
          std::cout << "TIN1," << batch_size << "," << elapsed << std::endl;
      } else if (mode == 1) {
          for (size_t i = 0; i < batch_size; ++i) {
              double m_arr[1] = {m_h(i, 0)};
              std::cout << "TIN1," << (i+1) << "," << mu2_h(i) << ","
                        << arrayToCSV(m_arr, 1) << ",[],"
                        << complexToCSV(res_h(i, 0)) << ","
                        << complexToCSV(res_h(i, 1)) << ","
                        << complexToCSV(res_h(i, 2)) << std::endl;
          }
      }
  }

  Kokkos::finalize();
  return 0;
}
