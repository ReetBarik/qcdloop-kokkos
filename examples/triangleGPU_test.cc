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
#include "qcdloop/triangleGPU.h"
#include "quad_inputs.h"
#include "ql_launch.h"

using std::vector;
using std::cout;
using std::endl;
using std::string;
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

double rs(double min, double max) {
    return dd_rands(min, max).hi;
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

        double low = 100;
        double up  = 1000000;

        // Create Kokkos Views for batch processing
        Kokkos::View<double*> mu2_d("mu2", batch_size);
        Kokkos::View<double* [3]> m_d("m", batch_size);
        Kokkos::View<double* [3]> p_d("p", batch_size);
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

        // Mass values used across test cases
        double m2  = dd_mass_10().hi;
        double m12 = dd_mass_4p9_sq().hi;
        double m32 = dd_mass_50_sq().hi;

        // Trigger TIN0 - TIN3: sweep over number of internal masses
        // With all-random nonzero p (|p| >= 100), Y01 >= 1e-4 >> iszero threshold,
        // so all four cases hit T0 (the general path for each mass count).
        // TIN0: all masses zero, random p   → T0 massless
        // TIN1: one mass nonzero, random p  → T0 one-mass
        // TIN2: two masses nonzero, random p → T0 two-mass
        // TIN3: three masses nonzero, random p → T0 three-mass
        for (int n_masses(0); n_masses<4; n_masses++) {
            std::srand(12345);
            for (size_t i(0); i<batch_size; ++i) {
                m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = 0.;
                if (n_masses >= 1) m_h(i, 2) = m2;
                if (n_masses >= 2) m_h(i, 1) = m12;
                if (n_masses >= 3) m_h(i, 0) = m32;
                p_h(i, 0) = rs(low, up);
                p_h(i, 1) = rs(low, up);
                p_h(i, 2) = rs(low, up);
            }
            Kokkos::deep_copy(mu2_d, mu2_h);
            Kokkos::deep_copy(m_d, m_h);
            Kokkos::deep_copy(p_d, p_h);
            tt.start();
            Kokkos::parallel_for("Triangle TIN", policy, KOKKOS_LAMBDA(const int& i) {
                ql::TR<complex, double, double>(res_d, mu2_d, m_d, p_d, i);
            });
            Kokkos::fence();
            double elapsed = tt.stop();
            Kokkos::deep_copy(res_h, res_d);
            if (mode == 0) {
                std::cout << "TIN" << n_masses << "," << batch_size << "," << elapsed << std::endl;
            } else if (mode == 1) {
                for (size_t i = 0; i < batch_size; ++i) {
                    double m_arr[3] = {m_h(i, 0), m_h(i, 1), m_h(i, 2)};
                    double p_arr[3] = {p_h(i, 0), p_h(i, 1), p_h(i, 2)};
                    std::cout << "TIN" << n_masses << ","
                              << (i+1) << ","
                              << mu2_h(i) << ","
                              << arrayToCSV(m_arr, 3) << ","
                              << arrayToCSV(p_arr, 3) << ","
                              << complexToCSV(res_h(i, 0)) << ","
                              << complexToCSV(res_h(i, 1)) << ","
                              << complexToCSV(res_h(i, 2)) << std::endl;
                }
            }
        }

        // T1 - I(0, 0, p3^2; 0, 0, 0): massless, two zero momenta
        // After SnglSort the two zeros land at psq[0] and psq[1], so Y01=Y12=0 → T1
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = 0.;
            p_h(i, 0) = 0.; p_h(i, 1) = 0.;
            p_h(i, 2) = r(low, up);
        }
        Kokkos::deep_copy(mu2_d, mu2_h);
        Kokkos::deep_copy(m_d, m_h);
        Kokkos::deep_copy(p_d, p_h);
        tt.start();
        Kokkos::parallel_for("Triangle T1", policy, KOKKOS_LAMBDA(const int& i) {
            ql::TR<complex, double, double>(res_d, mu2_d, m_d, p_d, i);
        });
        Kokkos::fence();
        double elapsed = tt.stop();
        Kokkos::deep_copy(res_h, res_d);
        if (mode == 0) {
            std::cout << "T1," << batch_size << "," << elapsed << std::endl;
        } else if (mode == 1) {
            for (size_t i = 0; i < batch_size; ++i) {
                double m_arr[3] = {m_h(i, 0), m_h(i, 1), m_h(i, 2)};
                double p_arr[3] = {p_h(i, 0), p_h(i, 1), p_h(i, 2)};
                std::cout << "T1," << (i+1) << "," << mu2_h(i) << ","
                          << arrayToCSV(m_arr, 3) << ","
                          << arrayToCSV(p_arr, 3) << ","
                          << complexToCSV(res_h(i, 0)) << ","
                          << complexToCSV(res_h(i, 1)) << ","
                          << complexToCSV(res_h(i, 2)) << std::endl;
            }
        }

        // T2 - I(0, p1^2, p2^2; 0, 0, 0): massless, one zero momentum
        // After SnglSort: psq[0]=0, Y01=0 but Y12≠0 → T2
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = 0.;
            p_h(i, 0) = 0.;
            p_h(i, 1) = rs(low, up);
            p_h(i, 2) = rs(low, up);
        }
        Kokkos::deep_copy(mu2_d, mu2_h);
        Kokkos::deep_copy(m_d, m_h);
        Kokkos::deep_copy(p_d, p_h);
        tt.start();
        Kokkos::parallel_for("Triangle T2", policy, KOKKOS_LAMBDA(const int& i) {
            ql::TR<complex, double, double>(res_d, mu2_d, m_d, p_d, i);
        });
        Kokkos::fence();
        elapsed = tt.stop();
        Kokkos::deep_copy(res_h, res_d);
        if (mode == 0) {
            std::cout << "T2," << batch_size << "," << elapsed << std::endl;
        } else if (mode == 1) {
            for (size_t i = 0; i < batch_size; ++i) {
                double m_arr[3] = {m_h(i, 0), m_h(i, 1), m_h(i, 2)};
                double p_arr[3] = {p_h(i, 0), p_h(i, 1), p_h(i, 2)};
                std::cout << "T2," << (i+1) << "," << mu2_h(i) << ","
                          << arrayToCSV(m_arr, 3) << ","
                          << arrayToCSV(p_arr, 3) << ","
                          << complexToCSV(res_h(i, 0)) << ","
                          << complexToCSV(res_h(i, 1)) << ","
                          << complexToCSV(res_h(i, 2)) << std::endl;
            }
        }

        // T3 - I(0, p2^2, p3^2; 0, 0, m^2): one mass, p[0]=0, p[1] and p[2] != m^2
        // After TriSort: msq=[0,0,m^2], psq=[p1,p2,p3] unchanged (m^2 already at index 2).
        // Y01 = -p[0]/2 = 0, Y02 = (m^2-p[2])/2 ≠ 0, Y12 = (m^2-p[1])/2 ≠ 0 → T3
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = m2;
            p_h(i, 0) = 0.;
            p_h(i, 1) = rs(low, up);
            p_h(i, 2) = rs(low, up);
        }
        Kokkos::deep_copy(mu2_d, mu2_h);
        Kokkos::deep_copy(m_d, m_h);
        Kokkos::deep_copy(p_d, p_h);
        tt.start();
        Kokkos::parallel_for("Triangle T3", policy, KOKKOS_LAMBDA(const int& i) {
            ql::TR<complex, double, double>(res_d, mu2_d, m_d, p_d, i);
        });
        Kokkos::fence();
        elapsed = tt.stop();
        Kokkos::deep_copy(res_h, res_d);
        if (mode == 0) {
            std::cout << "T3," << batch_size << "," << elapsed << std::endl;
        } else if (mode == 1) {
            for (size_t i = 0; i < batch_size; ++i) {
                double m_arr[3] = {m_h(i, 0), m_h(i, 1), m_h(i, 2)};
                double p_arr[3] = {p_h(i, 0), p_h(i, 1), p_h(i, 2)};
                std::cout << "T3," << (i+1) << "," << mu2_h(i) << ","
                          << arrayToCSV(m_arr, 3) << ","
                          << arrayToCSV(p_arr, 3) << ","
                          << complexToCSV(res_h(i, 0)) << ","
                          << complexToCSV(res_h(i, 1)) << ","
                          << complexToCSV(res_h(i, 2)) << std::endl;
            }
        }

        // T4a - I(0, p2^2, m^2; 0, 0, m^2): one mass, iszero(Y02) branch
        // Y01=0 (p[0]=0), Y02=(m-p[2])/2=0 (p[2]=m), Y12=(m-p[1])/2 ≠ 0 (p[1] random)
        // Randomize m per element; set p[2]=m. Calls T4(res, musq, msq[2], psq[1], i).
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            double mi = r(low, up);
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = mi;
            p_h(i, 0) = 0.;
            p_h(i, 1) = rs(low, up);  // random; different PRNG position from mi makes exact equality astronomically unlikely
            p_h(i, 2) = mi;           // p[2] = m → Y02 = 0
        }
        Kokkos::deep_copy(mu2_d, mu2_h);
        Kokkos::deep_copy(m_d, m_h);
        Kokkos::deep_copy(p_d, p_h);
        tt.start();
        Kokkos::parallel_for("Triangle T4a", policy, KOKKOS_LAMBDA(const int& i) {
            ql::TR<complex, double, double>(res_d, mu2_d, m_d, p_d, i);
        });
        Kokkos::fence();
        elapsed = tt.stop();
        Kokkos::deep_copy(res_h, res_d);
        if (mode == 0) {
            std::cout << "T4a," << batch_size << "," << elapsed << std::endl;
        } else if (mode == 1) {
            for (size_t i = 0; i < batch_size; ++i) {
                double m_arr[3] = {m_h(i, 0), m_h(i, 1), m_h(i, 2)};
                double p_arr[3] = {p_h(i, 0), p_h(i, 1), p_h(i, 2)};
                std::cout << "T4a," << (i+1) << "," << mu2_h(i) << ","
                          << arrayToCSV(m_arr, 3) << ","
                          << arrayToCSV(p_arr, 3) << ","
                          << complexToCSV(res_h(i, 0)) << ","
                          << complexToCSV(res_h(i, 1)) << ","
                          << complexToCSV(res_h(i, 2)) << std::endl;
            }
        }

        // T4b - I(0, m^2, p3^2; 0, 0, m^2): one mass, iszero(Y12) branch
        // Y01=0 (p[0]=0), Y12=(m-p[1])/2=0 (p[1]=m), Y02=(m-p[2])/2 ≠ 0 (p[2] random)
        // Randomize m per element; set p[1]=m. Calls T4(res, musq, msq[2], psq[2], i).
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            double mi = r(low, up);
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = mi;
            p_h(i, 0) = 0.;
            p_h(i, 1) = mi;           // p[1] = m → Y12 = 0
            p_h(i, 2) = rs(low, up);  // random; different PRNG position from mi makes exact equality astronomically unlikely
        }
        Kokkos::deep_copy(mu2_d, mu2_h);
        Kokkos::deep_copy(m_d, m_h);
        Kokkos::deep_copy(p_d, p_h);
        tt.start();
        Kokkos::parallel_for("Triangle T4b", policy, KOKKOS_LAMBDA(const int& i) {
            ql::TR<complex, double, double>(res_d, mu2_d, m_d, p_d, i);
        });
        Kokkos::fence();
        elapsed = tt.stop();
        Kokkos::deep_copy(res_h, res_d);
        if (mode == 0) {
            std::cout << "T4b," << batch_size << "," << elapsed << std::endl;
        } else if (mode == 1) {
            for (size_t i = 0; i < batch_size; ++i) {
                double m_arr[3] = {m_h(i, 0), m_h(i, 1), m_h(i, 2)};
                double p_arr[3] = {p_h(i, 0), p_h(i, 1), p_h(i, 2)};
                std::cout << "T4b," << (i+1) << "," << mu2_h(i) << ","
                          << arrayToCSV(m_arr, 3) << ","
                          << arrayToCSV(p_arr, 3) << ","
                          << complexToCSV(res_h(i, 0)) << ","
                          << complexToCSV(res_h(i, 1)) << ","
                          << complexToCSV(res_h(i, 2)) << std::endl;
            }
        }

        // T5 - I(0, m^2, m^2; 0, 0, m^2): one mass, p[0]=0, p[1]=p[2]=m^2
        // Y01=0, Y02=0, Y12=0 → T5
        // Randomize m per element; set p[1]=p[2]=m to maintain Y02=Y12=0.
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            double mi = r(low, up);
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = mi;
            p_h(i, 0) = 0.;
            p_h(i, 1) = mi;  // p[1] = m → Y12 = 0
            p_h(i, 2) = mi;  // p[2] = m → Y02 = 0
        }
        Kokkos::deep_copy(mu2_d, mu2_h);
        Kokkos::deep_copy(m_d, m_h);
        Kokkos::deep_copy(p_d, p_h);
        tt.start();
        Kokkos::parallel_for("Triangle T5", policy, KOKKOS_LAMBDA(const int& i) {
            ql::TR<complex, double, double>(res_d, mu2_d, m_d, p_d, i);
        });
        Kokkos::fence();
        elapsed = tt.stop();
        Kokkos::deep_copy(res_h, res_d);
        if (mode == 0) {
            std::cout << "T5," << batch_size << "," << elapsed << std::endl;
        } else if (mode == 1) {
            for (size_t i = 0; i < batch_size; ++i) {
                double m_arr[3] = {m_h(i, 0), m_h(i, 1), m_h(i, 2)};
                double p_arr[3] = {p_h(i, 0), p_h(i, 1), p_h(i, 2)};
                std::cout << "T5," << (i+1) << "," << mu2_h(i) << ","
                          << arrayToCSV(m_arr, 3) << ","
                          << arrayToCSV(p_arr, 3) << ","
                          << complexToCSV(res_h(i, 0)) << ","
                          << complexToCSV(res_h(i, 1)) << ","
                          << complexToCSV(res_h(i, 2)) << std::endl;
            }
        }

        // T6 - I(m1^2, p2^2, m3^2; 0, m1^2, m3^2): two masses, p[0]=m1^2, p[2]=m3^2
        // After TriSort with m=[0,m1^2,m3^2] (m3^2 > m1^2): msq=[0,m1^2,m3^2], psq unchanged.
        // Y01=(0+m1^2-m1^2)/2=0, Y02=(0+m3^2-m3^2)/2=0 → T6
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = m12; m_h(i, 2) = m32;
            p_h(i, 0) = m12;
            p_h(i, 1) = rs(low, up);
            p_h(i, 2) = m32;
        }
        Kokkos::deep_copy(mu2_d, mu2_h);
        Kokkos::deep_copy(m_d, m_h);
        Kokkos::deep_copy(p_d, p_h);
        tt.start();
        Kokkos::parallel_for("Triangle T6", policy, KOKKOS_LAMBDA(const int& i) {
            ql::TR<complex, double, double>(res_d, mu2_d, m_d, p_d, i);
        });
        Kokkos::fence();
        elapsed = tt.stop();
        Kokkos::deep_copy(res_h, res_d);
        if (mode == 0) {
            std::cout << "T6," << batch_size << "," << elapsed << std::endl;
        } else if (mode == 1) {
            for (size_t i = 0; i < batch_size; ++i) {
                double m_arr[3] = {m_h(i, 0), m_h(i, 1), m_h(i, 2)};
                double p_arr[3] = {p_h(i, 0), p_h(i, 1), p_h(i, 2)};
                std::cout << "T6," << (i+1) << "," << mu2_h(i) << ","
                          << arrayToCSV(m_arr, 3) << ","
                          << arrayToCSV(p_arr, 3) << ","
                          << complexToCSV(res_h(i, 0)) << ","
                          << complexToCSV(res_h(i, 1)) << ","
                          << complexToCSV(res_h(i, 2)) << std::endl;
            }
        }
    }

    Kokkos::finalize();
    return 0;
}
