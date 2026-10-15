// xpmath backend selected by XPMATH_BACKEND_{dd,ff,qf,tf}.
// Types and ql:: overloads come from kokkosMaths_xp.h via boxGPU.h.

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
#include "quad_inputs.h"
#include <algorithm>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>
#include "qcdloop/timer.h"
#include "qcdloop/boxGPU.h"
#include "ql_launch.h"

using std::vector;
using std::cout;
using std::endl;
using std::string;
using real_t = ql::xp_real;
using complex_t = ql::xp_complex;
using complex = complex_t;

void printDoubleBits(double x)
{
    // We'll copy the double bits into a 64-bit integer.
    // A union is a common trick, or we can use memcpy.
    union {
        double d;
        uint64_t u;
    } conv;

    conv.d = x;

    // Use C99's PRIx64 for a portable 64-bit hex format.
    // %.16g prints up to 16 significant digits in decimal (just for reference).
    // std::printf("decimal=%.16g\n", x);
    std::printf("0x%016" PRIx64, conv.u);
}

std::string doubleToHex(double x)
{
    // We'll copy the double bits into a 64-bit integer.
    // A union is a common trick, or we can use memcpy.
    union {
        double d;
        uint64_t u;
    } conv;

    conv.d = x;

    // Use C99's PRIx64 for a portable 64-bit hex format.
    char hex_str[19]; // "0x" + 16 hex digits + null terminator
    std::sprintf(hex_str, "0x%016" PRIx64, conv.u);
    return std::string(hex_str);
}


// Helper function to format vector for CSV output
template<typename T>
std::string vectorToCSV(const std::vector<T>& vec) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) ss << ",";
        ss << vec[i];
    }
    ss << "]";
    return ss.str();
}

// One decimal per value, matching Box_true.txt. The accuracy parser stores
// Python floats, so limbs past double are rounded here the same way.
double to_double(real_t const& x) {
#if defined(XPMATH_BACKEND_dd) || defined(XPMATH_BACKEND_ff)
    return static_cast<double>(x.hi) + static_cast<double>(x.lo);
#elif defined(XPMATH_BACKEND_tf)
    return static_cast<double>(x.f0) + static_cast<double>(x.f1)
         + static_cast<double>(x.f2);
#elif defined(XPMATH_BACKEND_qf)
    return static_cast<double>(x.f0) + static_cast<double>(x.f1)
         + static_cast<double>(x.f2) + static_cast<double>(x.f3);
#else
#error "boxGPU_test_xp.cc requires one XPMATH_BACKEND_* macro"
#endif
}

std::string fmt_real(real_t const& x) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.16e", to_double(x));
    return std::string(buf);
}

std::string arrayToCSV(const real_t* arr, size_t size) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < size; ++i) {
        if (i > 0) ss << ",";
        ss << fmt_real(arr[i]);
    }
    ss << "]";
    return ss.str();
}

std::string complexToCSV(const complex_t& c) {
    std::stringstream ss;
    ss << "(" << fmt_real(c.real()) << "," << fmt_real(c.imag()) << ")";
    return ss.str();
}

#if defined(XPMATH_BACKEND_dd)
// Limbs come from quad_inputs.cc, compiled with host g++. The quad
// evaluation is the same one the ddfun driver uses. A DoubleDouble built
// from a double alone sets the low limb to zero.
static inline real_t from_limbs(DdLimbs x) {
    return real_t(x.hi, x.lo);
}

real_t r(double min, double max) {
    return from_limbs(dd_rand(min, max));
}

real_t rs(double min, double max) {
    return from_limbs(dd_rands(min, max));
}
#else
// Same quad draw as DoubleDouble, then the successive float split each
// backend's double constructor uses. A double draw drops bits these
// types can still hold.
static inline real_t from_words(QuadWords x) {
#if defined(XPMATH_BACKEND_ff)
    return real_t(x.w0, x.w1);
#elif defined(XPMATH_BACKEND_tf)
    return real_t(x.w0, x.w1, x.w2);
#elif defined(XPMATH_BACKEND_qf)
    return real_t(x.w0, x.w1, x.w2, x.w3);
#else
#error "xpmath backend not selected"
#endif
}

real_t r(double min, double max) {
    return from_words(qw_rand(min, max));
}

real_t rs(double min, double max) {
    return from_words(qw_rands(min, max));
}
#endif


template <class F>
double launch_box(int n, F const& f) {
    ql::Timer total;
    total.start();
    if (n <= 0) return total.stop();
    ql::launch_range(0, n, f);
    Kokkos::fence();
    return total.stop();
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
        
        // Optional integral name, e.g. BIN4. Empty runs the full set.
        // BIN4 reseeds on its own, so selecting it does not change its rows.
        string only;
        if (argc > 3)
            only = argv[3];

        std::ios::sync_with_stdio(false);
        std::cout << "Running with mode = " << mode << std::endl;
        std::cout << "Running with batch_size = " << batch_size << std::endl;

        if (mode == 0) {
            // Print CSV header for performance benchmark mode
            std::cout << "Target Integral,Batch size,Time" << std::endl;
        } else if (mode == 1) {
            // Print CSV header for accuracy test mode
            std::cout << "Target Integral,Test ID,mu2,ms,ps,Coeff 1,Coeff 2,Coeff 3" << std::endl;
        }
        
        // Call the integral. 100 and 1000000 are exact in double. The
        // DoubleDouble path promotes them to quad inside dd_rand.
        double low = 100;
        double up  = 1000000;
		
        // Create Kokkos Views for batch processing
        Kokkos::View<real_t*> mu2_d("mu2", batch_size);
        Kokkos::View<real_t* [4]> m_d("m", batch_size);
        Kokkos::View<real_t* [6]> p_d("p", batch_size);
        Kokkos::View<complex_t* [3]> res_d("res", batch_size);
        
        auto mu2_h = Kokkos::create_mirror_view(mu2_d);
        auto m_h = Kokkos::create_mirror_view(m_d);
        auto p_h = Kokkos::create_mirror_view(p_d);
        auto res_h = Kokkos::create_mirror_view(res_d);
        
        // Initialize mu2
        for (size_t i = 0; i < batch_size; ++i) {
#if defined(XPMATH_BACKEND_dd)
            mu2_h(i) = from_limbs(dd_mu2());
#else
            mu2_h(i) = from_words(qw_mu2());
#endif
        }
        
        // Trigger BIN0 - BIN4
        for (int n_masses(0); n_masses<5; n_masses++) {
            if (!only.empty() && only != string("BIN") + std::to_string(n_masses))
                continue;
            // Fill host mirrors
            std::srand(12345);
            for (size_t i(0); i<batch_size; ++i) {
                // should probably make this select randomly from {10., 50., 100., 200};
                for(int j(0); j<4; ++j) {
                    m_h(i, j) = 0.;
                }
                for(int j(0); j<n_masses; ++j) {
                    m_h(i, j) = real_t(10.0);
                }
                p_h(i, 0) = rs(low,up);
                p_h(i, 1) = rs(low,up);
                p_h(i, 2) = rs(low,up);
                p_h(i, 3) = rs(low,up);
                p_h(i, 4) = r(low,up);
                p_h(i, 5) = r(low,up);
            }
            
            // Copy to device
            Kokkos::deep_copy(mu2_d, mu2_h);
            Kokkos::deep_copy(m_d, m_h);
            Kokkos::deep_copy(p_d, p_h);
            
            // Launch parallel_for with timing
            double elapsed = launch_box(batch_size, KOKKOS_LAMBDA(const int& i) {
                ql::BO<complex_t, real_t, real_t>(res_d, mu2_d, m_d, p_d, i);
            });
            
            // Copy results back
            Kokkos::deep_copy(res_h, res_d);
            
            // Process results
            if (mode == 0) {
                std::cout << "BIN" << n_masses << "," << batch_size << "," << elapsed << std::endl;
            } else if (mode == 1) {
                for (size_t i = 0; i < batch_size; ++i) {
                    real_t m_arr[4] = {m_h(i, 0), m_h(i, 1), m_h(i, 2), m_h(i, 3)};
                    real_t p_arr[6] = {p_h(i, 0), p_h(i, 1), p_h(i, 2), p_h(i, 3), p_h(i, 4), p_h(i, 5)};
                    std::cout << "BIN" << n_masses << "," 
                              << (i+1) << "," 
                              << fmt_real(mu2_h(i)) << "," 
                              << arrayToCSV(m_arr, 4) << "," 
                              << arrayToCSV(p_arr, 6) << "," 
                              << complexToCSV(res_h(i, 0)) << "," 
                              << complexToCSV(res_h(i, 1)) << "," 
                              << complexToCSV(res_h(i, 2)) << std::endl;
                }
            }
        }
	
        // A named integral stops here. The views must be destroyed before
        // Kokkos::finalize, so this returns through the end of main.
        if (only.empty()) {

        // Zero mass integrals - B1
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = 0.; m_h(i, 3) = 0.;
            p_h(i, 0) = 0.; p_h(i, 1) = 0.; p_h(i, 2) = 0.; p_h(i, 3) = 0.;
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        Kokkos::deep_copy(mu2_d, mu2_h);
        Kokkos::deep_copy(m_d, m_h);
        Kokkos::deep_copy(p_d, p_h);
        double elapsed = launch_box(batch_size, KOKKOS_LAMBDA(const int& i) {
            ql::BO<complex_t, real_t, real_t>(res_d, mu2_d, m_d, p_d, i);
        });
        Kokkos::deep_copy(res_h, res_d);
        if (mode == 0) {
            std::cout << "B1," << batch_size << "," << elapsed << std::endl;
        } else if (mode == 1) {
            for (size_t i = 0; i < batch_size; ++i) {
                real_t m_arr[4] = {m_h(i, 0), m_h(i, 1), m_h(i, 2), m_h(i, 3)};
                real_t p_arr[6] = {p_h(i, 0), p_h(i, 1), p_h(i, 2), p_h(i, 3), p_h(i, 4), p_h(i, 5)};
                std::cout << "B1," << (i+1) << "," << fmt_real(mu2_h(i)) << "," 
                          << arrayToCSV(m_arr, 4) << "," << arrayToCSV(p_arr, 6) << "," 
                          << complexToCSV(res_h(i, 0)) << "," << complexToCSV(res_h(i, 1)) << "," 
                          << complexToCSV(res_h(i, 2)) << std::endl;
            }
        }

        // B2
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = 0.; m_h(i, 3) = 0.;
            p_h(i, 0) = 0.; p_h(i, 1) = 0.; p_h(i, 2) = 0.;
            p_h(i, 3) = rs(low,up); p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        Kokkos::deep_copy(mu2_d, mu2_h);
        Kokkos::deep_copy(m_d, m_h);
        Kokkos::deep_copy(p_d, p_h);
        elapsed = launch_box(batch_size, KOKKOS_LAMBDA(const int& i) {
            ql::BO<complex_t, real_t, real_t>(res_d, mu2_d, m_d, p_d, i);
        });
        Kokkos::deep_copy(res_h, res_d);
        if (mode == 0) {
            std::cout << "B2," << batch_size << "," << elapsed << std::endl;
        } else if (mode == 1) {
            for (size_t i = 0; i < batch_size; ++i) {
                real_t m_arr[4] = {m_h(i, 0), m_h(i, 1), m_h(i, 2), m_h(i, 3)};
                real_t p_arr[6] = {p_h(i, 0), p_h(i, 1), p_h(i, 2), p_h(i, 3), p_h(i, 4), p_h(i, 5)};
                std::cout << "B2," << (i+1) << "," << fmt_real(mu2_h(i)) << "," 
                          << arrayToCSV(m_arr, 4) << "," << arrayToCSV(p_arr, 6) << "," 
                          << complexToCSV(res_h(i, 0)) << "," << complexToCSV(res_h(i, 1)) << "," 
                          << complexToCSV(res_h(i, 2)) << std::endl;
            }
        }

        // B3
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = 0.; m_h(i, 3) = 0.;
            p_h(i, 0) = 0.; p_h(i, 1) = rs(low,up); p_h(i, 2) = 0.;
            p_h(i, 3) = rs(low,up); p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        Kokkos::deep_copy(mu2_d, mu2_h);
        Kokkos::deep_copy(m_d, m_h);
        Kokkos::deep_copy(p_d, p_h);
        elapsed = launch_box(batch_size, KOKKOS_LAMBDA(const int& i) {
            ql::BO<complex_t, real_t, real_t>(res_d, mu2_d, m_d, p_d, i);
        });
        Kokkos::deep_copy(res_h, res_d);
        if (mode == 0) {
            std::cout << "B3," << batch_size << "," << elapsed << std::endl;
        } else if (mode == 1) {
            for (size_t i = 0; i < batch_size; ++i) {
                real_t m_arr[4] = {m_h(i, 0), m_h(i, 1), m_h(i, 2), m_h(i, 3)};
                real_t p_arr[6] = {p_h(i, 0), p_h(i, 1), p_h(i, 2), p_h(i, 3), p_h(i, 4), p_h(i, 5)};
                std::cout << "B3," << (i+1) << "," << fmt_real(mu2_h(i)) << "," 
                          << arrayToCSV(m_arr, 4) << "," << arrayToCSV(p_arr, 6) << "," 
                          << complexToCSV(res_h(i, 0)) << "," << complexToCSV(res_h(i, 1)) << "," 
                          << complexToCSV(res_h(i, 2)) << std::endl;
            }
        }

        // B4
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = 0.; m_h(i, 3) = 0.;
            p_h(i, 0) = 0.; p_h(i, 1) = 0.;
            p_h(i, 2) = rs(low,up); p_h(i, 3) = rs(low,up);
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        Kokkos::deep_copy(mu2_d, mu2_h);
        Kokkos::deep_copy(m_d, m_h);
        Kokkos::deep_copy(p_d, p_h);
        elapsed = launch_box(batch_size, KOKKOS_LAMBDA(const int& i) {
            ql::BO<complex_t, real_t, real_t>(res_d, mu2_d, m_d, p_d, i);
        });
        Kokkos::deep_copy(res_h, res_d);
        if (mode == 0) {
            std::cout << "B4," << batch_size << "," << elapsed << std::endl;
        } else if (mode == 1) {
            for (size_t i = 0; i < batch_size; ++i) {
                real_t m_arr[4] = {m_h(i, 0), m_h(i, 1), m_h(i, 2), m_h(i, 3)};
                real_t p_arr[6] = {p_h(i, 0), p_h(i, 1), p_h(i, 2), p_h(i, 3), p_h(i, 4), p_h(i, 5)};
                std::cout << "B4," << (i+1) << "," << fmt_real(mu2_h(i)) << "," 
                          << arrayToCSV(m_arr, 4) << "," << arrayToCSV(p_arr, 6) << "," 
                          << complexToCSV(res_h(i, 0)) << "," << complexToCSV(res_h(i, 1)) << "," 
                          << complexToCSV(res_h(i, 2)) << std::endl;
            }
        }

        // B5
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = 0.; m_h(i, 3) = 0.;
            p_h(i, 0) = 0.;
            p_h(i, 1) = rs(low,up); p_h(i, 2) = rs(low,up); p_h(i, 3) = rs(low,up);
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        Kokkos::deep_copy(mu2_d, mu2_h);
        Kokkos::deep_copy(m_d, m_h);
        Kokkos::deep_copy(p_d, p_h);
        elapsed = launch_box(batch_size, KOKKOS_LAMBDA(const int& i) {
            ql::BO<complex_t, real_t, real_t>(res_d, mu2_d, m_d, p_d, i);
        });
        Kokkos::deep_copy(res_h, res_d);
        if (mode == 0) {
            std::cout << "B5," << batch_size << "," << elapsed << std::endl;
        } else if (mode == 1) {
            for (size_t i = 0; i < batch_size; ++i) {
                real_t m_arr[4] = {m_h(i, 0), m_h(i, 1), m_h(i, 2), m_h(i, 3)};
                real_t p_arr[6] = {p_h(i, 0), p_h(i, 1), p_h(i, 2), p_h(i, 3), p_h(i, 4), p_h(i, 5)};
                std::cout << "B5," << (i+1) << "," << fmt_real(mu2_h(i)) << "," 
                          << arrayToCSV(m_arr, 4) << "," << arrayToCSV(p_arr, 6) << "," 
                          << complexToCSV(res_h(i, 0)) << "," << complexToCSV(res_h(i, 1)) << "," 
                          << complexToCSV(res_h(i, 2)) << std::endl;
            }
        }

        // single mass integrals
#if defined(XPMATH_BACKEND_dd)
        real_t m2 = from_limbs(dd_mass_10());
#else
        real_t m2 = from_words(qw_mass_10());
#endif
        
        // B6
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = 0.; m_h(i, 3) = m2;
            p_h(i, 0) = 0.; p_h(i, 1) = 0.;
            p_h(i, 2) = m2; p_h(i, 3) = m2;
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        Kokkos::deep_copy(mu2_d, mu2_h);
        Kokkos::deep_copy(m_d, m_h);
        Kokkos::deep_copy(p_d, p_h);
        elapsed = launch_box(batch_size, KOKKOS_LAMBDA(const int& i) {
            ql::BO<complex_t, real_t, real_t>(res_d, mu2_d, m_d, p_d, i);
        });
        Kokkos::deep_copy(res_h, res_d);
        if (mode == 0) {
            std::cout << "B6," << batch_size << "," << elapsed << std::endl;
        } else if (mode == 1) {
            for (size_t i = 0; i < batch_size; ++i) {
                real_t m_arr[4] = {m_h(i, 0), m_h(i, 1), m_h(i, 2), m_h(i, 3)};
                real_t p_arr[6] = {p_h(i, 0), p_h(i, 1), p_h(i, 2), p_h(i, 3), p_h(i, 4), p_h(i, 5)};
                std::cout << "B6," << (i+1) << "," << fmt_real(mu2_h(i)) << "," 
                          << arrayToCSV(m_arr, 4) << "," << arrayToCSV(p_arr, 6) << "," 
                          << complexToCSV(res_h(i, 0)) << "," << complexToCSV(res_h(i, 1)) << "," 
                          << complexToCSV(res_h(i, 2)) << std::endl;
            }
        }

        // B7
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = 0.; m_h(i, 3) = m2;
            p_h(i, 0) = 0.; p_h(i, 1) = 0.;
            p_h(i, 2) = m2; p_h(i, 3) = rs(low,up);
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        Kokkos::deep_copy(mu2_d, mu2_h);
        Kokkos::deep_copy(m_d, m_h);
        Kokkos::deep_copy(p_d, p_h);
        elapsed = launch_box(batch_size, KOKKOS_LAMBDA(const int& i) {
            ql::BO<complex_t, real_t, real_t>(res_d, mu2_d, m_d, p_d, i);
        });
        Kokkos::deep_copy(res_h, res_d);
        if (mode == 0) {
            std::cout << "B7," << batch_size << "," << elapsed << std::endl;
        } else if (mode == 1) {
            for (size_t i = 0; i < batch_size; ++i) {
                real_t m_arr[4] = {m_h(i, 0), m_h(i, 1), m_h(i, 2), m_h(i, 3)};
                real_t p_arr[6] = {p_h(i, 0), p_h(i, 1), p_h(i, 2), p_h(i, 3), p_h(i, 4), p_h(i, 5)};
                std::cout << "B7," << (i+1) << "," << fmt_real(mu2_h(i)) << "," 
                          << arrayToCSV(m_arr, 4) << "," << arrayToCSV(p_arr, 6) << "," 
                          << complexToCSV(res_h(i, 0)) << "," << complexToCSV(res_h(i, 1)) << "," 
                          << complexToCSV(res_h(i, 2)) << std::endl;
            }
        }

        // B8
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = 0.; m_h(i, 3) = m2;
            p_h(i, 0) = 0.; p_h(i, 1) = 0.;
            p_h(i, 2) = rs(low,up); p_h(i, 3) = rs(low,up);
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        Kokkos::deep_copy(mu2_d, mu2_h);
        Kokkos::deep_copy(m_d, m_h);
        Kokkos::deep_copy(p_d, p_h);
        elapsed = launch_box(batch_size, KOKKOS_LAMBDA(const int& i) {
            ql::BO<complex_t, real_t, real_t>(res_d, mu2_d, m_d, p_d, i);
        });
        Kokkos::deep_copy(res_h, res_d);
        if (mode == 0) {
            std::cout << "B8," << batch_size << "," << elapsed << std::endl;
        } else if (mode == 1) {
            for (size_t i = 0; i < batch_size; ++i) {
                real_t m_arr[4] = {m_h(i, 0), m_h(i, 1), m_h(i, 2), m_h(i, 3)};
                real_t p_arr[6] = {p_h(i, 0), p_h(i, 1), p_h(i, 2), p_h(i, 3), p_h(i, 4), p_h(i, 5)};
                std::cout << "B8," << (i+1) << "," << fmt_real(mu2_h(i)) << "," 
                          << arrayToCSV(m_arr, 4) << "," << arrayToCSV(p_arr, 6) << "," 
                          << complexToCSV(res_h(i, 0)) << "," << complexToCSV(res_h(i, 1)) << "," 
                          << complexToCSV(res_h(i, 2)) << std::endl;
            }
        }

        // B9
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = 0.; m_h(i, 3) = m2;
            p_h(i, 0) = 0.;
            p_h(i, 1) = rs(low,up); p_h(i, 2) = rs(low,up); p_h(i, 3) = m2;
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        Kokkos::deep_copy(mu2_d, mu2_h);
        Kokkos::deep_copy(m_d, m_h);
        Kokkos::deep_copy(p_d, p_h);
        elapsed = launch_box(batch_size, KOKKOS_LAMBDA(const int& i) {
            ql::BO<complex_t, real_t, real_t>(res_d, mu2_d, m_d, p_d, i);
        });
        Kokkos::deep_copy(res_h, res_d);
        if (mode == 0) {
            std::cout << "B9," << batch_size << "," << elapsed << std::endl;
        } else if (mode == 1) {
            for (size_t i = 0; i < batch_size; ++i) {
                real_t m_arr[4] = {m_h(i, 0), m_h(i, 1), m_h(i, 2), m_h(i, 3)};
                real_t p_arr[6] = {p_h(i, 0), p_h(i, 1), p_h(i, 2), p_h(i, 3), p_h(i, 4), p_h(i, 5)};
                std::cout << "B9," << (i+1) << "," << fmt_real(mu2_h(i)) << "," 
                          << arrayToCSV(m_arr, 4) << "," << arrayToCSV(p_arr, 6) << "," 
                          << complexToCSV(res_h(i, 0)) << "," << complexToCSV(res_h(i, 1)) << "," 
                          << complexToCSV(res_h(i, 2)) << std::endl;
            }
        }

        // B10
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = 0.; m_h(i, 3) = m2;
            p_h(i, 0) = 0.;
            p_h(i, 1) = rs(low,up); p_h(i, 2) = rs(low,up); p_h(i, 3) = rs(low,up);
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        Kokkos::deep_copy(mu2_d, mu2_h);
        Kokkos::deep_copy(m_d, m_h);
        Kokkos::deep_copy(p_d, p_h);
        elapsed = launch_box(batch_size, KOKKOS_LAMBDA(const int& i) {
            ql::BO<complex_t, real_t, real_t>(res_d, mu2_d, m_d, p_d, i);
        });
        Kokkos::deep_copy(res_h, res_d);
        if (mode == 0) {
            std::cout << "B10," << batch_size << "," << elapsed << std::endl;
        } else if (mode == 1) {
            for (size_t i = 0; i < batch_size; ++i) {
                real_t m_arr[4] = {m_h(i, 0), m_h(i, 1), m_h(i, 2), m_h(i, 3)};
                real_t p_arr[6] = {p_h(i, 0), p_h(i, 1), p_h(i, 2), p_h(i, 3), p_h(i, 4), p_h(i, 5)};
                std::cout << "B10," << (i+1) << "," << fmt_real(mu2_h(i)) << "," 
                          << arrayToCSV(m_arr, 4) << "," << arrayToCSV(p_arr, 6) << "," 
                          << complexToCSV(res_h(i, 0)) << "," << complexToCSV(res_h(i, 1)) << "," 
                          << complexToCSV(res_h(i, 2)) << std::endl;
            }
        }

        // two mass integrals
#if defined(XPMATH_BACKEND_dd)
        real_t m22 = from_limbs(dd_mass_4p9_sq());
        real_t m32 = from_limbs(dd_mass_10());
        real_t m42 = from_limbs(dd_mass_50_sq());
#else
        real_t m22 = from_words(qw_mass_4p9_sq());
        real_t m32 = from_words(qw_mass_10());
        real_t m42 = from_words(qw_mass_50_sq());
#endif
        
        // B11
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = m32; m_h(i, 3) = m42;
            p_h(i, 0) = 0.; p_h(i, 1) = m32;
            p_h(i, 2) = rs(low,up); p_h(i, 3) = m42;
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        Kokkos::deep_copy(mu2_d, mu2_h);
        Kokkos::deep_copy(m_d, m_h);
        Kokkos::deep_copy(p_d, p_h);
        elapsed = launch_box(batch_size, KOKKOS_LAMBDA(const int& i) {
            ql::BO<complex_t, real_t, real_t>(res_d, mu2_d, m_d, p_d, i);
        });
        Kokkos::deep_copy(res_h, res_d);
        if (mode == 0) {
            std::cout << "B11," << batch_size << "," << elapsed << std::endl;
        } else if (mode == 1) {
            for (size_t i = 0; i < batch_size; ++i) {
                real_t m_arr[4] = {m_h(i, 0), m_h(i, 1), m_h(i, 2), m_h(i, 3)};
                real_t p_arr[6] = {p_h(i, 0), p_h(i, 1), p_h(i, 2), p_h(i, 3), p_h(i, 4), p_h(i, 5)};
                std::cout << "B11," << (i+1) << "," << fmt_real(mu2_h(i)) << "," 
                          << arrayToCSV(m_arr, 4) << "," << arrayToCSV(p_arr, 6) << "," 
                          << complexToCSV(res_h(i, 0)) << "," << complexToCSV(res_h(i, 1)) << "," 
                          << complexToCSV(res_h(i, 2)) << std::endl;
            }
        }
	
        // B12
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = m32; m_h(i, 3) = m42;
            p_h(i, 0) = 0.;
            p_h(i, 1) = rs(low,up); p_h(i, 2) = rs(low,up); p_h(i, 3) = m42;
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        Kokkos::deep_copy(mu2_d, mu2_h);
        Kokkos::deep_copy(m_d, m_h);
        Kokkos::deep_copy(p_d, p_h);
        elapsed = launch_box(batch_size, KOKKOS_LAMBDA(const int& i) {
            ql::BO<complex_t, real_t, real_t>(res_d, mu2_d, m_d, p_d, i);
        });
        Kokkos::deep_copy(res_h, res_d);
        if (mode == 0) {
            std::cout << "B12," << batch_size << "," << elapsed << std::endl;
        } else if (mode == 1) {
            for (size_t i = 0; i < batch_size; ++i) {
                real_t m_arr[4] = {m_h(i, 0), m_h(i, 1), m_h(i, 2), m_h(i, 3)};
                real_t p_arr[6] = {p_h(i, 0), p_h(i, 1), p_h(i, 2), p_h(i, 3), p_h(i, 4), p_h(i, 5)};
                std::cout << "B12," << (i+1) << "," << fmt_real(mu2_h(i)) << "," 
                          << arrayToCSV(m_arr, 4) << "," << arrayToCSV(p_arr, 6) << "," 
                          << complexToCSV(res_h(i, 0)) << "," << complexToCSV(res_h(i, 1)) << "," 
                          << complexToCSV(res_h(i, 2)) << std::endl;
            }
        }
	
        // B13
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = m32; m_h(i, 3) = m42;
            p_h(i, 0) = 0.;
            p_h(i, 1) = rs(low,up); p_h(i, 2) = rs(low,up); p_h(i, 3) = rs(low,up);
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        Kokkos::deep_copy(mu2_d, mu2_h);
        Kokkos::deep_copy(m_d, m_h);
        Kokkos::deep_copy(p_d, p_h);
        elapsed = launch_box(batch_size, KOKKOS_LAMBDA(const int& i) {
            ql::BO<complex_t, real_t, real_t>(res_d, mu2_d, m_d, p_d, i);
        });
        Kokkos::deep_copy(res_h, res_d);
        if (mode == 0) {
            std::cout << "B13," << batch_size << "," << elapsed << std::endl;
        } else if (mode == 1) {
            for (size_t i = 0; i < batch_size; ++i) {
                real_t m_arr[4] = {m_h(i, 0), m_h(i, 1), m_h(i, 2), m_h(i, 3)};
                real_t p_arr[6] = {p_h(i, 0), p_h(i, 1), p_h(i, 2), p_h(i, 3), p_h(i, 4), p_h(i, 5)};
                std::cout << "B13," << (i+1) << "," << fmt_real(mu2_h(i)) << "," 
                          << arrayToCSV(m_arr, 4) << "," << arrayToCSV(p_arr, 6) << "," 
                          << complexToCSV(res_h(i, 0)) << "," << complexToCSV(res_h(i, 1)) << "," 
                          << complexToCSV(res_h(i, 2)) << std::endl;
            }
        }

        // B14
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = m22; m_h(i, 2) = 0.; m_h(i, 3) = m42;
            p_h(i, 0) = m22; p_h(i, 1) = m22;
            p_h(i, 2) = m42; p_h(i, 3) = m42;
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        Kokkos::deep_copy(mu2_d, mu2_h);
        Kokkos::deep_copy(m_d, m_h);
        Kokkos::deep_copy(p_d, p_h);
        elapsed = launch_box(batch_size, KOKKOS_LAMBDA(const int& i) {
            ql::BO<complex_t, real_t, real_t>(res_d, mu2_d, m_d, p_d, i);
        });
        Kokkos::deep_copy(res_h, res_d);
        if (mode == 0) {
            std::cout << "B14," << batch_size << "," << elapsed << std::endl;
        } else if (mode == 1) {
            for (size_t i = 0; i < batch_size; ++i) {
                real_t m_arr[4] = {m_h(i, 0), m_h(i, 1), m_h(i, 2), m_h(i, 3)};
                real_t p_arr[6] = {p_h(i, 0), p_h(i, 1), p_h(i, 2), p_h(i, 3), p_h(i, 4), p_h(i, 5)};
                std::cout << "B14," << (i+1) << "," << fmt_real(mu2_h(i)) << "," 
                          << arrayToCSV(m_arr, 4) << "," << arrayToCSV(p_arr, 6) << "," 
                          << complexToCSV(res_h(i, 0)) << "," << complexToCSV(res_h(i, 1)) << "," 
                          << complexToCSV(res_h(i, 2)) << std::endl;
            }
        }

        // B15
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = m22; m_h(i, 2) = 0.; m_h(i, 3) = m42;
            p_h(i, 0) = m22;
            p_h(i, 1) = rs(low,up); p_h(i, 2) = rs(low,up); p_h(i, 3) = m42;
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        Kokkos::deep_copy(mu2_d, mu2_h);
        Kokkos::deep_copy(m_d, m_h);
        Kokkos::deep_copy(p_d, p_h);
        elapsed = launch_box(batch_size, KOKKOS_LAMBDA(const int& i) {
            ql::BO<complex_t, real_t, real_t>(res_d, mu2_d, m_d, p_d, i);
        });
        Kokkos::deep_copy(res_h, res_d);
        if (mode == 0) {
            std::cout << "B15," << batch_size << "," << elapsed << std::endl;
        } else if (mode == 1) {
            for (size_t i = 0; i < batch_size; ++i) {
                real_t m_arr[4] = {m_h(i, 0), m_h(i, 1), m_h(i, 2), m_h(i, 3)};
                real_t p_arr[6] = {p_h(i, 0), p_h(i, 1), p_h(i, 2), p_h(i, 3), p_h(i, 4), p_h(i, 5)};
                std::cout << "B15," << (i+1) << "," << fmt_real(mu2_h(i)) << "," 
                          << arrayToCSV(m_arr, 4) << "," << arrayToCSV(p_arr, 6) << "," 
                          << complexToCSV(res_h(i, 0)) << "," << complexToCSV(res_h(i, 1)) << "," 
                          << complexToCSV(res_h(i, 2)) << std::endl;
            }
        }

        // three mass integrals - B16
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = m22; m_h(i, 2) = m32; m_h(i, 3) = m42;
            p_h(i, 0) = m22;
            p_h(i, 1) = rs(low,up); p_h(i, 2) = rs(low,up); p_h(i, 3) = m42;
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        Kokkos::deep_copy(mu2_d, mu2_h);
        Kokkos::deep_copy(m_d, m_h);
        Kokkos::deep_copy(p_d, p_h);
        elapsed = launch_box(batch_size, KOKKOS_LAMBDA(const int& i) {
            ql::BO<complex_t, real_t, real_t>(res_d, mu2_d, m_d, p_d, i);
        });
        Kokkos::deep_copy(res_h, res_d);
        if (mode == 0) {
            std::cout << "B16," << batch_size << "," << elapsed << std::endl;
        } else if (mode == 1) {
            for (size_t i = 0; i < batch_size; ++i) {
                real_t m_arr[4] = {m_h(i, 0), m_h(i, 1), m_h(i, 2), m_h(i, 3)};
                real_t p_arr[6] = {p_h(i, 0), p_h(i, 1), p_h(i, 2), p_h(i, 3), p_h(i, 4), p_h(i, 5)};
                std::cout << "B16," << (i+1) << "," << fmt_real(mu2_h(i)) << "," 
                          << arrayToCSV(m_arr, 4) << "," << arrayToCSV(p_arr, 6) << "," 
                          << complexToCSV(res_h(i, 0)) << "," << complexToCSV(res_h(i, 1)) << "," 
                          << complexToCSV(res_h(i, 2)) << std::endl;
            }
        }

        } // only.empty()
        
        
    }
    Kokkos::finalize();
    return 0;
}