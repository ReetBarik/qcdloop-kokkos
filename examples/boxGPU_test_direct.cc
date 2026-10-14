//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//          Stefan Hoeche   : shoeche@fnal.gov
//          Max Knobbe      : mknobbe@fnal.gov
//
// Direct-leaf-call variant of boxGPU_test.cc: each named section already
// knows statically which single integral it exercises (per
// ~/xpmath-kokkos/repro/REPORT.md section 2.3), so instead of going through
// ql::BO<...>()'s unconditional 5-way-then-5-way-again dispatch, each
// section calls its leaf (box/leaves/*.h) directly, doing only the
// classification/permutation prep that specific leaf needs. This file does
// NOT include qcdloop/boxGPU.h at all -- only the specific leaf headers
// each section actually calls, plus the classifier/Ycalc helpers -- so
// each kernel's compiled code is bounded by its own leaf, not the whole
// dispatch tree. See examples/boxGPU_test.cc for the BO()-routed oracle
// this is validated against (bit-for-bit, same RNG seeding).

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>
#include "qcdloop/timer.h"
#include "qcdloop/box/classify.h"
#include "qcdloop/box/box_common.h"
#include "quad_inputs.h"

using std::vector;
using std::cout;
using std::endl;
using std::string;
using complex = Kokkos::complex<double>;

void printDoubleBits(double x)
{
    union {
        double d;
        uint64_t u;
    } conv;

    conv.d = x;
    std::printf("0x%016" PRIx64, conv.u);
}

std::string doubleToHex(double x)
{
    union {
        double d;
        uint64_t u;
    } conv;

    conv.d = x;
    char hex_str[19];
    std::sprintf(hex_str, "0x%016" PRIx64, conv.u);
    return std::string(hex_str);
}

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

//
// ---- Shared, leaf-agnostic prep helpers (mirror what every box/dispatch/*.h
//      pruned BO() already duplicates 5x -- consolidating once here, same as
//      there, does not reference any leaf and so does not affect which
//      leaf's code ends up reachable from a given kernel). ----
//

template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
void prep_xpi(
    const Kokkos::View<TScale*>& mu2, const Kokkos::View<TMass* [4]>& m, const Kokkos::View<TScale* [6]>& p,
    const int i, TScale& scalefac, Kokkos::Array<TMass, 13>& xpi, TScale& musq) {

    scalefac = ql::Max(
        ql::kAbs(p(i, 4)),
        ql::Max(ql::kAbs(p(i, 5)),
        ql::Max(ql::kAbs(p(i, 0)),
        ql::Max(ql::kAbs(p(i, 1)),
        ql::Max(ql::kAbs(p(i, 2)),
        ql::kAbs(p(i, 3)))))));

    xpi[0] = m(i, 0) / scalefac;
    xpi[1] = m(i, 1) / scalefac;
    xpi[2] = m(i, 2) / scalefac;
    xpi[3] = m(i, 3) / scalefac;
    xpi[4] = TMass(p(i, 0) / scalefac);
    xpi[5] = TMass(p(i, 1) / scalefac);
    xpi[6] = TMass(p(i, 2) / scalefac);
    xpi[7] = TMass(p(i, 3) / scalefac);
    xpi[8] = TMass(p(i, 4) / scalefac);
    xpi[9] = TMass(p(i, 5) / scalefac);
    xpi[10] = xpi[4] + xpi[5] + xpi[6] + xpi[7] - xpi[8] - xpi[9];
    xpi[11] = -xpi[4] + xpi[5] - xpi[6] + xpi[7] + xpi[8] + xpi[9];
    xpi[12] = xpi[4] - xpi[5] + xpi[6] - xpi[7] + xpi[8] + xpi[9];

    musq = mu2(i) / scalefac;
}

template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
bool early_out(const Kokkos::Array<TMass, 13>& xpi, const Kokkos::View<TOutput* [3]>& res, const int i) {
    const TMass y13 = xpi[0] + xpi[2] - xpi[8];
    const TMass y24 = xpi[1] + xpi[3] - xpi[9];
    if (ql::iszero<TOutput, TMass, TScale>(y13) || ql::iszero<TOutput, TMass, TScale>(y24)) {
        res(i, 0) = ql::Constants<TOutput>::_zero();
        res(i, 1) = ql::Constants<TOutput>::_zero();
        res(i, 2) = ql::Constants<TOutput>::_zero();
        return true;
    }
    return false;
}

template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
void finalize(const Kokkos::View<TOutput* [3]>& res, const int i, const TScale& scalefac) {
    const TScale scalefac2 = scalefac * scalefac;
    res(i, 0) /= scalefac2;
    res(i, 1) /= scalefac2;
    res(i, 2) /= scalefac2;
}

template<typename TMass>
KOKKOS_INLINE_FUNCTION
Kokkos::Array<Kokkos::Array<TMass, 4>, 4> build_Y(const Kokkos::Array<TMass, 13>& x) {
    Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Y;
    Y[0][0] = x[0];
    Y[1][1] = x[1];
    Y[2][2] = x[2];
    Y[3][3] = x[3];
    Y[0][1] = Y[1][0] = ql::Constants<TMass>::_half() * (x[0] + x[1] - x[4]);
    Y[0][2] = Y[2][0] = ql::Constants<TMass>::_half() * (x[0] + x[2] - x[8]);
    Y[0][3] = Y[3][0] = ql::Constants<TMass>::_half() * (x[0] + x[3] - x[7]);
    Y[1][2] = Y[2][1] = ql::Constants<TMass>::_half() * (x[1] + x[2] - x[5]);
    Y[1][3] = Y[3][1] = ql::Constants<TMass>::_half() * (x[1] + x[3] - x[9]);
    Y[2][3] = Y[3][2] = ql::Constants<TMass>::_half() * (x[2] + x[3] - x[6]);
    return Y;
}

template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
void wrong_branch(const Kokkos::View<TOutput* [3]>& res, const int i, const char* who) {
    Kokkos::printf("%s: landed in a branch this direct driver did not expect\n", who);
    res(i, 0) = ql::Constants<TOutput>::_zero();
    res(i, 1) = ql::Constants<TOutput>::_zero();
    res(i, 2) = ql::Constants<TOutput>::_zero();
}

//
// ---- B0m group leaves: BIN0, B1, B2, B3, B4, B5 ----
// classify_offshell_b0m() + swap_b0m()/jsort_b0m() from box/classify.h
// decide the permutation; the offshell count also picks the leaf, which
// each wrapper below checks and reports a wrong_branch() on mismatch
// rather than silently calling a different leaf.
//

#include "qcdloop/box/leaves/BIN0.h"
template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
void call_BIN0(const Kokkos::View<TOutput* [3]>& res, const Kokkos::View<TScale*>& mu2,
               const Kokkos::View<TMass* [4]>& m, const Kokkos::View<TScale* [6]>& p, const int i) {
    TScale scalefac, musq; Kokkos::Array<TMass, 13> xpi;
    prep_xpi<TOutput, TMass, TScale>(mu2, m, p, i, scalefac, xpi, musq);
    if (early_out<TOutput, TMass, TScale>(xpi, res, i)) return;
    const ql::ClassifyB0m cls = ql::classify_offshell_b0m<TOutput, TMass, TScale>(xpi);
    if (cls.offshell != 4) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_BIN0"); return; }
    const auto Y = build_Y<TMass>(xpi);
    ql::BIN0<TOutput, TMass, TScale>(res, Y, i);
    finalize<TOutput, TMass, TScale>(res, i, scalefac);
}

#include "qcdloop/box/leaves/B1.h"
template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
void call_B1(const Kokkos::View<TOutput* [3]>& res, const Kokkos::View<TScale*>& mu2,
             const Kokkos::View<TMass* [4]>& m, const Kokkos::View<TScale* [6]>& p, const int i) {
    TScale scalefac, musq; Kokkos::Array<TMass, 13> xpi;
    prep_xpi<TOutput, TMass, TScale>(mu2, m, p, i, scalefac, xpi, musq);
    if (early_out<TOutput, TMass, TScale>(xpi, res, i)) return;
    const ql::ClassifyB0m cls = ql::classify_offshell_b0m<TOutput, TMass, TScale>(xpi);
    if (cls.offshell != 0) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_B1"); return; }
    const auto Y = build_Y<TMass>(xpi);
    ql::B1<TOutput, TMass, TScale>(res, Y, musq, i);
    finalize<TOutput, TMass, TScale>(res, i, scalefac);
}

#include "qcdloop/box/leaves/B2.h"
template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
void call_B2(const Kokkos::View<TOutput* [3]>& res, const Kokkos::View<TScale*>& mu2,
             const Kokkos::View<TMass* [4]>& m, const Kokkos::View<TScale* [6]>& p, const int i) {
    TScale scalefac, musq; Kokkos::Array<TMass, 13> xpi;
    prep_xpi<TOutput, TMass, TScale>(mu2, m, p, i, scalefac, xpi, musq);
    if (early_out<TOutput, TMass, TScale>(xpi, res, i)) return;
    const ql::ClassifyB0m cls = ql::classify_offshell_b0m<TOutput, TMass, TScale>(xpi);
    if (cls.offshell != 1) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_B2"); return; }
    Kokkos::Array<TMass, 13> xo;
    for (size_t j = 0; j < 13; j++) xo[ql::swap_b0m(j, cls.jsort1 - 1)] = xpi[j];
    const auto Y = build_Y<TMass>(xo);
    ql::B2<TOutput, TMass, TScale>(res, Y, musq, i);
    finalize<TOutput, TMass, TScale>(res, i, scalefac);
}

#include "qcdloop/box/leaves/B3.h"
template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
void call_B3(const Kokkos::View<TOutput* [3]>& res, const Kokkos::View<TScale*>& mu2,
             const Kokkos::View<TMass* [4]>& m, const Kokkos::View<TScale* [6]>& p, const int i) {
    TScale scalefac, musq; Kokkos::Array<TMass, 13> xpi;
    prep_xpi<TOutput, TMass, TScale>(mu2, m, p, i, scalefac, xpi, musq);
    if (early_out<TOutput, TMass, TScale>(xpi, res, i)) return;
    const ql::ClassifyB0m cls = ql::classify_offshell_b0m<TOutput, TMass, TScale>(xpi);
    const int jdiff = cls.jsort2 - cls.jsort1;
    if (cls.offshell != 2) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_B3"); return; }
    Kokkos::Array<TMass, 13> xo;
    const int idx = (jdiff == 3) ? 0 : (cls.jsort2 - 1);
    for (size_t j = 0; j < 13; j++) xo[ql::swap_b0m(j, idx)] = xpi[j];
    if (ql::iszero<TOutput, TMass, TScale>(xo[5])) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_B3"); return; }
    const auto Y = build_Y<TMass>(xo);
    ql::B3<TOutput, TMass, TScale>(res, Y, musq, i);
    finalize<TOutput, TMass, TScale>(res, i, scalefac);
}

#include "qcdloop/box/leaves/B4.h"
template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
void call_B4(const Kokkos::View<TOutput* [3]>& res, const Kokkos::View<TScale*>& mu2,
             const Kokkos::View<TMass* [4]>& m, const Kokkos::View<TScale* [6]>& p, const int i) {
    TScale scalefac, musq; Kokkos::Array<TMass, 13> xpi;
    prep_xpi<TOutput, TMass, TScale>(mu2, m, p, i, scalefac, xpi, musq);
    if (early_out<TOutput, TMass, TScale>(xpi, res, i)) return;
    const ql::ClassifyB0m cls = ql::classify_offshell_b0m<TOutput, TMass, TScale>(xpi);
    const int jdiff = cls.jsort2 - cls.jsort1;
    if (cls.offshell != 2) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_B4"); return; }
    Kokkos::Array<TMass, 13> xo;
    const int idx = (jdiff == 3) ? 0 : (cls.jsort2 - 1);
    for (size_t j = 0; j < 13; j++) xo[ql::swap_b0m(j, idx)] = xpi[j];
    if (!ql::iszero<TOutput, TMass, TScale>(xo[5]) ||
        ql::iszero<TOutput, TMass, TScale>(xo[6]) || ql::iszero<TOutput, TMass, TScale>(xo[7])) {
        wrong_branch<TOutput, TMass, TScale>(res, i, "call_B4"); return;
    }
    const auto Y = build_Y<TMass>(xo);
    ql::B4<TOutput, TMass, TScale>(res, Y, musq, i);
    finalize<TOutput, TMass, TScale>(res, i, scalefac);
}

#include "qcdloop/box/leaves/B5.h"
template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
void call_B5(const Kokkos::View<TOutput* [3]>& res, const Kokkos::View<TScale*>& mu2,
             const Kokkos::View<TMass* [4]>& m, const Kokkos::View<TScale* [6]>& p, const int i) {
    TScale scalefac, musq; Kokkos::Array<TMass, 13> xpi;
    prep_xpi<TOutput, TMass, TScale>(mu2, m, p, i, scalefac, xpi, musq);
    if (early_out<TOutput, TMass, TScale>(xpi, res, i)) return;
    const ql::ClassifyB0m cls = ql::classify_offshell_b0m<TOutput, TMass, TScale>(xpi);
    if (cls.offshell != 3) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_B5"); return; }
    Kokkos::Array<TMass, 13> xo;
    for (size_t j = 0; j < 13; j++) xo[ql::swap_b0m(j, ql::jsort_b0m(cls.jsort0))] = xpi[j];
    const auto Y = build_Y<TMass>(xo);
    ql::B5<TOutput, TMass, TScale>(res, Y, musq, i);
    finalize<TOutput, TMass, TScale>(res, i, scalefac);
}

//
// ---- B1m group leaves: BIN1, B6, B7, B8, B9, B10 ----
// classify_mass_b1m()/swap_b1m() pick the permutation; the Y-zero-pattern
// (zY01/zY12/zY23/zY30) picks the leaf and, for some leaves, a second
// Ycalc-permuted path. Each wrapper replicates the same else-if chain
// box/dispatch/B1m.h uses, so only the owning branch is a real leaf call --
// every other branch is a wrong_branch() report, never a sibling leaf call.
//

template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
void b1m_prep(const Kokkos::Array<TMass, 13>& xpi, Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
              bool& zY01, bool& zY12, bool& zY23, bool& zY30) {
    const int jsort = ql::classify_mass_b1m<TOutput, TMass, TScale>(xpi);
    Kokkos::Array<TMass, 13> xo;
    for (size_t j = 0; j < 13; j++) xo[ql::swap_b1m(j, jsort)] = xpi[j];
    Y = build_Y<TMass>(xo);
    zY01 = ql::iszero<TOutput, TMass, TScale>(Y[0][1]);
    zY12 = ql::iszero<TOutput, TMass, TScale>(Y[1][2]);
    zY23 = ql::iszero<TOutput, TMass, TScale>(Y[2][3]);
    zY30 = ql::iszero<TOutput, TMass, TScale>(Y[3][0]);
}

#include "qcdloop/box/leaves/B6.h"
template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
void call_B6(const Kokkos::View<TOutput* [3]>& res, const Kokkos::View<TScale*>& mu2,
             const Kokkos::View<TMass* [4]>& m, const Kokkos::View<TScale* [6]>& p, const int i) {
    TScale scalefac, musq; Kokkos::Array<TMass, 13> xpi;
    prep_xpi<TOutput, TMass, TScale>(mu2, m, p, i, scalefac, xpi, musq);
    if (early_out<TOutput, TMass, TScale>(xpi, res, i)) return;
    Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Y; bool zY01, zY12, zY23, zY30;
    b1m_prep<TOutput, TMass, TScale>(xpi, Y, zY01, zY12, zY23, zY30);
    if (zY01 && zY12 && zY23 && zY30) {
        ql::B6<TOutput, TMass, TScale>(res, Y, musq, i);
    } else {
        wrong_branch<TOutput, TMass, TScale>(res, i, "call_B6"); return;
    }
    finalize<TOutput, TMass, TScale>(res, i, scalefac);
}

#include "qcdloop/box/leaves/B7.h"
template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
void call_B7(const Kokkos::View<TOutput* [3]>& res, const Kokkos::View<TScale*>& mu2,
             const Kokkos::View<TMass* [4]>& m, const Kokkos::View<TScale* [6]>& p, const int i) {
    TScale scalefac, musq; Kokkos::Array<TMass, 13> xpi;
    prep_xpi<TOutput, TMass, TScale>(mu2, m, p, i, scalefac, xpi, musq);
    if (early_out<TOutput, TMass, TScale>(xpi, res, i)) return;
    Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Y; bool zY01, zY12, zY23, zY30;
    b1m_prep<TOutput, TMass, TScale>(xpi, Y, zY01, zY12, zY23, zY30);
    if (zY01 && zY12 && zY23 && zY30) {
        wrong_branch<TOutput, TMass, TScale>(res, i, "call_B7"); return;
    } else if (zY01 && zY12 && zY23) {
        ql::B7<TOutput, TMass, TScale>(res, Y, musq, i);
    } else if (zY01 && zY12 && zY30) {
        Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Yalt;
        ql::Ycalc<TOutput, TMass, TScale>(Y, Yalt, 1);
        ql::B7<TOutput, TMass, TScale>(res, Yalt, musq, i);
    } else {
        wrong_branch<TOutput, TMass, TScale>(res, i, "call_B7"); return;
    }
    finalize<TOutput, TMass, TScale>(res, i, scalefac);
}

#include "qcdloop/box/leaves/B8.h"
template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
void call_B8(const Kokkos::View<TOutput* [3]>& res, const Kokkos::View<TScale*>& mu2,
             const Kokkos::View<TMass* [4]>& m, const Kokkos::View<TScale* [6]>& p, const int i) {
    TScale scalefac, musq; Kokkos::Array<TMass, 13> xpi;
    prep_xpi<TOutput, TMass, TScale>(mu2, m, p, i, scalefac, xpi, musq);
    if (early_out<TOutput, TMass, TScale>(xpi, res, i)) return;
    Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Y; bool zY01, zY12, zY23, zY30;
    b1m_prep<TOutput, TMass, TScale>(xpi, Y, zY01, zY12, zY23, zY30);
    if (zY01 && zY12 && zY23 && zY30) {
        wrong_branch<TOutput, TMass, TScale>(res, i, "call_B8"); return;
    } else if (zY01 && zY12 && zY23) {
        wrong_branch<TOutput, TMass, TScale>(res, i, "call_B8"); return;
    } else if (zY01 && zY12 && zY30) {
        wrong_branch<TOutput, TMass, TScale>(res, i, "call_B8"); return;
    } else if (zY01 && zY12) {
        ql::B8<TOutput, TMass, TScale>(res, Y, musq, i);
    } else {
        wrong_branch<TOutput, TMass, TScale>(res, i, "call_B8"); return;
    }
    finalize<TOutput, TMass, TScale>(res, i, scalefac);
}

#include "qcdloop/box/leaves/B9.h"
template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
void call_B9(const Kokkos::View<TOutput* [3]>& res, const Kokkos::View<TScale*>& mu2,
             const Kokkos::View<TMass* [4]>& m, const Kokkos::View<TScale* [6]>& p, const int i) {
    TScale scalefac, musq; Kokkos::Array<TMass, 13> xpi;
    prep_xpi<TOutput, TMass, TScale>(mu2, m, p, i, scalefac, xpi, musq);
    if (early_out<TOutput, TMass, TScale>(xpi, res, i)) return;
    Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Y; bool zY01, zY12, zY23, zY30;
    b1m_prep<TOutput, TMass, TScale>(xpi, Y, zY01, zY12, zY23, zY30);
    if (zY01 && zY12 && zY23 && zY30) {
        wrong_branch<TOutput, TMass, TScale>(res, i, "call_B9"); return;
    } else if (zY01 && zY12 && zY23) {
        wrong_branch<TOutput, TMass, TScale>(res, i, "call_B9"); return;
    } else if (zY01 && zY12 && zY30) {
        wrong_branch<TOutput, TMass, TScale>(res, i, "call_B9"); return;
    } else if (zY01 && zY12) {
        wrong_branch<TOutput, TMass, TScale>(res, i, "call_B9"); return;
    } else if (zY01 && zY30) {
        ql::B9<TOutput, TMass, TScale>(res, Y, musq, i);
    } else if (zY12 && zY23) {
        Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Yalt;
        ql::Ycalc<TOutput, TMass, TScale>(Y, Yalt, 1);
        ql::B9<TOutput, TMass, TScale>(res, Yalt, musq, i);
    } else {
        wrong_branch<TOutput, TMass, TScale>(res, i, "call_B9"); return;
    }
    finalize<TOutput, TMass, TScale>(res, i, scalefac);
}

#include "qcdloop/box/leaves/B10.h"
template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
void call_B10(const Kokkos::View<TOutput* [3]>& res, const Kokkos::View<TScale*>& mu2,
              const Kokkos::View<TMass* [4]>& m, const Kokkos::View<TScale* [6]>& p, const int i) {
    TScale scalefac, musq; Kokkos::Array<TMass, 13> xpi;
    prep_xpi<TOutput, TMass, TScale>(mu2, m, p, i, scalefac, xpi, musq);
    if (early_out<TOutput, TMass, TScale>(xpi, res, i)) return;
    Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Y; bool zY01, zY12, zY23, zY30;
    b1m_prep<TOutput, TMass, TScale>(xpi, Y, zY01, zY12, zY23, zY30);
    if (zY01 && zY12 && zY23 && zY30) {
        wrong_branch<TOutput, TMass, TScale>(res, i, "call_B10"); return;
    } else if (zY01 && zY12 && zY23) {
        wrong_branch<TOutput, TMass, TScale>(res, i, "call_B10"); return;
    } else if (zY01 && zY12 && zY30) {
        wrong_branch<TOutput, TMass, TScale>(res, i, "call_B10"); return;
    } else if (zY01 && zY12) {
        wrong_branch<TOutput, TMass, TScale>(res, i, "call_B10"); return;
    } else if (zY01 && zY30) {
        wrong_branch<TOutput, TMass, TScale>(res, i, "call_B10"); return;
    } else if (zY12 && zY23) {
        wrong_branch<TOutput, TMass, TScale>(res, i, "call_B10"); return;
    } else if (zY01) {
        ql::B10<TOutput, TMass, TScale>(res, Y, musq, i);
    } else if (zY12) {
        Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Yalt;
        ql::Ycalc<TOutput, TMass, TScale>(Y, Yalt, 1);
        ql::B10<TOutput, TMass, TScale>(res, Yalt, musq, i);
    } else {
        wrong_branch<TOutput, TMass, TScale>(res, i, "call_B10"); return;
    }
    finalize<TOutput, TMass, TScale>(res, i, scalefac);
}

#include "qcdloop/box/leaves/BIN1.h"
template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
void call_BIN1(const Kokkos::View<TOutput* [3]>& res, const Kokkos::View<TScale*>& mu2,
                const Kokkos::View<TMass* [4]>& m, const Kokkos::View<TScale* [6]>& p, const int i) {
    TScale scalefac, musq; Kokkos::Array<TMass, 13> xpi;
    prep_xpi<TOutput, TMass, TScale>(mu2, m, p, i, scalefac, xpi, musq);
    if (early_out<TOutput, TMass, TScale>(xpi, res, i)) return;
    Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Y; bool zY01, zY12, zY23, zY30;
    b1m_prep<TOutput, TMass, TScale>(xpi, Y, zY01, zY12, zY23, zY30);
    if (zY01 && zY12 && zY23 && zY30) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_BIN1"); return; }
    if (zY01 && zY12 && zY23) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_BIN1"); return; }
    if (zY01 && zY12 && zY30) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_BIN1"); return; }
    if (zY01 && zY12) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_BIN1"); return; }
    if (zY01 && zY30) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_BIN1"); return; }
    if (zY12 && zY23) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_BIN1"); return; }
    if (zY01) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_BIN1"); return; }
    if (zY12) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_BIN1"); return; }
    ql::BIN1<TOutput, TMass, TScale>(res, Y, i);
    finalize<TOutput, TMass, TScale>(res, i, scalefac);
}

//
// ---- B2m group leaves: BIN2, B11, B12, B13, B14, B15 ----
// classify_mass_b2m()/swap_b2m() pick the permutation and the adjacent
// (B2ma) vs. opposite (B2mo) sub-path via jdiff. Each wrapper below
// replicates the top-level split plus whichever one sub-path's chain it
// actually needs (B2ma's or B2mo's), per box/dispatch/B2m.h.
//

template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
bool b2m_prep(const Kokkos::Array<TMass, 13>& xpi, Kokkos::Array<TMass, 13>& xo, int& jdiff) {
    const ql::ClassifyB2m c = ql::classify_mass_b2m<TOutput, TMass, TScale>(xpi);
    jdiff = c.jdiff;
    if (jdiff == 1 || jdiff == 2) {
        for (size_t j = 0; j < 13; j++) xo[ql::swap_b2m(j, c.jsort2)] = xpi[j];
    } else if (jdiff == 3) {
        for (size_t j = 0; j < 13; j++) xo[ql::swap_b2m(j, 0)] = xpi[j];
    } else {
        return false;
    }
    return true;
}

#include "qcdloop/box/leaves/BIN2.h"
template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
void call_BIN2(const Kokkos::View<TOutput* [3]>& res, const Kokkos::View<TScale*>& mu2,
                const Kokkos::View<TMass* [4]>& m, const Kokkos::View<TScale* [6]>& p, const int i) {
    TScale scalefac, musq; Kokkos::Array<TMass, 13> xpi;
    prep_xpi<TOutput, TMass, TScale>(mu2, m, p, i, scalefac, xpi, musq);
    if (early_out<TOutput, TMass, TScale>(xpi, res, i)) return;
    Kokkos::Array<TMass, 13> xo; int jdiff;
    if (!b2m_prep<TOutput, TMass, TScale>(xpi, xo, jdiff)) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_BIN2"); return; }
    if (jdiff == 2) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_BIN2"); return; } // B2mo path not needed by this driver's BIN2 section
    // B2ma path
    Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Y = build_Y<TMass>(xo);
    Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Yalt;
    ql::Ycalc<TOutput, TMass, TScale>(Y, Yalt, 2, ql::iszero<TOutput, TMass, TScale>(xo[2]));
    const bool zY01 = ql::iszero<TOutput, TMass, TScale>(Y[0][1]);
    const bool zY12 = ql::iszero<TOutput, TMass, TScale>(Y[1][2]);
    const bool zY03 = ql::iszero<TOutput, TMass, TScale>(Y[0][3]);
    if (zY01 && zY12 && zY03) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_BIN2"); return; }
    if (zY01 && zY12 && !zY03) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_BIN2"); return; }
    if (ql::iszero<TOutput, TMass, TScale>(Yalt[0][1]) && ql::iszero<TOutput, TMass, TScale>(Yalt[1][2]) && !ql::iszero<TOutput, TMass, TScale>(Yalt[0][3])) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_BIN2"); return; }
    if (zY01 && !zY12 && !zY03) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_BIN2"); return; }
    ql::BIN2<TOutput, TMass, TScale>(res, Y, i);
    finalize<TOutput, TMass, TScale>(res, i, scalefac);
}

#include "qcdloop/box/leaves/B11.h"
template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
void call_B11(const Kokkos::View<TOutput* [3]>& res, const Kokkos::View<TScale*>& mu2,
               const Kokkos::View<TMass* [4]>& m, const Kokkos::View<TScale* [6]>& p, const int i) {
    TScale scalefac, musq; Kokkos::Array<TMass, 13> xpi;
    prep_xpi<TOutput, TMass, TScale>(mu2, m, p, i, scalefac, xpi, musq);
    if (early_out<TOutput, TMass, TScale>(xpi, res, i)) return;
    Kokkos::Array<TMass, 13> xo; int jdiff;
    if (!b2m_prep<TOutput, TMass, TScale>(xpi, xo, jdiff) || jdiff == 2) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_B11"); return; }
    const auto Y = build_Y<TMass>(xo);
    const bool zY01 = ql::iszero<TOutput, TMass, TScale>(Y[0][1]);
    const bool zY12 = ql::iszero<TOutput, TMass, TScale>(Y[1][2]);
    const bool zY03 = ql::iszero<TOutput, TMass, TScale>(Y[0][3]);
    if (!(zY01 && zY12 && zY03)) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_B11"); return; }
    ql::B11<TOutput, TMass, TScale>(res, Y, musq, i);
    finalize<TOutput, TMass, TScale>(res, i, scalefac);
}

#include "qcdloop/box/leaves/B12.h"
template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
void call_B12(const Kokkos::View<TOutput* [3]>& res, const Kokkos::View<TScale*>& mu2,
               const Kokkos::View<TMass* [4]>& m, const Kokkos::View<TScale* [6]>& p, const int i) {
    TScale scalefac, musq; Kokkos::Array<TMass, 13> xpi;
    prep_xpi<TOutput, TMass, TScale>(mu2, m, p, i, scalefac, xpi, musq);
    if (early_out<TOutput, TMass, TScale>(xpi, res, i)) return;
    Kokkos::Array<TMass, 13> xo; int jdiff;
    if (!b2m_prep<TOutput, TMass, TScale>(xpi, xo, jdiff) || jdiff == 2) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_B12"); return; }
    Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Y = build_Y<TMass>(xo);
    Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Yalt;
    ql::Ycalc<TOutput, TMass, TScale>(Y, Yalt, 2, ql::iszero<TOutput, TMass, TScale>(xo[2]));
    const bool zY01 = ql::iszero<TOutput, TMass, TScale>(Y[0][1]);
    const bool zY12 = ql::iszero<TOutput, TMass, TScale>(Y[1][2]);
    const bool zY03 = ql::iszero<TOutput, TMass, TScale>(Y[0][3]);
    if (zY01 && zY12 && zY03) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_B12"); return; }
    if (zY01 && zY12 && !zY03) {
        ql::B12<TOutput, TMass, TScale>(res, Y, musq, i);
    } else if (ql::iszero<TOutput, TMass, TScale>(Yalt[0][1]) && ql::iszero<TOutput, TMass, TScale>(Yalt[1][2]) && !ql::iszero<TOutput, TMass, TScale>(Yalt[0][3])) {
        ql::B12<TOutput, TMass, TScale>(res, Yalt, musq, i);
    } else {
        wrong_branch<TOutput, TMass, TScale>(res, i, "call_B12"); return;
    }
    finalize<TOutput, TMass, TScale>(res, i, scalefac);
}

#include "qcdloop/box/leaves/B13.h"
template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
void call_B13(const Kokkos::View<TOutput* [3]>& res, const Kokkos::View<TScale*>& mu2,
               const Kokkos::View<TMass* [4]>& m, const Kokkos::View<TScale* [6]>& p, const int i) {
    TScale scalefac, musq; Kokkos::Array<TMass, 13> xpi;
    prep_xpi<TOutput, TMass, TScale>(mu2, m, p, i, scalefac, xpi, musq);
    if (early_out<TOutput, TMass, TScale>(xpi, res, i)) return;
    Kokkos::Array<TMass, 13> xo; int jdiff;
    if (!b2m_prep<TOutput, TMass, TScale>(xpi, xo, jdiff) || jdiff == 2) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_B13"); return; }
    const auto Y = build_Y<TMass>(xo);
    const bool zY01 = ql::iszero<TOutput, TMass, TScale>(Y[0][1]);
    const bool zY12 = ql::iszero<TOutput, TMass, TScale>(Y[1][2]);
    const bool zY03 = ql::iszero<TOutput, TMass, TScale>(Y[0][3]);
    if (!(zY01 && !zY12 && !zY03)) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_B13"); return; }
    ql::B13<TOutput, TMass, TScale>(res, Y, musq, i);
    finalize<TOutput, TMass, TScale>(res, i, scalefac);
}

#include "qcdloop/box/leaves/B14.h"
template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
void call_B14(const Kokkos::View<TOutput* [3]>& res, const Kokkos::View<TScale*>& mu2,
               const Kokkos::View<TMass* [4]>& m, const Kokkos::View<TScale* [6]>& p, const int i) {
    TScale scalefac, musq; Kokkos::Array<TMass, 13> xpi;
    prep_xpi<TOutput, TMass, TScale>(mu2, m, p, i, scalefac, xpi, musq);
    if (early_out<TOutput, TMass, TScale>(xpi, res, i)) return;
    Kokkos::Array<TMass, 13> xo; int jdiff;
    if (!b2m_prep<TOutput, TMass, TScale>(xpi, xo, jdiff) || jdiff != 2) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_B14"); return; }
    const auto Y = build_Y<TMass>(xo);
    const bool zY00 = ql::iszero<TOutput, TMass, TScale>(Y[0][0]);
    const bool zY22 = ql::iszero<TOutput, TMass, TScale>(Y[2][2]);
    const bool zY01 = ql::iszero<TOutput, TMass, TScale>(Y[0][1]);
    const bool zY03 = ql::iszero<TOutput, TMass, TScale>(Y[0][3]);
    const bool zY12 = ql::iszero<TOutput, TMass, TScale>(Y[1][2]);
    const bool zY23 = ql::iszero<TOutput, TMass, TScale>(Y[2][3]);
    if (!(zY00 && zY22 && zY01 && zY03 && zY12 && zY23)) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_B14"); return; }
    ql::B14<TOutput, TMass, TScale>(res, Y, musq, i);
    finalize<TOutput, TMass, TScale>(res, i, scalefac);
}

#include "qcdloop/box/leaves/B15.h"
template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
void call_B15(const Kokkos::View<TOutput* [3]>& res, const Kokkos::View<TScale*>& mu2,
               const Kokkos::View<TMass* [4]>& m, const Kokkos::View<TScale* [6]>& p, const int i) {
    TScale scalefac, musq; Kokkos::Array<TMass, 13> xpi;
    prep_xpi<TOutput, TMass, TScale>(mu2, m, p, i, scalefac, xpi, musq);
    if (early_out<TOutput, TMass, TScale>(xpi, res, i)) return;
    Kokkos::Array<TMass, 13> xo; int jdiff;
    if (!b2m_prep<TOutput, TMass, TScale>(xpi, xo, jdiff) || jdiff != 2) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_B15"); return; }
    Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Y = build_Y<TMass>(xo);
    const bool zY00 = ql::iszero<TOutput, TMass, TScale>(Y[0][0]);
    const bool zY22 = ql::iszero<TOutput, TMass, TScale>(Y[2][2]);
    const bool zY01 = ql::iszero<TOutput, TMass, TScale>(Y[0][1]);
    const bool zY03 = ql::iszero<TOutput, TMass, TScale>(Y[0][3]);
    const bool zY12 = ql::iszero<TOutput, TMass, TScale>(Y[1][2]);
    const bool zY23 = ql::iszero<TOutput, TMass, TScale>(Y[2][3]);
    if (zY00 && zY22 && zY01 && zY03 && zY12 && zY23) { wrong_branch<TOutput, TMass, TScale>(res, i, "call_B15"); return; }
    if (zY00 && zY22 && zY01 && zY03) {
        ql::B15<TOutput, TMass, TScale>(res, Y, musq, i);
    } else if (zY00 && zY22 && zY12 && zY23) {
        Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Yalt;
        ql::Ycalc<TOutput, TMass, TScale>(Y, Yalt, 2, ql::iszero<TOutput, TMass, TScale>(xo[2]));
        ql::B15<TOutput, TMass, TScale>(res, Yalt, musq, i);
    } else {
        wrong_branch<TOutput, TMass, TScale>(res, i, "call_B15"); return;
    }
    finalize<TOutput, TMass, TScale>(res, i, scalefac);
}

//
// ---- B3m group leaves: BIN3, B16 ----
//

#include "qcdloop/box/leaves/BIN3.h"
template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
void call_BIN3(const Kokkos::View<TOutput* [3]>& res, const Kokkos::View<TScale*>& mu2,
                const Kokkos::View<TMass* [4]>& m, const Kokkos::View<TScale* [6]>& p, const int i) {
    TScale scalefac, musq; Kokkos::Array<TMass, 13> xpi;
    prep_xpi<TOutput, TMass, TScale>(mu2, m, p, i, scalefac, xpi, musq);
    if (early_out<TOutput, TMass, TScale>(xpi, res, i)) return;
    const int jsort = ql::classify_mass_b3m<TOutput, TMass, TScale>(xpi);
    Kokkos::Array<TMass, 13> xo;
    for (size_t j = 0; j < 13; j++) xo[ql::swap_b3m(j, jsort)] = xpi[j];
    const auto Y = build_Y<TMass>(xo);
    if (ql::iszero<TOutput, TMass, TScale>(Y[0][0]) && ql::iszero<TOutput, TMass, TScale>(Y[0][1]) && ql::iszero<TOutput, TMass, TScale>(Y[0][3])) {
        wrong_branch<TOutput, TMass, TScale>(res, i, "call_BIN3"); return;
    }
    ql::BIN3<TOutput, TMass, TScale>(res, Y, i);
    finalize<TOutput, TMass, TScale>(res, i, scalefac);
}

#include "qcdloop/box/leaves/B16.h"
template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
void call_B16(const Kokkos::View<TOutput* [3]>& res, const Kokkos::View<TScale*>& mu2,
              const Kokkos::View<TMass* [4]>& m, const Kokkos::View<TScale* [6]>& p, const int i) {
    TScale scalefac, musq; Kokkos::Array<TMass, 13> xpi;
    prep_xpi<TOutput, TMass, TScale>(mu2, m, p, i, scalefac, xpi, musq);
    if (early_out<TOutput, TMass, TScale>(xpi, res, i)) return;
    const int jsort = ql::classify_mass_b3m<TOutput, TMass, TScale>(xpi);
    Kokkos::Array<TMass, 13> xo;
    for (size_t j = 0; j < 13; j++) xo[ql::swap_b3m(j, jsort)] = xpi[j];
    const auto Y = build_Y<TMass>(xo);
    if (!(ql::iszero<TOutput, TMass, TScale>(Y[0][0]) && ql::iszero<TOutput, TMass, TScale>(Y[0][1]) && ql::iszero<TOutput, TMass, TScale>(Y[0][3]))) {
        wrong_branch<TOutput, TMass, TScale>(res, i, "call_B16"); return;
    }
    ql::B16<TOutput, TMass, TScale>(res, Y, musq, i);
    finalize<TOutput, TMass, TScale>(res, i, scalefac);
}

//
// ---- B4m group leaf: BIN4 ----
// No data-dependent classification -- just build Y and call BIN4.
//

#include "qcdloop/box/leaves/BIN4.h"
template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
void call_BIN4(const Kokkos::View<TOutput* [3]>& res, const Kokkos::View<TScale*>& mu2,
                const Kokkos::View<TMass* [4]>& m, const Kokkos::View<TScale* [6]>& p, const int i) {
    TScale scalefac, musq; Kokkos::Array<TMass, 13> xpi;
    prep_xpi<TOutput, TMass, TScale>(mu2, m, p, i, scalefac, xpi, musq);
    if (early_out<TOutput, TMass, TScale>(xpi, res, i)) return;
    const auto Y = build_Y<TMass>(xpi);
    ql::BIN4<TOutput, TMass, TScale>(res, Y, i);
    finalize<TOutput, TMass, TScale>(res, i, scalefac);
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        int mode = 1;
        int batch_size = 1000000;

        if (argc < 2) {
            std::cout << "Usage: " << argv[0] << " <mode> [batch_size]" << std::endl;
            std::cout << "  mode: 0 for performance benchmark, 1 for accuracy test (required)" << std::endl;
            std::cout << "  batch_size: Number of batch iterations (default: 1000000)" << std::endl;
            Kokkos::finalize();
            return 1;
        }

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
            std::cout << "Target Integral,Batch size,Time" << std::endl;
        } else if (mode == 1) {
            std::cout << "Target Integral,Test ID,mu2,ms,ps,Coeff 1,Coeff 2,Coeff 3" << std::endl;
        }

        double low = 100;
        double up  = 1000000;

        Kokkos::View<double*> mu2_d("mu2", batch_size);
        Kokkos::View<double* [4]> m_d("m", batch_size);
        Kokkos::View<double* [6]> p_d("p", batch_size);
        Kokkos::View<complex* [3]> res_d("res", batch_size);

        auto mu2_h = Kokkos::create_mirror_view(mu2_d);
        auto m_h = Kokkos::create_mirror_view(m_d);
        auto p_h = Kokkos::create_mirror_view(p_d);
        auto res_h = Kokkos::create_mirror_view(res_d);

        for (size_t i = 0; i < batch_size; ++i) {
            mu2_h(i) = dd_mu2().hi;
        }

        ql::Timer tt;
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, batch_size);

        // Macro to avoid retyping the copy-in/launch/copy-out/print boilerplate
        // for each of the 21 sections -- the boilerplate itself is identical to
        // boxGPU_test.cc's, only the lambda body (the direct leaf call) differs.
#define RUN_SECTION(NAME, CALLFN) \
        Kokkos::deep_copy(mu2_d, mu2_h); \
        Kokkos::deep_copy(m_d, m_h); \
        Kokkos::deep_copy(p_d, p_h); \
        tt.start(); \
        Kokkos::parallel_for("Box Integral " NAME, policy, KOKKOS_LAMBDA(const int& i) { \
            CALLFN<complex, double, double>(res_d, mu2_d, m_d, p_d, i); \
        }); \
        Kokkos::fence(); \
        { \
        double elapsed = tt.stop(); \
        Kokkos::deep_copy(res_h, res_d); \
        if (mode == 0) { \
            std::cout << NAME << "," << batch_size << "," << elapsed << std::endl; \
        } else if (mode == 1) { \
            for (size_t i = 0; i < batch_size; ++i) { \
                double m_arr[4] = {m_h(i, 0), m_h(i, 1), m_h(i, 2), m_h(i, 3)}; \
                double p_arr[6] = {p_h(i, 0), p_h(i, 1), p_h(i, 2), p_h(i, 3), p_h(i, 4), p_h(i, 5)}; \
                std::cout << NAME << "," \
                          << (i+1) << "," \
                          << mu2_h(i) << "," \
                          << arrayToCSV(m_arr, 4) << "," \
                          << arrayToCSV(p_arr, 6) << "," \
                          << complexToCSV(res_h(i, 0)) << "," \
                          << complexToCSV(res_h(i, 1)) << "," \
                          << complexToCSV(res_h(i, 2)) << std::endl; \
            } \
        } \
        }

        // Trigger BIN0 - BIN4 (n_masses = 0..4, matching boxGPU_test.cc's loop)

        // n_masses = 0 -> BIN0
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            for(int j(0); j<4; ++j) m_h(i, j) = 0.;
            p_h(i, 0) = rs(low,up);
            p_h(i, 1) = rs(low,up);
            p_h(i, 2) = rs(low,up);
            p_h(i, 3) = rs(low,up);
            p_h(i, 4) = r(low,up);
            p_h(i, 5) = r(low,up);
        }
        RUN_SECTION("BIN0", call_BIN0)

        // n_masses = 1 -> BIN1
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            for(int j(0); j<4; ++j) m_h(i, j) = 0.;
            m_h(i, 0) = 10;
            p_h(i, 0) = rs(low,up);
            p_h(i, 1) = rs(low,up);
            p_h(i, 2) = rs(low,up);
            p_h(i, 3) = rs(low,up);
            p_h(i, 4) = r(low,up);
            p_h(i, 5) = r(low,up);
        }
        RUN_SECTION("BIN1", call_BIN1)

        // n_masses = 2 -> BIN2
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            for(int j(0); j<4; ++j) m_h(i, j) = 0.;
            m_h(i, 0) = 10; m_h(i, 1) = 10;
            p_h(i, 0) = rs(low,up);
            p_h(i, 1) = rs(low,up);
            p_h(i, 2) = rs(low,up);
            p_h(i, 3) = rs(low,up);
            p_h(i, 4) = r(low,up);
            p_h(i, 5) = r(low,up);
        }
        RUN_SECTION("BIN2", call_BIN2)

        // n_masses = 3 -> BIN3
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            for(int j(0); j<4; ++j) m_h(i, j) = 0.;
            m_h(i, 0) = 10; m_h(i, 1) = 10; m_h(i, 2) = 10;
            p_h(i, 0) = rs(low,up);
            p_h(i, 1) = rs(low,up);
            p_h(i, 2) = rs(low,up);
            p_h(i, 3) = rs(low,up);
            p_h(i, 4) = r(low,up);
            p_h(i, 5) = r(low,up);
        }
        RUN_SECTION("BIN3", call_BIN3)

        // n_masses = 4 -> BIN4
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            for(int j(0); j<4; ++j) m_h(i, j) = 0.;
            m_h(i, 0) = 10; m_h(i, 1) = 10; m_h(i, 2) = 10; m_h(i, 3) = 10;
            p_h(i, 0) = rs(low,up);
            p_h(i, 1) = rs(low,up);
            p_h(i, 2) = rs(low,up);
            p_h(i, 3) = rs(low,up);
            p_h(i, 4) = r(low,up);
            p_h(i, 5) = r(low,up);
        }
        RUN_SECTION("BIN4", call_BIN4)

        // Zero mass integrals - B1
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = 0.; m_h(i, 3) = 0.;
            p_h(i, 0) = 0.; p_h(i, 1) = 0.; p_h(i, 2) = 0.; p_h(i, 3) = 0.;
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        RUN_SECTION("B1", call_B1)

        // B2
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = 0.; m_h(i, 3) = 0.;
            p_h(i, 0) = 0.; p_h(i, 1) = 0.; p_h(i, 2) = 0.;
            p_h(i, 3) = rs(low,up); p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        RUN_SECTION("B2", call_B2)

        // B3
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = 0.; m_h(i, 3) = 0.;
            p_h(i, 0) = 0.; p_h(i, 1) = rs(low,up); p_h(i, 2) = 0.;
            p_h(i, 3) = rs(low,up); p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        RUN_SECTION("B3", call_B3)

        // B4
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = 0.; m_h(i, 3) = 0.;
            p_h(i, 0) = 0.; p_h(i, 1) = 0.;
            p_h(i, 2) = rs(low,up); p_h(i, 3) = rs(low,up);
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        RUN_SECTION("B4", call_B4)

        // B5
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = 0.; m_h(i, 3) = 0.;
            p_h(i, 0) = 0.;
            p_h(i, 1) = rs(low,up); p_h(i, 2) = rs(low,up); p_h(i, 3) = rs(low,up);
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        RUN_SECTION("B5", call_B5)

        // single mass integrals. 10 is exact.
        double m2 = dd_mass_10().hi;

        // B6
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = 0.; m_h(i, 3) = m2;
            p_h(i, 0) = 0.; p_h(i, 1) = 0.;
            p_h(i, 2) = m2; p_h(i, 3) = m2;
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        RUN_SECTION("B6", call_B6)

        // B7
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = 0.; m_h(i, 3) = m2;
            p_h(i, 0) = 0.; p_h(i, 1) = 0.;
            p_h(i, 2) = m2; p_h(i, 3) = rs(low,up);
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        RUN_SECTION("B7", call_B7)

        // B8
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = 0.; m_h(i, 3) = m2;
            p_h(i, 0) = 0.; p_h(i, 1) = 0.;
            p_h(i, 2) = rs(low,up); p_h(i, 3) = rs(low,up);
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        RUN_SECTION("B8", call_B8)

        // B9
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = 0.; m_h(i, 3) = m2;
            p_h(i, 0) = 0.;
            p_h(i, 1) = rs(low,up); p_h(i, 2) = rs(low,up); p_h(i, 3) = m2;
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        RUN_SECTION("B9", call_B9)

        // B10
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = 0.; m_h(i, 3) = m2;
            p_h(i, 0) = 0.;
            p_h(i, 1) = rs(low,up); p_h(i, 2) = rs(low,up); p_h(i, 3) = rs(low,up);
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        RUN_SECTION("B10", call_B10)

        // two mass integrals. 10 and 2500 are exact; 4.9^2 and 50^2 come from quad.
        double m22 = dd_mass_4p9_sq().hi;
        double m32 = dd_mass_10().hi;
        double m42 = dd_mass_50_sq().hi;

        // B11
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = m32; m_h(i, 3) = m42;
            p_h(i, 0) = 0.; p_h(i, 1) = m32;
            p_h(i, 2) = rs(low,up); p_h(i, 3) = m42;
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        RUN_SECTION("B11", call_B11)

        // B12
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = m32; m_h(i, 3) = m42;
            p_h(i, 0) = 0.;
            p_h(i, 1) = rs(low,up); p_h(i, 2) = rs(low,up); p_h(i, 3) = m42;
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        RUN_SECTION("B12", call_B12)

        // B13
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = 0.; m_h(i, 2) = m32; m_h(i, 3) = m42;
            p_h(i, 0) = 0.;
            p_h(i, 1) = rs(low,up); p_h(i, 2) = rs(low,up); p_h(i, 3) = rs(low,up);
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        RUN_SECTION("B13", call_B13)

        // B14
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = m22; m_h(i, 2) = 0.; m_h(i, 3) = m42;
            p_h(i, 0) = m22; p_h(i, 1) = m22;
            p_h(i, 2) = m42; p_h(i, 3) = m42;
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        RUN_SECTION("B14", call_B14)

        // B15
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = m22; m_h(i, 2) = 0.; m_h(i, 3) = m42;
            p_h(i, 0) = m22;
            p_h(i, 1) = rs(low,up); p_h(i, 2) = rs(low,up); p_h(i, 3) = m42;
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        RUN_SECTION("B15", call_B15)

        // three mass integrals - B16
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.; m_h(i, 1) = m22; m_h(i, 2) = m32; m_h(i, 3) = m42;
            p_h(i, 0) = m22;
            p_h(i, 1) = rs(low,up); p_h(i, 2) = rs(low,up); p_h(i, 3) = m42;
            p_h(i, 4) = r(low,up); p_h(i, 5) = r(low,up);
        }
        RUN_SECTION("B16", call_B16)

#undef RUN_SECTION
    }
    Kokkos::finalize();
    return 0;
}
