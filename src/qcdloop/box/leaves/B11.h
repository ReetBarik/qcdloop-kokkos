//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//          Stefan Hoeche   : shoeche@fnal.gov
//          Max Knobbe      : mknobbe@fnal.gov
//
// Leaf integral: box with 2 internal masses (adjacent), 3 zero Y entries.
// Pure computation on pre-built Y; no classification logic. See
// box/dispatch/B2m.h for the caller.

#pragma once

#include "kokkosUtils.h"


namespace ql
{

    // gfx90a, ROCm 7.0.2: B11's complex-root arm (ratgam + cLn) is inlined into
    // ql::BO. A wavefront with one negative discriminant takes that arm on a
    // single lane. The other lanes stay on Lnrat and come back NaN or wrong,
    // while the complex lane itself matches DoubleDouble. B12 takes the same
    // kind of branch and does not. Outlining the arm makes the divergence a
    // call, so the complex body is not in the kernel's divergent region.
#if defined(__HIPCC__)
#define QL_B11_NOINLINE __attribute__((noinline))
#else
#define QL_B11_NOINLINE
#endif

    template<typename TOutput, typename TMass, typename TScale>
    QL_B11_NOINLINE KOKKOS_INLINE_FUNCTION
    void B11_ln43_complex(
        TOutput& ln43p, TOutput& ln43m,
        TMass const& p3sq, TMass const& m4sq, TMass const& m3sq) {
        TOutput rat2p, rat2m;
        TScale ieps2;
        ql::ratgam<TOutput, TMass, TScale>(rat2p, rat2m, ieps2, p3sq, m4sq, m3sq);
        ln43p = ql::cLn<TOutput, TMass, TScale>(rat2p, ieps2);
        ln43m = ql::cLn<TOutput, TMass, TScale>(rat2m, ieps2);
    }

#undef QL_B11_NOINLINE

    /*!
    * Implementation of the formulae from Ellis et al. \cite Ellis:2007qk.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B11(
        const Kokkos::View<TOutput* [3]>& res,
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass m3sq = Y[2][2];
        const TMass m4sq = Y[3][3];
        const TMass sibar = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass tabar = ql::Constants<TMass>::_two() * Y[1][3];
        const TMass p3sq = -(ql::Constants<TMass>::_two() * Y[2][3] - Y[2][2] - Y[3][3]);
        const TMass m3mu = ql::kSqrt(m3sq * mu2);
        const TMass m4mu = ql::kSqrt(m4sq * mu2);

        const TOutput wlogt = ql::Lnrat<TOutput, TMass, TScale>(tabar, m4mu);
        const TOutput wlogs = ql::Lnrat<TOutput, TMass, TScale>(sibar, m3mu);

        TOutput root;
        TMass x43p, x43pm1, x43m, x43mm1;
        if (ql::iszero<TOutput, TMass, TScale>(p3sq)) {
            root = ql::Constants<TOutput>::_one();
            x43p = -ql::Constants<TMass>::_one();
            x43pm1 = -ql::Constants<TMass>::_one();
            x43m = m3sq;
            x43mm1 = m4sq;
        } else {
            root = ql::kSqrt(TOutput(ql::kPow<TOutput, TMass, TScale>(p3sq + m3sq - m4sq, 2) - ql::Constants<TMass>::_four() * m3sq * p3sq));
            const auto g43   = ql::rootpair<TOutput, TMass, TScale>(TOutput(p3sq + m3sq - m4sq), root, TOutput(ql::Constants<TMass>::_four() * m3sq * p3sq));
            const auto g43m1 = ql::rootpair<TOutput, TMass, TScale>(TOutput(-p3sq + m3sq - m4sq), root, TOutput(ql::Constants<TMass>::_four() * m4sq * p3sq));
            const TOutput ga43p = g43[0], ga43m = g43[1], ga43pm1 = g43m1[0], ga43mm1 = g43m1[1];

            x43p = -ql::Real(ga43p);
            x43pm1 = -ql::Real(ga43pm1);
            x43m = ql::Real(ga43m);
            x43mm1 = ql::Real(ga43mm1);
        }

        // deal with real roots
        TOutput ln43p, ln43m;
        if (ql::iszero<TOutput, TMass, TScale>(ql::Imag(root))) {
            ln43p = ql::Lnrat<TOutput, TMass, TScale>(x43p, x43pm1);
            ln43m = ql::Lnrat<TOutput, TMass, TScale>(x43m, x43mm1);
        } else {
            ql::B11_ln43_complex<TOutput, TMass, TScale>(ln43p, ln43m, p3sq, m4sq, m3sq);
        }

        TOutput intbit;
        if (ql::iszero<TOutput, TMass, TScale>(p3sq)) {
            intbit = -ql::Constants<TOutput>::_half() * TOutput(ql::kPow<TOutput, TMass, TScale>(ql::kLog(m3sq / m4sq), 2));
        } else {
            intbit = -ql::Constants<TOutput>::_half() * (ql::kPow<TOutput, TMass, TScale>(ln43p, 2) + ql::kPow<TOutput, TMass, TScale>(ln43m, 2));
        }

        res(i,2) = ql::Constants<TOutput>::_one();
        res(i,1) = -wlogt - wlogs;
        res(i,0) = intbit
            + ql::Constants<TOutput>::_two() * wlogt * wlogs - TOutput(ql::Constants<TScale>::_half() * ql::Constants<TScale>::_pi2())
            + TOutput(ql::kPow<TOutput, TMass, TScale>(ql::kLog(m3sq / m4sq), 2) / ql::Constants<TScale>::_four());

        const TOutput d = TOutput(sibar * tabar);
        for (size_t j = 0; j < 3; j++) {
            res(i,j) /= d;
        }
    }

}
