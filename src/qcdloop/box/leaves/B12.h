//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//          Stefan Hoeche   : shoeche@fnal.gov
//          Max Knobbe      : mknobbe@fnal.gov
//
// Leaf integral: box with 2 internal masses (adjacent), 2 zero Y entries.
// Pure computation on pre-built Y; no classification logic. See
// box/dispatch/B2m.h for the caller.

#pragma once

#include "kokkosUtils.h"


namespace ql
{

    /*!
    * Implementation of the formulae from Ellis et al. \cite Ellis:2007qk.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B12(
        const Kokkos::View<TOutput* [3]>& res,
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass m3sq = Y[2][2];
        const TMass m4sq = Y[3][3];
        const TMass sibar = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass tabar = ql::Constants<TMass>::_two() * Y[1][3];
        const TMass m4sqbar = ql::Constants<TMass>::_two() * Y[0][3];
        const TMass p3sq = -(ql::Constants<TMass>::_two() * Y[2][3] - Y[2][2] - Y[3][3]);

        const TMass mean = ql::kSqrt(mu2 * ql::Real(m3sq));
        const TOutput fac = TOutput(sibar * tabar);

        const TOutput wlogsmu = ql::Lnrat<TOutput, TMass, TScale>(sibar, mean);
        const TOutput wlogtmu = ql::Lnrat<TOutput, TMass, TScale>(tabar, mean);
        const TOutput wlog4mu = ql::Lnrat<TOutput, TMass, TScale>(m4sqbar, mean);
        const TOutput wlog = wlogsmu + wlogtmu - wlog4mu;

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

        const TOutput dilog1 = ql::Li2omrat<TOutput, TMass, TScale>(m4sqbar, tabar);
        TOutput dilog2, dilog3;

        // deal with real roots
        TScale ieps2, ieps1;
        TMass rat1;
        TOutput ln43p, ln43m, rat2p, rat2m, zrat1;
        if (ql::iszero<TOutput, TMass, TScale>(ql::Imag(root))) {
            ln43p = ql::Lnrat<TOutput, TMass, TScale>(x43p, x43pm1);
            ln43m = ql::Lnrat<TOutput, TMass, TScale>(x43m, x43mm1);
            dilog2 = ql::Li2omx2<TOutput, TMass, TScale>(m4sqbar, x43p, sibar, x43pm1);
            dilog3 = ql::Li2omx2<TOutput, TMass, TScale>(m4sqbar, x43m, sibar, x43mm1);
        } else {
            ql::ratreal<TOutput, TMass, TScale>(m4sqbar, sibar, rat1, ieps1);
            ql::ratgam<TOutput, TMass, TScale>(rat2p, rat2m, ieps2, p3sq, m4sq, m3sq);
            zrat1 = TOutput(rat1);
            ln43p = ql::cLn<TOutput, TMass, TScale>(rat2p, ieps2);
            ln43m = ql::cLn<TOutput, TMass, TScale>(rat2m, ieps2);

            dilog2 = ql::spencer<TOutput, TMass, TScale>(zrat1, rat2p, ieps1, ieps2);
            dilog3 = ql::spencer<TOutput, TMass, TScale>(zrat1, rat2m, ieps1, ieps2);
        }

        res(i,2) = ql::Constants<TOutput>::_half();
        res(i,1) = -wlog;
        res(i,0) = -TOutput(ql::Constants<TScale>::template _pi2o12<TOutput, TMass, TScale>())
            + ql::Constants<TOutput>::_two() * wlogsmu * wlogtmu - wlog4mu * wlog4mu
            + (wlog4mu - wlogsmu) * ql::kLog(m4sq / m3sq) - ql::Constants<TOutput>::_half() * (ln43p * ln43p + ln43m * ln43m)
            - ql::Constants<TOutput>::_two() * dilog1 - dilog2 - dilog3;

        for (size_t j = 0; j < 3; j++)
            res(i,j) /= fac;
    }

}
