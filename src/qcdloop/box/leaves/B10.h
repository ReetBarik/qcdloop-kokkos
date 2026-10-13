//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//          Stefan Hoeche   : shoeche@fnal.gov
//          Max Knobbe      : mknobbe@fnal.gov
//
// Leaf integral: box with 1 internal mass (massive == 1), Y[0][1] == 0.
// Pure computation on pre-built Y; no classification logic. See
// box/dispatch/B1m.h for the caller.

#pragma once

#include "kokkosUtils.h"


namespace ql
{

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B10(
        const Kokkos::View<TOutput* [3]>& res,
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass msq = Y[3][3];
        const TMass si = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass tabar = ql::Constants<TMass>::_two() * Y[1][3];
        const TMass m4sqbar = ql::Constants<TMass>::_two() * Y[0][3];
        const TMass m3sqbar = ql::Constants<TMass>::_two() * Y[2][3];
        const TMass mp2sq = ql::Constants<TMass>::_two() * Y[1][2];
        const TMass mean = ql::kSqrt(mu2 * msq);

        const TOutput fac = TOutput(si * tabar - mp2sq * m4sqbar);
        const TOutput wlogsmu = ql::Lnrat<TOutput, TMass, TScale>(si, mu2);
        const TOutput wlogtmu = ql::Lnrat<TOutput, TMass, TScale>(tabar, mu2);
        const TOutput wlog2mu = ql::Lnrat<TOutput, TMass, TScale>(mp2sq, mu2);
        const TOutput wlog4mu = ql::Lnrat<TOutput, TMass, TScale>(m4sqbar, mu2);

        const TOutput dilog1 = ql::Li2omrat<TOutput, TMass, TScale>(mp2sq, si);
        const TOutput dilog2 = ql::Li2omrat<TOutput, TMass, TScale>(tabar, m4sqbar);
        const TOutput dilog3 = ql::Li2omx2<TOutput, TMass, TScale>(mp2sq, m4sqbar, si, tabar);
        const TOutput dilog4 = ql::Li2omx2<TOutput, TMass, TScale>(m3sqbar, tabar, mp2sq, msq);
        const TOutput dilog5 = ql::Li2omx2<TOutput, TMass, TScale>(m3sqbar, m4sqbar, si, msq);

        res(i,2) = ql::Constants<TOutput>::_zero();
        res(i,1) = wlog2mu + wlog4mu - wlogsmu - wlogtmu;
        res(i,0) = dilog4 - dilog5
            - ql::Constants<TOutput>::_two() * dilog1
            + ql::Constants<TOutput>::_two() * dilog2
            + ql::Constants<TOutput>::_two() * dilog3
            + ql::Constants<TOutput>::_two() * res(i,1) * ql::Lnrat<TOutput, TMass, TScale>(mean, tabar);

        for (size_t j = 0; j < 3; j++) {
            res(i,j) /= fac;
        }
    }

}
