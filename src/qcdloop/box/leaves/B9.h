//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//          Stefan Hoeche   : shoeche@fnal.gov
//          Max Knobbe      : mknobbe@fnal.gov
//
// Leaf integral: box with 1 internal mass (massive == 1), Y[0][1] ==
// Y[3][0] == 0. Pure computation on pre-built Y; no classification
// logic. See box/dispatch/B1m.h for the caller.

#pragma once

#include "kokkosUtils.h"


namespace ql
{

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B9(
        const Kokkos::View<TOutput* [3]>& res,
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass msq = Y[3][3];
        const TMass mean = ql::kSqrt(TMass(mu2 * msq));
        const TMass tabar = ql::Constants<TMass>::_two() * Y[1][3];
        const TMass si = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass m3sqbar = ql::Constants<TMass>::_two() * Y[2][3];
        const TMass mp2sq = ql::Constants<TMass>::_two() * Y[1][2];
        const TOutput fac = TOutput(si * tabar);

        const TOutput wlogt = ql::Lnrat<TOutput, TMass, TScale>(tabar, mean);
        const TOutput wlog2 = ql::Lnrat<TOutput, TMass, TScale>(si, mp2sq);

        const TOutput dilog1 = ql::Li2omx2<TOutput, TMass, TScale>(m3sqbar, tabar, mp2sq, msq);
        const TOutput dilog2 = ql::Li2omrat<TOutput, TMass, TScale>(si, mp2sq);

        res(i,2) = ql::Constants<TOutput>::_half();
        res(i,1) = -wlogt - wlog2;
        res(i,0) = dilog1 + ql::Constants<TOutput>::_two() * dilog2 + TOutput(ql::Constants<TScale>::template _pi2o12<TOutput, TMass, TScale>()) + (wlogt + wlog2) * (wlogt + wlog2);

        for (size_t j = 0; j < 3; j++) {
            res(i,j) /= fac;
        }
    }

}
