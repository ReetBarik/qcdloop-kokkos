//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//          Stefan Hoeche   : shoeche@fnal.gov
//          Max Knobbe      : mknobbe@fnal.gov
//
// Leaf integral: box with 1 internal mass (massive == 1), Y[0][1] ==
// Y[1][2] == Y[2][3] == 0 (Y[3][0] != 0). Pure computation on pre-built
// Y; no classification logic. See box/dispatch/B1m.h for the caller.

#pragma once

#include "kokkosUtils.h"


namespace ql
{

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B7(
        const Kokkos::View<TOutput* [3]>& res,
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass tabar = ql::Constants<TMass>::_two() * Y[1][3];
        const TMass p4sqbar = ql::Constants<TMass>::_two() * Y[0][3];
        const TMass si = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass msq = Y[3][3];
        const TOutput wlogs = ql::Lnrat<TOutput, TMass, TScale>(si, msq);
        const TOutput wlogt = ql::Lnrat<TOutput, TMass, TScale>(tabar, msq);
        const TOutput wlogp = ql::Lnrat<TOutput, TMass, TScale>(p4sqbar, msq);
        const TOutput wlogm = ql::Lnrat<TOutput, TMass, TScale>(mu2, msq);

        res(i,2) = ql::Constants<TOutput>::_three() / ql::Constants<TOutput>::_two();
        res(i,1) = (ql::Constants<TOutput>::_three() / ql::Constants<TOutput>::_two()) * wlogm - ql::Constants<TOutput>::_two() * wlogt - wlogs + wlogp;
        res(i,0) = ql::Constants<TOutput>::_two() * wlogs * wlogt - wlogp * wlogp - TOutput(ql::Constants<TScale>::template _pi2o12<TOutput, TMass, TScale>() * ql::Constants<TScale>::_five())
                 + (ql::Constants<TOutput>::_three() / ql::Constants<TOutput>::_four()) * wlogm * wlogm + wlogm * (-ql::Constants<TOutput>::_two() * wlogt - wlogs + wlogp)
                 - ql::Constants<TOutput>::_two() * ql::Li2omrat<TOutput, TMass, TScale>(p4sqbar, tabar);

        const TOutput d = TOutput(si * tabar);
        for (size_t j = 0; j < 3; j++) {
            res(i,j) /= d;
        }
    }

}
