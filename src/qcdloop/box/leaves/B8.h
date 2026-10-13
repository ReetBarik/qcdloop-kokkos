//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//          Stefan Hoeche   : shoeche@fnal.gov
//          Max Knobbe      : mknobbe@fnal.gov
//
// Leaf integral: box with 1 internal mass (massive == 1), Y[0][1] ==
// Y[1][2] == 0. Pure computation on pre-built Y; no classification
// logic. See box/dispatch/B1m.h for the caller.

#pragma once

#include "kokkosUtils.h"


namespace ql
{

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B8(
        const Kokkos::View<TOutput* [3]>& res,
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass msq = Y[3][3];
        const TMass tabar = ql::Constants<TMass>::_two() * Y[1][3];
        const TMass si = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass p3sqbar = ql::Constants<TMass>::_two() * Y[2][3];
        const TMass p4sqbar = ql::Constants<TMass>::_two() * Y[0][3];
        const TOutput wlogs = ql::Lnrat<TOutput, TMass, TScale>(si, mu2);
        const TOutput wlogp3 = ql::Lnrat<TOutput, TMass, TScale>(p3sqbar, tabar);
        const TOutput wlogp4 = ql::Lnrat<TOutput, TMass, TScale>(p4sqbar, tabar);

        const TOutput dilog3 = ql::Li2omrat<TOutput, TMass, TScale>(p3sqbar, tabar);
        const TOutput dilog4 = ql::Li2omrat<TOutput, TMass, TScale>(p4sqbar, tabar);
        const TOutput dilog34 = ql::Li2omx2<TOutput, TMass, TScale>(p3sqbar, p4sqbar, si, msq);
        const TOutput ln_si_mu2 = ql::Lnrat<TOutput, TMass, TScale>(si, mu2);

        res(i,2) = ql::Constants<TOutput>::_one();
        res(i,1) = wlogp3 + wlogp4 - wlogs;
        res(i,0) = -ql::Constants<TOutput>::_two() * dilog3 - ql::Constants<TOutput>::_two() * dilog4 - dilog34
                - TOutput(ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>()) + ql::Constants<TOutput>::_half() * (ln_si_mu2 * ln_si_mu2 - ql::kPow<TOutput, TMass, TScale>(ql::Lnrat<TOutput, TMass, TScale>(si, msq), 2))
                + ql::Constants<TOutput>::_two() * ln_si_mu2 * ql::Lnrat<TOutput, TMass, TScale>(tabar, msq)
                - ql::Lnrat<TOutput, TMass, TScale>(p3sqbar, mu2) * ql::Lnrat<TOutput, TMass, TScale>(p3sqbar, msq)
                - ql::Lnrat<TOutput, TMass, TScale>(p4sqbar, mu2) * ql::Lnrat<TOutput, TMass, TScale>(p4sqbar, msq);

        const TOutput d = TOutput(si * tabar);
        for (size_t j = 0; j < 3; j++) {
            res(i,j) /= d;
        }
    }

}
