//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//          Stefan Hoeche   : shoeche@fnal.gov
//          Max Knobbe      : mknobbe@fnal.gov
//
// Leaf integral: box with 2 internal masses (opposite), fully degenerate
// case. Pure computation on pre-built Y; no classification logic. See
// box/dispatch/B2m.h for the caller.

#pragma once

#include "kokkosUtils.h"


namespace ql
{

    /*!
    * Implementation of the formulae from Beenakker et al. \cite Beenakker:1988jr.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B14(
        const Kokkos::View<TOutput* [3]>& res,
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass m2sq = Y[1][1];
        const TMass m4sq = Y[3][3];
        const TMass ta = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass si = ql::Constants<TMass>::_two() * Y[1][3] - Y[1][1] - Y[3][3];
        const TMass m2 = ql::kSqrt(m2sq);
        const TMass m4 = ql::kSqrt(m4sq);

        const TOutput wlogtmu = ql::Lnrat<TOutput, TMass, TScale>(mu2, ta);

        TScale ieps = ql::Constants<TScale>::_zero();
        Kokkos::Array<TOutput, 3> cxs;
        ql::kfn<TOutput, TMass, TScale>(cxs, ieps, -si, m2, m4);
        const TScale xs = ql::Real(cxs[0]);
        const TScale imxs = ql::Imag(cxs[0]);

        TOutput fac;
        if ( ql::iszero<TOutput, TMass, TScale>(xs - ql::Constants<TScale>::_one()) && ql::iszero<TOutput, TMass, TScale>(imxs))
            fac = TOutput(-xs / (m2 * m4 * ta));
        else {
            const TOutput xlog = ql::cLn<TOutput, TMass, TScale>(cxs[0], ieps);
            fac = TOutput(ql::Constants<TMass>::_two() / (m2 * m4 * ta)) * cxs[0] / (cxs[1] * cxs[2]) * xlog;
        }
        res(i,2) = ql::Constants<TOutput>::_zero();
        res(i,1) = fac;
        res(i,0) = fac * wlogtmu;
    }

}
