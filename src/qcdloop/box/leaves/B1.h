//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//          Stefan Hoeche   : shoeche@fnal.gov
//          Max Knobbe      : mknobbe@fnal.gov
//
// Leaf integral: box with 0 internal masses, all 4 legs on-shell
// (massive == 0, offshell == 0). Pure computation on pre-built Y; no
// classification logic. See box/dispatch/B0m.h for the caller.

#pragma once

#include "kokkosUtils.h"


namespace ql
{

    /*!
    * The integral is defined as:
    * \f[
    * I_4^{D=4-2\epsilon}(0,0,0,0;s_{12},s_{23};0,0,0,0) = \frac{\mu^{2\epsilon}}{s_{12}s_{23}}\left[ \frac{2}{\epsilon^2}\left( (-s_{12}-i\epsilon)^{-\epsilon} + (-s_{23}-i \epsilon)^{-\epsilon} \right) - \ln^2 \left( \frac{-s_{12}-i\epsilon}{-s_{23}-i\epsilon} \right) - \pi^2 \right] + \mathcal{O}(\epsilon)
    * \f]
    * Implementation of the formulae from Bern et al. \cite Bern:1993kr.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B1(
        const Kokkos::View<TOutput* [3]>& res,
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass si = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass ta = ql::Constants<TMass>::_two() * Y[1][3];
        const TOutput fac = ql::Constants<TOutput>::_one() / (si * ta);

        const TOutput lnrat_tamu2 = ql::Lnrat<TOutput, TMass, TScale>(ta, mu2);
        const TOutput lnrat_simu2 = ql::Lnrat<TOutput, TMass, TScale>(si, mu2);
        const TOutput lnrat_tasi = ql::Lnrat<TOutput, TMass, TScale>(ta, si);

        res(i,2) = fac * ql::Constants<TOutput>::_two() * ql::Constants<TOutput>::_two();
        res(i,1) = fac * ql::Constants<TOutput>::_two() * (-lnrat_tamu2 - lnrat_simu2);
        res(i,0) = fac * (lnrat_tamu2 * lnrat_tamu2 + lnrat_simu2 * lnrat_simu2 - lnrat_tasi * lnrat_tasi - TOutput(ql::Constants<TScale>::_pi2()));
    }

}
