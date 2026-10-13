//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//          Stefan Hoeche   : shoeche@fnal.gov
//          Max Knobbe      : mknobbe@fnal.gov
//
// Leaf integral: box with 0 internal masses, 2 opposite off-shell legs
// (massive == 0, offshell == 2, xpi_in[6] and xpi_in[7] nonzero). Pure
// computation on pre-built Y; no classification logic. See
// box/dispatch/B0m.h for the caller.

#pragma once

#include "kokkosUtils.h"


namespace ql
{

    /*!
    * The integral is defined as:
    * \f[
    * I_4^{D=4-2\epsilon}(0,0,p_3^2,p_4^2;s_{12},s_{23};0,0,0,0) = \frac{\mu^{2\epsilon}}{s_{12}s_{23}} \\
    *     \left[ \frac{2}{\epsilon^2}\left( (-s_{12})^{-\epsilon} + (-s_{23})^{-\epsilon}-(-p_3^2)^{-\epsilon}-(-p_4^2)^{-\epsilon} \right) + \frac{1}{\epsilon^2} \left( (-p_3^2)^{-\epsilon}(-p_4)^{-\epsilon} \right) / (-s_{12})^{-\epsilon} \\
    * - 2 {\rm Li}_2 \left( 1-\frac{p_3^2}{s_{23}} \right) - 2 {\rm Li}_2 \left( 1-\frac{p_4^2}{s_{23}} \right) -  \ln^2 \left( \frac{-s_{12}}{-s_{23}} \right) \right] + \mathcal{O}(\epsilon)
    * \f]
    * Implementation of the formulae from Bern et al. \cite Bern:1993kr.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B4(
        const Kokkos::View<TOutput* [3]>& res,
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass si = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass ta = ql::Constants<TMass>::_two() * Y[1][3];
        const TMass mp4sq = ql::Constants<TMass>::_two() * Y[0][3];
        const TMass mp3sq = ql::Constants<TMass>::_two() * Y[2][3];
        const TOutput fac = ql::Constants<TOutput>::_one() / (si * ta);

        const TOutput ln_ta_mu2 = ql::Lnrat<TOutput, TMass, TScale>(ta, mu2);
        const TOutput ln_si_mu2 = ql::Lnrat<TOutput, TMass, TScale>(si, mu2);
        const TOutput ln_mp3_mu2 = ql::Lnrat<TOutput, TMass, TScale>(mp3sq, mu2);
        const TOutput ln_mp4_mu2 = ql::Lnrat<TOutput, TMass, TScale>(mp4sq, mu2);
        const TOutput ln_si_ta = ql::Lnrat<TOutput, TMass, TScale>(si, ta);
        const TOutput ln_si_mp3 = ql::Lnrat<TOutput, TMass, TScale>(si, mp3sq);

        res(i,2) = fac;
        res(i,1) = -fac * (ln_si_mp3 + ql::Lnrat<TOutput, TMass, TScale>(ta, mp4sq) + ln_ta_mu2);
        res(i,0) = fac * (ln_ta_mu2 * ln_ta_mu2
                    + ql::Constants<TOutput>::_half() * ln_si_mu2 * ln_si_mu2
                    - ql::Constants<TOutput>::_half() * ln_mp3_mu2 * ln_mp3_mu2
                    - ql::Constants<TOutput>::_half() * ln_mp4_mu2 * ln_mp4_mu2
                    + ql::Constants<TOutput>::_two() * (-ql::Li2omrat<TOutput, TMass, TScale>(mp3sq, ta) - ql::Li2omrat<TOutput, TMass, TScale>(mp4sq, ta) +
                                    ql::Constants<TOutput>::_half() * (ln_si_mp3 * ql::Lnrat<TOutput, TMass, TScale>(si, mp4sq) - ln_si_ta * ln_si_ta)));
    }

}
