//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//          Stefan Hoeche   : shoeche@fnal.gov
//          Max Knobbe      : mknobbe@fnal.gov
//
// Leaf integral: box with 0 internal masses, 2 adjacent off-shell legs
// (massive == 0, offshell == 2, xpi_in[5] nonzero). Pure computation on
// pre-built Y; no classification logic. See box/dispatch/B0m.h for the
// caller.

#pragma once

#include "kokkosUtils.h"


namespace ql
{

    /*!
    * The integral is defined as:
    * \f[
    * I_4^{D=4-2\epsilon}(0,p_2^2,0,p_4^2;s_{12},s_{23};0,0,0,0) = \frac{\mu^{2\epsilon}}{s_{12}s_{23}-p_4^2 p_2^2} \\
    *    \left[ \frac{2}{\epsilon^2}\left( (-s_{12})^{-\epsilon} + (-s_{23})^{-\epsilon}-(-p_2^2)^{-\epsilon}-(-p_4^2)^{-\epsilon} \right) \\
    * - 2 {\rm Li}_2 \left( 1-\frac{p_2^2}{s_{12}} \right) - 2 {\rm Li}_2 \left( 1-\frac{p_2^2}{s_{23}} \right) - 2 {\rm Li}_2 \left( 1-\frac{p_4^2}{s_{12}} \right) - 2 {\rm Li}_2 \left( 1-\frac{p_4^2}{s_{23}} \right) \\
    * + 2 {\rm Li}_2 \left( 1-\frac{p_4^2 p_2^2}{s_{23}s_{12}} \right) -  \ln^2 \left( \frac{-s_{12}}{-s_{23}} \right) \right] + \mathcal{O}(\epsilon)
    * \f]
    * Implementation of the formulae from Bern et al. \cite Bern:1993kr.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B3(
        const Kokkos::View<TOutput* [3]>& res,
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass si = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass ta = ql::Constants<TMass>::_two() * Y[1][3];
        const TMass mp4sq = ql::Constants<TMass>::_two() * Y[0][3];
        const TMass mp2sq = ql::Constants<TMass>::_two() * Y[1][2];
        const TMass r = ql::Constants<TMass>::_one() - mp2sq * mp4sq / (si * ta);
        const TScale SignRealsi = ql::Sign(ql::Real(si));
        const TScale SignRealmp2sq = ql::Sign(ql::Real(mp2sq));

        // Use expansion only in cases where signs are not ++-- or --++
        const bool landau = ((SignRealsi == ql::Sign(ql::Real(ta))) &&
                            (SignRealmp2sq == ql::Sign(ql::Real(mp4sq))) &&
                            (SignRealsi != SignRealmp2sq));

        if (ql::kAbs(r) < ql::Constants<TScale>::_eps() && landau == false) {
            // Expanded case
            const TOutput fac = ql::Constants<TOutput>::_one() / (si * ta);
            const TOutput ln_si_mu2 = ql::Lnrat<TOutput, TMass, TScale>(si, mu2);
            const TOutput ln_ta_mp4 = ql::Lnrat<TOutput, TMass, TScale>(ta, mp4sq);
            const TOutput l0_mp4_ta = ql::L0<TOutput, TMass, TScale>(mp4sq, ta);
            const TOutput l0_mp4_si = ql::L0<TOutput, TMass, TScale>(mp4sq, si);
            const TOutput l1_mp4_ta = ql::L1<TOutput, TMass, TScale>(mp4sq, ta);
            const TOutput l1_mp4_si = ql::L1<TOutput, TMass, TScale>(mp4sq, si);

            res(i,2) = ql::Constants<TOutput>::_zero();
            res(i,1) = -(ql::Constants<TOutput>::_two() + r) * fac;
            res(i,0) = fac * ( ql::Constants<TOutput>::_two() - ql::Constants<TOutput>::_half() * r + (ql::Constants<TOutput>::_two() + r) * (ln_si_mu2 + ln_ta_mp4)
                        + ql::Constants<TOutput>::_two() * (l0_mp4_ta + l0_mp4_si) + r * (l1_mp4_ta + l1_mp4_si));
        } else {
            // General case
            const TOutput fac = ql::Constants<TOutput>::_one() / (si * ta - mp2sq * mp4sq);
            const TOutput ln_mp2_mu2 = ql::Lnrat<TOutput, TMass, TScale>(mp2sq, mu2);
            const TOutput ln_mp4_mu2 = ql::Lnrat<TOutput, TMass, TScale>(mp4sq, mu2);
            const TOutput ln_si_mu2 = ql::Lnrat<TOutput, TMass, TScale>(si, mu2);
            const TOutput ln_ta_mu2 = ql::Lnrat<TOutput, TMass, TScale>(ta, mu2);
            const TOutput ln_s_t = ql::Lnrat<TOutput, TMass, TScale>(si, ta);
            const TOutput li2_1 = ql::Li2omrat<TOutput, TMass, TScale>(mp2sq, si);
            const TOutput li2_2 = ql::Li2omrat<TOutput, TMass, TScale>(mp2sq, ta);
            const TOutput li2_3 = ql::Li2omrat<TOutput, TMass, TScale>(mp4sq, si);
            const TOutput li2_4 = ql::Li2omrat<TOutput, TMass, TScale>(mp4sq, ta);
            const TOutput li2_5 = ql::Li2omx2<TOutput, TMass, TScale>(mp2sq, mp4sq, si, ta);

            res(i,2) = ql::Constants<TOutput>::_zero();
            res(i,1) = fac * ql::Constants<TOutput>::_two() * (ql::Lnrat<TOutput, TMass, TScale>(mp2sq, si) + ql::Lnrat<TOutput, TMass, TScale>(mp4sq, ta));
            res(i,0) = fac * (ln_si_mu2 * ln_si_mu2 + ln_ta_mu2 * ln_ta_mu2
                        - ln_mp2_mu2 * ln_mp2_mu2 - ln_mp4_mu2 * ln_mp4_mu2
                        + ql::Constants<TOutput>::_two() * (li2_5 - li2_1 - li2_2 - li2_3 - li2_4 - ql::Constants<TOutput>::_half() * ln_s_t * ln_s_t));
        }
    }

}
