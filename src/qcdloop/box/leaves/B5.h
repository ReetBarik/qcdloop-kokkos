//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//          Stefan Hoeche   : shoeche@fnal.gov
//          Max Knobbe      : mknobbe@fnal.gov
//
// Leaf integral: box with 0 internal masses, 3 off-shell legs
// (massive == 0, offshell == 3). Pure computation on pre-built Y; no
// classification logic. See box/dispatch/B0m.h for the caller.

#pragma once

#include "kokkosUtils.h"


namespace ql
{

    /*!
    * The integral is defined as:
    * \f[
    * I_4^{D=4-2\epsilon}(0,p_2^2,p_3^2,p_4^2;s_{12},s_{23};0,0,0,0) = \frac{\mu^{2\epsilon}}{s_{12}s_{23}-p_2^2 p_4^2} \\
    *     \left[ \frac{2}{\epsilon^2}\left( (-s_{12})^{-\epsilon} + (-s_{23})^{-\epsilon}-(-p_2^2)^{-\epsilon}-(-p_3^2)^{-\epsilon} -(-p_4^2)^{-\epsilon} \right) \\
    *  + \frac{1}{\epsilon^2} \left( (-p_2^2)^{-\epsilon}(-p_3)^{-\epsilon} \right) / (-s_{23})^{-\epsilon}+ \frac{1}{\epsilon^2} \left( (-p_3^2)^{-\epsilon}(-p_4)^{-\epsilon} \right) / (-s_{12})^{-\epsilon} \\
    * - 2 {\rm Li}_2 \left( 1-\frac{p_2^2}{s_{12}} \right) - 2 {\rm Li}_2 \left( 1-\frac{p_4^2}{s_{23}} \right) + 2 {\rm Li}_2 \left( 1-\frac{p_2^2 p_4^2}{s_{12} s_{23}} \right) -  \ln^2 \left( \frac{-s_{12}}{-s_{23}} \right) \right] + \mathcal{O}(\epsilon)
    * \f]
    * Implementation of the formulae from Bern et al. \cite Bern:1993kr.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B5(
        const Kokkos::View<TOutput* [3]>& res,
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass si = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass ta = ql::Constants<TMass>::_two() * Y[1][3];
        const TMass mp2sq = ql::Constants<TMass>::_two() * Y[1][2];
        const TMass mp3sq = ql::Constants<TMass>::_two() * Y[2][3];
        const TMass mp4sq = ql::Constants<TMass>::_two() * Y[0][3];
        const TMass r = ql::Constants<TMass>::_one() - mp2sq * mp4sq / (si * ta);
        const TScale SignRealsi = ql::Sign(ql::Real(si));
        const TScale SignRealmp2sq = ql::Sign(ql::Real(mp2sq));

        // Use expansion only in cases where signs are not ++-- or --++
        const bool landau = ((SignRealsi == ql::Sign(ql::Real(ta))) &&
                            (SignRealmp2sq == ql::Sign(ql::Real(mp4sq))) &&
                            (SignRealsi != SignRealmp2sq));

        if (ql::kAbs(r) < ql::Constants<TScale>::_eps() && landau == false) {
            const TOutput l0_mp4_ta = ql::L0<TOutput, TMass, TScale>(mp4sq, ta);
            const TOutput l1_mp4_ta = ql::L1<TOutput, TMass, TScale>(mp4sq, ta);

            // Expanded case
            res(i,2) = ql::Constants<TOutput>::_zero();
            res(i,1) = -(ql::Constants<TOutput>::_one() + ql::Constants<TOutput>::_half() * r) / (si * ta);
            res(i,0) = res(i,1) * (ql::Lnrat<TOutput, TMass, TScale>(mu2, si) +
                                ql::Lnrat<TOutput, TMass, TScale>(mp3sq, ta) - ql::Constants<TOutput>::_two() -
                                (ql::Constants<TOutput>::_one() + mp4sq / ta) * l0_mp4_ta) +
                    (r / (si * ta)) * (l1_mp4_ta - l0_mp4_ta - ql::Constants<TOutput>::_one());
        } else {
            // General case
            const TOutput fac = ql::Constants<TOutput>::_one() / (si * ta - mp2sq * mp4sq);
            const TOutput li2_1 = ql::Li2omrat<TOutput, TMass, TScale>(mp2sq, si);
            const TOutput li2_2 = ql::Li2omrat<TOutput, TMass, TScale>(mp4sq, ta);
            const TOutput li2_3 = ql::Li2omx2<TOutput, TMass, TScale>(mp2sq, mp4sq, si, ta);
            const TOutput ln_ta_mp2 = ql::Lnrat<TOutput, TMass, TScale>(ta, mp2sq);
            const TOutput ln_si_mp4 = ql::Lnrat<TOutput, TMass, TScale>(si, mp4sq);
            const TOutput ln_si_ta = ql::Lnrat<TOutput, TMass, TScale>(si, ta);

            res(i,2) = ql::Constants<TOutput>::_zero();
            res(i,1) = -ln_ta_mp2 - ln_si_mp4;
            res(i,0) = -ql::Constants<TOutput>::_half() * (ln_ta_mp2 * ln_ta_mp2 + ln_si_mp4 * ln_si_mp4) -
                    (ql::Lnrat<TOutput, TMass, TScale>(mp3sq, ta) +
                        ql::Lnrat<TOutput, TMass, TScale>(mu2, ta)) * ln_ta_mp2 -
                    (ql::Lnrat<TOutput, TMass, TScale>(mp3sq, si) +
                        ql::Lnrat<TOutput, TMass, TScale>(mu2, si)) * ln_si_mp4 -
                    ql::Constants<TOutput>::_two() * (li2_1 + li2_2 - li2_3) - ln_si_ta * ln_si_ta;

            res(i,1) *= fac;
            res(i,0) *= fac;
        }
    }

}
