//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//          Stefan Hoeche   : shoeche@fnal.gov
//          Max Knobbe      : mknobbe@fnal.gov
//
// Leaf integral: box with 0 internal masses, 1 off-shell leg
// (massive == 0, offshell == 1). Pure computation on pre-built Y; no
// classification logic. See box/dispatch/B0m.h for the caller.

#pragma once

#include "kokkosUtils.h"
#include "box/leaves/_lnrat_cuda_fix.h"


namespace ql
{

    // A100 (sm_80) nvcc/ptxas silently corrupts TripleFloat results here
    // once ql::BO()'s full dispatch kernel is compiled in (register
    // pressure from the other 20 leaves, not this one's own formula --
    // see box/leaves/_lnrat_cuda_fix.h for the full root-cause writeup).
    // ql::lnrat_fix + volatile_roundtrip_fix are no-ops on every backend
    // except CUDA device code.

    /*!
    * The integral is defined as:
    * \f[
    * I_4^{D=4-2\epsilon}(0,0,0,p_4^2;s_{12},s_{23};0,0,0,0) = \frac{\mu^{2\epsilon}}{s_{12}s_{23}}\left[ \frac{2}{\epsilon^2}\left( (-s_{12})^{-\epsilon} + (-s_{23})^{-\epsilon} -(-p_4^2)^{-\epsilon} \right) - 2 {\rm Li}_2 \left( 1-\frac{p_4^2}{s_{12}} \right) - 2 {\rm Li}_2 \left( 1-\frac{p_4^2}{s_{23}} \right) -  \ln^2 \left( \frac{-s_{12}}{-s_{23}} \right) - \frac{\pi^2}{3} \right] + \mathcal{O}(\epsilon)
    * \f]
    * Implementation of the formulae from Bern et al. \cite Bern:1993kr.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B2(
        const Kokkos::View<TOutput* [3]>& res,
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass si = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass ta = ql::Constants<TMass>::_two() * Y[1][3];
        const TMass mp4sq = ql::Constants<TMass>::_two() * Y[0][3];
        const TOutput fac = ql::Constants<TOutput>::_one() / (si * ta);

        const TOutput ln_mp4_mu2 = ql::volatile_roundtrip_fix(ql::lnrat_fix<TOutput, TMass, TScale>(mp4sq, mu2));
        const TOutput ln_ta_mu2 = ql::volatile_roundtrip_fix(ql::lnrat_fix<TOutput, TMass, TScale>(ta, mu2));
        const TOutput ln_si_mu2 = ql::volatile_roundtrip_fix(ql::lnrat_fix<TOutput, TMass, TScale>(si, mu2));
        const TOutput ln_mp4_ta = ql::volatile_roundtrip_fix(ql::lnrat_fix<TOutput, TMass, TScale>(mp4sq, ta));
        const TOutput ln_mp4_si = ql::volatile_roundtrip_fix(ql::lnrat_fix<TOutput, TMass, TScale>(mp4sq, si));
        const TOutput ln_ta_si = ql::volatile_roundtrip_fix(ql::lnrat_fix<TOutput, TMass, TScale>(ta, si));

        res(i,2) = fac * ql::Constants<TOutput>::_two();
        res(i,1) = res(i,2) * (ln_mp4_mu2 - ln_ta_mu2 - ln_si_mu2);
        res(i,0) = fac * (-ln_mp4_mu2 * ln_mp4_mu2 + ln_ta_mu2 * ln_ta_mu2 + ln_si_mu2 * ln_si_mu2
                    + ql::Constants<TOutput>::_two() * (ql::Li2omrat<TOutput, TMass, TScale>(ta, mp4sq) + ql::Li2omrat<TOutput, TMass, TScale>(si, mp4sq) - TOutput(ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>()))
                    + ln_mp4_ta * ln_mp4_ta + ln_mp4_si * ln_mp4_si - ln_ta_si * ln_ta_si);
    }

}
