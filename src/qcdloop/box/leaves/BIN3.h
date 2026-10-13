//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//          Stefan Hoeche   : shoeche@fnal.gov
//          Max Knobbe      : mknobbe@fnal.gov
//
// Leaf integral: box with 3 internal masses, fully general (massive == 3).
// Pure computation on pre-built Y; no classification logic. See
// box/dispatch/B3m.h for the caller.

#pragma once

#include "kokkosUtils.h"


namespace ql
{

    /*!
    * Finite box with 3 non-zero masses. Formulae from \cite Denner:1991qq.
    * \param res output object res[0,1,2] the coefficients in the Laurent series, following the LoopTools implementation \cite Hahn:2006qw.
    * \param Y the modified Cayley matrix.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void BIN3(
        const Kokkos::View<TOutput* [3]>& res,
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        const int i) {

        const TMass m2 = Y[1][1];
        const TMass m3 = Y[2][2];
        const TMass m4 = Y[3][3];
        const TMass m_0 = ql::kSqrt(m3 * m4);
        const TMass m_1 = ql::kSqrt(m2 * m3);
        const TMass m_2 = ql::kSqrt(m2 * m4);

        const TMass k12 = ql::Constants<TMass>::_two() * Y[2][3] / m_0;
        const TMass k13 = ql::Constants<TMass>::_two() * Y[0][2] / m3;
        const TMass k14 = ql::Constants<TMass>::_two() * Y[1][2] / m_1;
        const TMass k23 = ql::Constants<TMass>::_two() * Y[0][3] / m_0;
        const TMass k24 = ql::Constants<TMass>::_two() * Y[1][3] / m_2;
        const TMass k34 = ql::Constants<TMass>::_two() * Y[0][1] / m_1;

        int ir12 = 0, ir14 = 0, ir24 = 0;
        const TOutput r12 = ql::Constants<TOutput>::_half() * (TOutput(k12) + TOutput(ql::Sign(ql::Real(k12))) * ql::kSqrt(TOutput((k12 - ql::Constants<TMass>::_two()) * (k12 + ql::Constants<TMass>::_two()))));
        const TOutput r14 = ql::Constants<TOutput>::_half() * (TOutput(k14) + TOutput(ql::Sign(ql::Real(k14))) * ql::kSqrt(TOutput((k14 - ql::Constants<TMass>::_two()) * (k14 + ql::Constants<TMass>::_two()))));
        const TOutput r24 = ql::Constants<TOutput>::_half() * (TOutput(k24) + TOutput(ql::Sign(ql::Real(k24))) * ql::kSqrt(TOutput((k24 - ql::Constants<TMass>::_two()) * (k24 + ql::Constants<TMass>::_two()))));
        if (ql::Real(k12) < -ql::Constants<TMass>::_two()) ir12 = 10 * ql::SignInt(ql::Constants<TScale>::_one() - ql::kAbs(r12));
        if (ql::Real(k14) < -ql::Constants<TMass>::_two()) ir14 = 10 * ql::SignInt(ql::Constants<TScale>::_one() - ql::kAbs(r14));
        if (ql::Real(k24) < -ql::Constants<TMass>::_two()) ir24 = 10 * ql::SignInt(ql::Constants<TScale>::_one() - ql::kAbs(r24));

        const TOutput q24 = r24 - ql::Constants<TOutput>::_one() / r24;
        const TOutput q12 = TOutput(k12) - r24 * TOutput(k14);

        const TOutput a = TOutput(k34) / r24 - TOutput(k23);
        const TOutput b = TOutput(k12 * k34) - TOutput(k13) * q24 - TOutput(k14 * k23);
        const TOutput c = TOutput(k13) * q12 + r24 * TOutput(k34) - TOutput(k23);
        const TOutput d = TOutput((k12 * k34 - k13 * k24 - k14 * k23) * (k12 * k34 - k13 * k24 - k14 * k23)) -
                        TOutput(ql::Constants<TScale>::_four()) * TOutput(k13 * (k13 - k23 * (k12 - k14 * k24)) +
                                        k23 * (k23 - k24 * k34) + k34 * (k34 - k13 * k14));
        const TOutput discr = ql::kSqrt(d);

        Kokkos::Array<TOutput, 2> x4;
        Kokkos::Array<TOutput, 2> x1;
        Kokkos::Array<TOutput, 2> l4;
        x4[0] = ql::Constants<TOutput>::_half() * (b - discr) / a;
        x4[1] = ql::Constants<TOutput>::_half() * (b + discr) / a;
        if (ql::kAbs(x4[0]) > ql::kAbs(x4[1]))
            x4[1] = c / (a * x4[0]);
        else
            x4[0] = c / (a * x4[1]);

        const TOutput dd = -TOutput(k34) * r24 + TOutput(k23);
        Kokkos::Array<TScale, 2> ix4;
        Kokkos::Array<TScale, 2> ix1;
        ix4[0] = ql::Sign(ql::Real(dd));
        ix4[1] = -ix4[0];

        x1[0] = x4[0] / r24;
        x1[1] = x4[1] / r24;
        ix1[0] = ql::Sign(ix4[0] * ql::Real(r24));
        ix1[1] = -ix1[0];

        const TOutput cc = ql::cLn<TOutput, TMass, TScale>(ql::Real(k13), -ql::Constants<TScale>::_one());
        l4[0] = cc + ql::cLn<TOutput, TMass, TScale>((q12 + q24 * x4[0]) / dd, ql::Real(q24 * ix4[0] / dd));
        l4[1] = cc + ql::cLn<TOutput, TMass, TScale>((q12 + q24 * x4[1]) / dd, ql::Real(q24 * ix4[1] / dd));

        res(i, 2) = res(i, 1) = ql::Constants<TOutput>::_zero();
        res(i, 0) = (
            ql::xspence<TOutput, TMass, TScale>(x4, ix4, r14, ql::from_int<TScale>(ir14)) +
            ql::xspence<TOutput, TMass, TScale>(x4, ix4, ql::Constants<TOutput>::_one() / r14, ql::from_int<TScale>(-ir14)) -
            ql::xspence<TOutput, TMass, TScale>(x4, ix4, TOutput(k34 / k13), -ql::Real(k13)) -
            ql::xspence<TOutput, TMass, TScale>(x1, ix1, r12, ql::from_int<TScale>(ir12)) -
            ql::xspence<TOutput, TMass, TScale>(x1, ix1, ql::Constants<TOutput>::_one() / r12, ql::from_int<TScale>(-ir12)) +
            ql::xspence<TOutput, TMass, TScale>(x1, ix1, TOutput(k23 / k13), -ql::Real(k13)) -
            TOutput{ql::Constants<TScale>::_zero(), ql::Constants<TScale>::_two() * ql::Constants<TScale>::_pi()} *
                ql::xetatilde<TOutput, TMass, TScale>(x4, ix4, ql::Constants<TOutput>::_one() / r24, ql::from_int<TScale>(-ir24), l4)
        ) / (TOutput(m3 * m_2) * discr);
    }

}
