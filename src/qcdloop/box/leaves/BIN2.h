//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//          Stefan Hoeche   : shoeche@fnal.gov
//          Max Knobbe      : mknobbe@fnal.gov
//
// Leaf integral: box with 2 internal masses, fully general (massive == 2,
// no vanishing Y entries). Pure computation on pre-built Y; no
// classification logic. See box/dispatch/B2m.h for the caller.

#pragma once

#include "kokkosUtils.h"


namespace ql
{

    /*!
    * Finite box with 2 non-zero masses. Formulae from \cite Denner:1991qq.
    * \param res output object res[0,1,2] the coefficients in the Laurent series, following the LoopTools implementation \cite Hahn:2006qw.
    * \param Y the modified Cayley matrix.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void BIN2(
        const Kokkos::View<TOutput* [3]>& res,
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        const int i) {

        const TMass m3 = Y[2][2];
        const TMass m4 = Y[3][3];
        const TMass m = ql::kSqrt(m3 * m4);

        const TMass k12 = ql::Constants<TMass>::_two() * Y[1][2] / m3;
        const TMass k13 = ql::Constants<TMass>::_two() * Y[0][2] / m3;
        const TMass k14 = ql::Constants<TMass>::_two() * Y[2][3] / m;
        const TMass k23 = ql::Constants<TMass>::_two() * Y[0][1] / m3;
        const TMass k24 = ql::Constants<TMass>::_two() * Y[1][3] / m;
        const TMass k34 = ql::Constants<TMass>::_two() * Y[0][3] / m;

        const TOutput k12c = TOutput(k12 - ql::Max(ql::kAbs(k12), TMass(ql::Constants<TMass>::_one())) * ql::Constants<TScale>::template _ieps50<TOutput, TMass, TScale>());
        const TOutput k13c = TOutput(k13 - ql::Max(ql::kAbs(k13), TMass(ql::Constants<TMass>::_one())) * ql::Constants<TScale>::template _ieps50<TOutput, TMass, TScale>());
        const TOutput k23c = TOutput(k23 - ql::Max(ql::kAbs(k23), TMass(ql::Constants<TMass>::_one())) * ql::Constants<TScale>::template _ieps50<TOutput, TMass, TScale>());
        const TOutput k24c = TOutput(k24 - ql::Max(ql::kAbs(k24), TMass(ql::Constants<TMass>::_one())) * ql::Constants<TScale>::template _ieps50<TOutput, TMass, TScale>()) / k12c;
        const TOutput k34c = TOutput(k34 - ql::Max(ql::kAbs(k34), TMass(ql::Constants<TMass>::_one())) * ql::Constants<TScale>::template _ieps50<TOutput, TMass, TScale>()) / k13c;

        TOutput r14 = ql::Constants<TOutput>::_half() * (TOutput(k14) + TOutput(ql::Sign(ql::Real(k14))) * ql::kSqrt(TOutput((k14 - ql::Constants<TMass>::_two()) * (k14 + ql::Constants<TMass>::_two()))));
        r14 *= (ql::Constants<TOutput>::_one() + ql::Constants<TOutput>::template _ieps50<TOutput, TMass, TScale>() * TOutput(ql::Sign(ql::Real(ql::Constants<TOutput>::_one() / r14 - r14))));

        const TMass a = k34 * k24 - k23;
        const TMass b = k13 * k24 + k12 * k34 - k14 * k23;
        const TOutput c = TOutput(k13 * k12) - TOutput(k23) * (ql::Constants<TOutput>::_one() - ql::Constants<TOutput>::template _ieps50<TOutput, TMass, TScale>());
        const TOutput disc = ql::kSqrt(TOutput(b * b) - TOutput(ql::Constants<TMass>::_four() * a) * c);

        Kokkos::Array<TOutput, 2> x4;
        x4[0] = ql::Constants<TOutput>::_half() * (TOutput(b) - disc) / TOutput(a);
        x4[1] = ql::Constants<TOutput>::_half() * (TOutput(b) + disc) / TOutput(a);
        if (ql::kAbs(x4[0]) > ql::kAbs(x4[1]))
            x4[1] = c / (TOutput(a) * x4[0]);
        else
            x4[0] = c / (TOutput(a) * x4[1]);

        const Kokkos::Array<TScale, 2> imzero = {ql::Constants<TScale>::_zero(), ql::Constants<TScale>::_zero()};
        res(i, 0) = (
            ql::xspence<TOutput, TMass, TScale>(x4, imzero, r14, ql::Constants<TScale>::_zero()) +
            ql::xspence<TOutput, TMass, TScale>(x4, imzero, ql::Constants<TOutput>::_one() / r14, ql::Constants<TScale>::_zero()) -
            ql::xspence<TOutput, TMass, TScale>(x4, imzero, k34c, ql::Constants<TScale>::_zero()) -
            ql::xspence<TOutput, TMass, TScale>(x4, imzero, k24c, ql::Constants<TScale>::_zero()) +
            (ql::kLog(x4[1]) - ql::kLog(x4[0])) * (ql::kLog(k12c) + ql::kLog(k13c) - ql::kLog(k23c))
        ) / (m3 * m * disc);

        res(i, 1) = res(i, 2) = ql::Constants<TOutput>::_zero();
    }

}
