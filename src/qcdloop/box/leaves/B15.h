//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//          Stefan Hoeche   : shoeche@fnal.gov
//          Max Knobbe      : mknobbe@fnal.gov
//
// Leaf integral: box with 2 internal masses (opposite), partially
// degenerate case. Pure computation on pre-built Y; no classification
// logic. See box/dispatch/B2m.h for the caller.

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
    void B15(
        const Kokkos::View<TOutput* [3]>& res,
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass m2sq = Y[1][1];
        const TMass m4sq = Y[3][3];
        const TMass m2sqbar = ql::Constants<TMass>::_two() * Y[1][2];
        const TMass m4sqbar = ql::Constants<TMass>::_two() * Y[2][3];
        const TMass si = ql::Constants<TMass>::_two() * Y[1][3] - Y[1][1] - Y[3][3];
        const TMass ta = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass m2 = ql::kSqrt(m2sq);
        const TMass m4 = ql::kSqrt(m4sq);

        TScale ieps = ql::Constants<TScale>::_zero();
        Kokkos::Array<TOutput, 3> cxs;
        ql::kfn<TOutput, TMass, TScale>(cxs, ieps, -si, m2, m4);
        const TOutput xs = cxs[0];

        TOutput fac;
        if (ql::iszero<TOutput, TMass, TScale>(m2sqbar) && !ql::iszero<TOutput, TMass, TScale>(m4sqbar)) {
            TMass yi;
            TScale iepyi;
            ql::ratreal<TOutput, TMass, TScale>(m4 * m2sqbar, m2 * m4sqbar, yi, iepyi);
            TOutput cyi = TOutput(yi);
            fac = xs / (ql::Constants<TOutput>::_one() - xs * xs) / TOutput(-m2 * m4 * ta);
            const TOutput xlog = ql::cLn<TOutput, TMass, TScale>(xs, ieps);
            res(i,2) = ql::Constants<TOutput>::_zero();
            res(i,1) = -xlog;
            res(i,0) = xlog * (-xlog - TOutput(ql::kLog(mu2 / m4sq))
                -ql::Constants<TOutput>::_two() * ql::Lnrat<TOutput, TMass, TScale>(m4sqbar, ta))
                -ql::cLi2omx2<TOutput, TMass, TScale>(xs, xs, ieps, ieps)
                +ql::cLi2omx2<TOutput, TMass, TScale>(xs, cyi, ieps, iepyi)
                -ql::cLi2omx2<TOutput, TMass, TScale>(ql::Constants<TOutput>::_one() / xs, cyi, -ieps, iepyi);

            for (size_t j = 0; j < 3; j++)
                res(i,j) *= TOutput(fac);

            return;
        } else if (ql::iszero<TOutput, TMass, TScale>(m4sqbar) && !ql::iszero<TOutput, TMass, TScale>(m2sqbar)) {
            TMass yy;
            TScale iepsyy;
            ql::ratreal<TOutput, TMass, TScale>(m2 * m4sqbar, m4 * m2sqbar, yy, iepsyy);
            TOutput cyy = TOutput(yy);
            fac = xs / (ql::Constants<TOutput>::_one() - xs * xs) / TOutput(-m2 * m4 * ta);
            const TOutput xlog = ql::cLn<TOutput, TMass, TScale>(xs, ieps);
            res(i,2) = ql::Constants<TOutput>::_zero();
            res(i,1) = -xlog;
            res(i,0) = xlog * (-xlog - TOutput(ql::kLog(mu2 / m2sq))
                      -ql::Constants<TOutput>::_two() * ql::Lnrat<TOutput, TMass, TScale>(m2sqbar, ta))
                -ql::cLi2omx2<TOutput, TMass, TScale>(xs, xs, ieps, ieps)
                +ql::cLi2omx2<TOutput, TMass, TScale>(xs, cyy, ieps, iepsyy)
                -ql::cLi2omx2<TOutput, TMass, TScale>(ql::Constants<TOutput>::_one() / xs, cyy, -ieps, iepsyy);

            for (size_t j = 0; j < 3; j++)
                res(i,j) *= TOutput(fac);

            return;
        } else if (ql::iszero<TOutput, TMass, TScale>(m4sqbar) && ql::iszero<TOutput, TMass, TScale>(m2sqbar)) {
            Kokkos::printf("Box::B15 wrong kinematics, this is really B14.");
        }
        TMass yy;
        TScale iepsyy;
        ql::ratreal<TOutput, TMass, TScale>(m2*m4sqbar, m4*m2sqbar, yy, iepsyy);
        const TScale rexs = ql::Real(xs);
        const TScale imxs = ql::Imag(xs);

        if (ql::iszero<TOutput, TMass, TScale>(rexs-ql::Constants<TScale>::_one()) && ql::iszero<TOutput, TMass, TScale>(imxs)) {
            fac = TOutput(ql::Constants<TOutput>::_half() / (m2 * m4 * ta));
            res(i,2) = ql::Constants<TOutput>::_zero();
            res(i,1) = ql::Constants<TOutput>::_one();
            res(i,0) = TOutput(ql::kLog(mu2 / (m2 * m4)))
                      -ql::Lnrat<TOutput, TMass, TScale>(m2sqbar, ta) - ql::Lnrat<TOutput, TMass, TScale>(m4sqbar, ta) - ql::Constants<TOutput>::_two()
                     -TOutput((ql::Constants<TMass>::_one() + yy) / (ql::Constants<TMass>::_one() - yy)) * ql::Lnrat<TOutput, TMass, TScale>(m2 * m4sqbar, m4 * m2sqbar);
        } else {
            fac = xs / (ql::Constants<TOutput>::_one() - xs * xs) / TOutput(-m2 * m4 * ta);
            const TOutput xlog = ql::cLn<TOutput, TMass, TScale>(xs, ieps);
            res(i,2) = ql::Constants<TOutput>::_zero();
            res(i,1) = -xlog;
            res(i,0) = xlog * (-ql::Constants<TOutput>::_half() * xlog - TOutput(ql::kLog(mu2 / (m2 * m4)))
                    -ql::Lnrat<TOutput, TMass, TScale>(m2sqbar, ta) - ql::Lnrat<TOutput, TMass, TScale>(m4sqbar, ta))
                    -ql::cLi2omx2<TOutput, TMass, TScale>(xs, xs, ieps, ieps)
                    +ql::Constants<TOutput>::_half() * ql::kPow<TOutput, TMass, TScale>(ql::Lnrat<TOutput, TMass, TScale>(m2 * m4sqbar, m4 * m2sqbar), 2)
                    +ql::cLi2omx2<TOutput, TMass, TScale>(xs, TOutput(yy), ieps, iepsyy)
                    +ql::cLi2omx2<TOutput, TMass, TScale>(xs, TOutput(ql::Constants<TMass>::_one() / yy), ieps, -iepsyy);
        }

        for (size_t j = 0; j < 3; j++)
            res(i,j) *= TOutput(fac);
    }

}
