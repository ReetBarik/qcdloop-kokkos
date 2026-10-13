//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//          Stefan Hoeche   : shoeche@fnal.gov
//          Max Knobbe      : mknobbe@fnal.gov
//
// Leaf integral: box with 3 internal masses, Y[0][0] == Y[0][1] == Y[0][3] == 0.
// Pure computation on pre-built Y; no classification logic. See
// box/dispatch/B3m.h for the caller.

#pragma once

#include "kokkosUtils.h"


namespace ql
{

    /*!
    * Implementation of the formulae from Beenakker et al. \cite Beenakker:1988jr.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B16(
        const Kokkos::View<TOutput* [3]>& res,
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass m2sq = Y[1][1];
        const TMass m3sq = Y[2][2];
        const TMass m4sq = Y[3][3];
        const TMass tabar = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass si = ql::Constants<TMass>::_two() * Y[1][3] - Y[1][1] - Y[3][3];
        const TMass mp2sq = ql::Constants<TMass>::_two() * Y[1][2] - m3sq - m2sq;
        const TMass mp3sq = ql::Constants<TMass>::_two() * Y[2][3] - m3sq - m4sq;
        const TMass m2 = ql::kSqrt(m2sq);
        const TMass m3 = ql::kSqrt(m3sq);
        const TMass m4 = ql::kSqrt(m4sq);
        const TMass mean = ql::kSqrt(ql::Real(m3sq) * mu2);

        TScale ieps = ql::Constants<TScale>::_zero(), iep2 = ql::Constants<TScale>::_zero(), iep3 = ql::Constants<TScale>::_zero();
        Kokkos::Array<TOutput, 3> cxs;
        Kokkos::Array<TOutput, 3> cx2;
        Kokkos::Array<TOutput, 3> cx3;
        ql::kfn<TOutput, TMass, TScale>(cxs, ieps, -si, m2, m4);
        ql::kfn<TOutput, TMass, TScale>(cx2, iep2, -mp2sq, m2, m3);
        ql::kfn<TOutput, TMass, TScale>(cx3, iep3, -mp3sq, m3, m4);

        const TOutput xs = cxs[0];
        const TScale rexs = ql::Real(xs);
        const TScale imxs = ql::Imag(xs);

        TOutput fac;
        if (ql::iszero<TOutput, TMass, TScale>(rexs - ql::Constants<TScale>::_one()) && ql::iszero<TOutput, TMass, TScale>(imxs)) {
            fac = TOutput(-ql::Constants<TMass>::_half() / (m2 * m4 * tabar));
            res(i,2) = ql::Constants<TOutput>::_zero();
            res(i,1) = ql::Constants<TOutput>::_one();
            res(i,0) = ql::Constants<TOutput>::_two() * ql::Lnrat<TOutput, TMass, TScale>(mean, tabar) - ql::Constants<TOutput>::_two();

            if (ql::iszero<TOutput, TMass, TScale>(ql::Real(cx2[0] - cx3[0])) && ql::iszero<TOutput, TMass, TScale>(ql::Imag(cx2[0] - cx3[0]))
                && ql::iszero<TOutput, TMass, TScale>(ql::Real(cx2[0] - ql::Constants<TOutput>::_one())) && ql::iszero<TOutput, TMass, TScale>(ql::Imag(cx2[0])))
                res(i,0) += ql::Constants<TOutput>::_four();
            else if (ql::iszero<TOutput, TMass, TScale>(ql::Real(cx2[0] - cx3[0])) && ql::iszero<TOutput, TMass, TScale>(ql::Imag(cx2[0] - cx3[0])))
                res(i,0) += ql::Constants<TOutput>::_two() + ql::Constants<TOutput>::_two() * (cx2[0] * cx2[0] + ql::Constants<TOutput>::_one()) * ql::cLn<TOutput, TMass, TScale>(cx2[0], iep2) / (cx2[0] * cx2[0] - ql::Constants<TOutput>::_one());
            else {
                const TOutput cln_cx2_iep2 = ql::cLn<TOutput, TMass, TScale>(cx2[0], iep2);
                const TOutput cln_cx3_iep3 = ql::cLn<TOutput, TMass, TScale>(cx3[0], iep3);
                res(i,0) += -(ql::Constants<TOutput>::_one() + cx2[0] * cx3[0]) / (ql::Constants<TOutput>::_one() - cx2[0] * cx3[0])
                    * (cln_cx2_iep2 + cln_cx3_iep3)
                    -(ql::Constants<TOutput>::_one() + cx2[0] / cx3[0]) / (ql::Constants<TOutput>::_one() - cx2[0] / cx3[0])
                    *(cln_cx2_iep2 - cln_cx3_iep3);
            }
        } else {
            fac = TOutput(-ql::Constants<TMass>::_one() / (m2 * m4 * tabar)) * cxs[0] / (ql::Constants<TOutput>::_one() - cxs[0] * cxs[0]);
            const TOutput xlog = ql::cLn<TOutput, TMass, TScale>(xs, ieps);
            const TOutput cln_cx2_iep2 = ql::cLn<TOutput, TMass, TScale>(cx2[0], iep2);
            const TOutput cln_cx3_iep3 = ql::cLn<TOutput, TMass, TScale>(cx3[0], iep3);
            res(i,2) = ql::Constants<TOutput>::_zero();
            res(i,1) = -xlog;
            res(i,0) = -ql::Constants<TOutput>::_two() * xlog * ql::Lnrat<TOutput, TMass, TScale>(mean, tabar)
                + cln_cx2_iep2 * cln_cx2_iep2 + cln_cx3_iep3 * cln_cx3_iep3
                - ql::cLi2omx2<TOutput, TMass, TScale>(xs, xs, ieps, ieps)
                + ql::cLi2omx3<TOutput, TMass, TScale>(cxs[0], cx2[0], cx3[0], ieps, iep2, iep3)
                + ql::cLi2omx3<TOutput, TMass, TScale>(cxs[0], ql::Constants<TOutput>::_one() / cx2[0], ql::Constants<TOutput>::_one() / cx3[0], ieps, -iep2, -iep3)
                + ql::cLi2omx3<TOutput, TMass, TScale>(cxs[0], cx2[0], ql::Constants<TOutput>::_one() / cx3[0], ieps, iep2, -iep3)
                + ql::cLi2omx3<TOutput, TMass, TScale>(cxs[0], ql::Constants<TOutput>::_one() / cx2[0], cx3[0], ieps, -iep2, iep3);
        }

        for (size_t j = 0; j < 3; j++)
            res(i,j) *= TOutput(fac);

    }

}
