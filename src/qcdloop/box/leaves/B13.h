//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//          Stefan Hoeche   : shoeche@fnal.gov
//          Max Knobbe      : mknobbe@fnal.gov
//
// Leaf integral: box with 2 internal masses (adjacent), 1 zero Y entry.
// Pure computation on pre-built Y; no classification logic. See
// box/dispatch/B2m.h for the caller.

#pragma once

#include "kokkosUtils.h"


namespace ql
{

    /*!
    * Implementation of the formulae from Ellis et al. \cite Ellis:2007qk.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B13(
        const Kokkos::View<TOutput* [3]>& res,
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass m3sq = Y[2][2];
        const TMass m4sq = Y[3][3];
        const TMass sibar = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass tabar = ql::Constants<TMass>::_two() * Y[1][3];
        const TMass m4sqbar = ql::Constants<TMass>::_two() * Y[0][3];
        const TMass m3sqbar = ql::Constants<TMass>::_two() * Y[1][2];
        const TMass p3sq = -(ql::Constants<TMass>::_two() * Y[2][3] - Y[2][2] - Y[3][3]);

        const TOutput fac = TOutput(sibar * tabar - m3sqbar * m4sqbar);
        const TOutput wlogsmu = ql::Lnrat<TOutput, TMass, TScale>(sibar, TMass(mu2));
        const TOutput wlogtmu = ql::Lnrat<TOutput, TMass, TScale>(tabar, TMass(mu2));
        const TOutput wlog3mu = ql::Lnrat<TOutput, TMass, TScale>(m3sqbar, TMass(mu2));
        const TOutput wlog4mu = ql::Lnrat<TOutput, TMass, TScale>(m4sqbar, TMass(mu2));
        const TOutput dilog1 = ql::Li2omrat<TOutput, TMass, TScale>(m3sqbar, sibar);
        const TOutput dilog4 = ql::Li2omrat<TOutput, TMass, TScale>(m4sqbar, tabar);
        const TOutput dilog7 = ql::Li2omx2<TOutput, TMass, TScale>(m3sqbar, m4sqbar, sibar, tabar);

        TOutput root, ga34p, ga34pm1, ga34m, ga34mm1;
        TOutput ga43p, ga43pm1, ga43m, ga43mm1;
        TMass x34p, x34pm1, x34m, x34mm1;
        TMass x43p, x43pm1, x43m, x43mm1;
        if (ql::iszero<TOutput, TMass, TScale>(p3sq)) {
            root = ql::Constants<TOutput>::_one();
            x34p = -ql::Constants<TMass>::_one();
            x34pm1 = -ql::Constants<TMass>::_one();
            x34m = m4sq;
            x34mm1 = m3sq;

            x43p = m3sq;
            x43pm1 = m4sq;
            x43m = -ql::Constants<TMass>::_one();
            x43mm1 = -ql::Constants<TMass>::_one();
        } else {
            root = ql::kSqrt(TOutput(ql::kPow<TOutput, TMass, TScale>(p3sq-m3sq+m4sq, 2) - ql::Constants<TMass>::_four() * m4sq * p3sq));
            const auto g34   = ql::rootpair<TOutput, TMass, TScale>(TOutput(p3sq + m4sq - m3sq), root, TOutput(ql::Constants<TMass>::_four() * m4sq * p3sq));
            const auto g34m1 = ql::rootpair<TOutput, TMass, TScale>(TOutput(-p3sq + m4sq - m3sq), root, TOutput(ql::Constants<TMass>::_four() * m3sq * p3sq));
            const auto g43   = ql::rootpair<TOutput, TMass, TScale>(TOutput(p3sq + m3sq - m4sq), root, TOutput(ql::Constants<TMass>::_four() * m3sq * p3sq));
            const auto g43m1 = ql::rootpair<TOutput, TMass, TScale>(TOutput(-p3sq + m3sq - m4sq), root, TOutput(ql::Constants<TMass>::_four() * m4sq * p3sq));
            ga34p = g34[0]; ga34m = g34[1]; ga34pm1 = g34m1[0]; ga34mm1 = g34m1[1];
            ga43p = g43[0]; ga43m = g43[1]; ga43pm1 = g43m1[0]; ga43mm1 = g43m1[1];

            x34p = -ql::Real(ga34p);
            x34pm1 = -ql::Real(ga34pm1);
            x34m = ql::Real(ga34m);
            x34mm1 = ql::Real(ga34mm1);

            x43p = -ql::Real(ga43p);
            x43pm1 = -ql::Real(ga43pm1);
            x43m = ql::Real(ga43m);
            x43mm1 = ql::Real(ga43mm1);
        }

        TMass rat3t, rat4s;
        TOutput ln43p, ln43m, dilog2, dilog3, dilog5, dilog6;
        TOutput zrat3t, zrat4s, rat34p, rat34m, rat43p, rat43m;
        TScale ieps3t, ieps4s, ieps34, ieps43;

        if (ql::iszero<TOutput, TMass, TScale>(ql::Imag(root))) {
            ln43p = ql::Lnrat<TOutput, TMass, TScale>(x43p, x43pm1);
            ln43m = ql::Lnrat<TOutput, TMass, TScale>(x43m, x43mm1);

            dilog2 = ql::Li2omx2<TOutput, TMass, TScale>(m3sqbar, x34p, tabar, x34pm1);
            dilog3 = ql::Li2omx2<TOutput, TMass, TScale>(m3sqbar, x34m, tabar, x34mm1);
            dilog5 = ql::Li2omx2<TOutput, TMass, TScale>(m4sqbar, x43p, sibar, x43pm1);
            dilog6 = ql::Li2omx2<TOutput, TMass, TScale>(m4sqbar, x43m, sibar, x43mm1);
        } else {
            ql::ratreal<TOutput, TMass, TScale>(m3sqbar, tabar, rat3t, ieps3t);
            ql::ratreal<TOutput, TMass, TScale>(m4sqbar, sibar, rat4s, ieps4s);

            ql::ratgam<TOutput, TMass, TScale>(rat34p, rat34m, ieps34, p3sq, m3sq, m4sq);
            ql::ratgam<TOutput, TMass, TScale>(rat43p, rat43m, ieps43, p3sq, m4sq, m3sq);

            zrat3t = TOutput(rat3t);
            zrat4s = TOutput(rat4s);

            dilog2 = ql::spencer<TOutput, TMass, TScale>(zrat3t, rat34p, ieps3t, ieps34);
            dilog3 = ql::spencer<TOutput, TMass, TScale>(zrat3t, rat34m, ieps3t, ieps34);
            dilog5 = ql::spencer<TOutput, TMass, TScale>(zrat4s, rat43p, ieps4s, ieps43);
            dilog6 = ql::spencer<TOutput, TMass, TScale>(zrat4s, rat43m, ieps4s, ieps43);

            ln43p = ql::cLn<TOutput, TMass, TScale>(rat43p, ql::Constants<TScale>::_zero());
            ln43m = ql::cLn<TOutput, TMass, TScale>(rat43m, ql::Constants<TScale>::_zero());
        }


        res(i,2) = ql::Constants<TOutput>::_zero();
        res(i,1) = wlog3mu + wlog4mu - wlogsmu - wlogtmu;
        res(i,0) = -ql::Constants<TOutput>::_two() * dilog1 - dilog2 - dilog3
                 -ql::Constants<TOutput>::_two() * dilog4 - dilog5 - dilog6
                 +ql::Constants<TOutput>::_two() * dilog7
                 +ql::Constants<TOutput>::_two() * wlogsmu * wlogtmu - wlog3mu * wlog3mu - wlog4mu * wlog4mu
                 +(wlog3mu - wlogtmu) * ql::kLog(m3sq / mu2)
                 +(wlog4mu - wlogsmu) * ql::kLog(m4sq / mu2)
                 -ql::Constants<TOutput>::_half() * (ln43p * ln43p + ln43m * ln43m);

        for (size_t j = 0; j < 3; j++)
            res(i,j) /= fac;
    }

}
