//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//          Stefan Hoeche   : shoeche@fnal.gov
//          Max Knobbe      : mknobbe@fnal.gov
//
// Dispatch for box integrals with 0 internal masses: classifies the
// off-shell pattern, builds Y, and calls the matching leaf from
// box/leaves/. Contains the B0m dispatcher and (for selective
// compilation) a pruned BO() for massive == 0 only.

#pragma once

#include "box/classify.h"
#include "box/leaves/BIN0.h"
#include "box/leaves/B1.h"
#include "box/leaves/B2.h"
#include "box/leaves/B3.h"
#include "box/leaves/B4.h"
#include "box/leaves/B5.h"


namespace ql
{

    /*!
    * This function triggers the topologies with 0 internal masses.
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param xpi array with masses and momenta squared
    * \param mu2 is the square of the scale mu
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B0m(
        const Kokkos::View<TOutput* [3]>& res,
        const Kokkos::Array<TMass, 13>& xpi,
        const TScale& mu2,
        const int i) {

        const ClassifyB0m cls = ql::classify_offshell_b0m<TOutput, TMass, TScale>(xpi);
        const int offshell = cls.offshell, jsort0 = cls.jsort0, jsort1 = cls.jsort1, jsort2 = cls.jsort2;

        Kokkos::Array<TMass, 13> xpiout;
        bool swapped = true;
        const int jdiff = jsort2 - jsort1;

        if (offshell == 1) {
            for (size_t j = 0; j < 13; j++) {
                xpiout[ql::swap_b0m(j, jsort1 - 1)] = xpi[j];
            }
        } else if (offshell == 2 && (jdiff == 2 || jdiff == 1)) {
            for (size_t j = 0; j < 13; j++) {
                xpiout[ql::swap_b0m(j, jsort2 - 1)] = xpi[j];
            }
        } else if (offshell == 2 && jdiff == 3) {
            for (size_t j = 0; j < 13; j++) {
                xpiout[ql::swap_b0m(j, 0)] = xpi[j];
            }
        } else if (offshell == 3) {
            for (size_t j = 0; j < 13; j++) {
                xpiout[ql::swap_b0m(j, ql::jsort_b0m(jsort0))] = xpi[j];
            }
        } else {
            swapped = false;
        }

        Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Y;
        Kokkos::Array<TMass, 13> xpi_in;
        if (swapped)
            xpi_in = xpiout;
        else
            xpi_in = xpi;


        Y[0][0] = xpi_in[0];
        Y[1][1] = xpi_in[1];
        Y[2][2] = xpi_in[2];
        Y[3][3] = xpi_in[3];
        Y[0][1] = Y[1][0] = ql::Constants<TMass>::_half() * (xpi_in[0] + xpi_in[1] - xpi_in[4]);
        Y[0][2] = Y[2][0] = ql::Constants<TMass>::_half() * (xpi_in[0] + xpi_in[2] - xpi_in[8]);
        Y[0][3] = Y[3][0] = ql::Constants<TMass>::_half() * (xpi_in[0] + xpi_in[3] - xpi_in[7]);
        Y[1][2] = Y[2][1] = ql::Constants<TMass>::_half() * (xpi_in[1] + xpi_in[2] - xpi_in[5]);
        Y[1][3] = Y[3][1] = ql::Constants<TMass>::_half() * (xpi_in[1] + xpi_in[3] - xpi_in[9]);
        Y[2][3] = Y[3][2] = ql::Constants<TMass>::_half() * (xpi_in[2] + xpi_in[3] - xpi_in[6]);

        if (offshell == 4)
            ql::BIN0<TOutput, TMass, TScale>(res, Y, i);
        else if (offshell == 3)
            ql::B5<TOutput, TMass, TScale>(res, Y, mu2, i);
        else if (offshell == 2) {
            if (!ql::iszero<TOutput, TMass, TScale>(xpi_in[5]))
                ql::B3<TOutput, TMass, TScale>(res, Y, mu2, i);
            else if (!ql::iszero<TOutput, TMass, TScale>(xpi_in[6]) && !ql::iszero<TOutput, TMass, TScale>(xpi_in[7]))
                ql::B4<TOutput, TMass, TScale>(res, Y, mu2, i);
        } else if (offshell == 1)
            ql::B2<TOutput, TMass, TScale>(res, Y, mu2, i);
        else if (offshell == 0)
            ql::B1<TOutput, TMass, TScale>(res, Y, mu2, i);
    }

#ifndef QCDLOOP_BOX_FULL_DISPATCH
    /*!
    * Pruned BO for massive == 0 (0 internal masses).
    * Includes only the B0m dispatch path.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void BO(
        const Kokkos::View<TOutput* [3]>& res,
        const Kokkos::View<TScale*>& mu2,
        const Kokkos::View<TMass* [4]>& m,
        const Kokkos::View<TScale* [6]>& p,
        const int i) {

        const TScale scalefac = ql::Max(
            ql::kAbs(p(i, 4)),
            ql::Max(ql::kAbs(p(i, 5)),
            ql::Max(ql::kAbs(p(i, 0)),
            ql::Max(ql::kAbs(p(i, 1)),
            ql::Max(ql::kAbs(p(i, 2)),
            ql::kAbs(p(i, 3)))))));

        Kokkos::Array<TMass, 13> xpi;
        xpi[0] = m(i, 0) / scalefac;
        xpi[1] = m(i, 1) / scalefac;
        xpi[2] = m(i, 2) / scalefac;
        xpi[3] = m(i, 3) / scalefac;
        xpi[4] = TMass(p(i, 0) / scalefac);
        xpi[5] = TMass(p(i, 1) / scalefac);
        xpi[6] = TMass(p(i, 2) / scalefac);
        xpi[7] = TMass(p(i, 3) / scalefac);
        xpi[8] = TMass(p(i, 4) / scalefac);
        xpi[9] = TMass(p(i, 5) / scalefac);
        xpi[10] = xpi[4] + xpi[5] + xpi[6] + xpi[7] - xpi[8] - xpi[9];
        xpi[11] = -xpi[4] + xpi[5] - xpi[6] + xpi[7] + xpi[8] + xpi[9];
        xpi[12] = xpi[4] - xpi[5] + xpi[6] - xpi[7] + xpi[8] + xpi[9];

        const TScale musq = mu2(i) / scalefac;

        int massive = 0;
        for (size_t j = 0; j < 4; j++) {
            if (!ql::iszero<TOutput, TMass, TScale>(ql::kAbs(xpi[j])))
                massive += 1;
        }

        const TMass y13 = xpi[0] + xpi[2] - xpi[8];
        const TMass y24 = xpi[1] + xpi[3] - xpi[9];

        if (ql::iszero<TOutput, TMass, TScale>(y13) ||
            ql::iszero<TOutput, TMass, TScale>(y24)) {
            res(i, 0) = ql::Constants<TOutput>::_zero();
            res(i, 1) = ql::Constants<TOutput>::_zero();
            res(i, 2) = ql::Constants<TOutput>::_zero();
            return;
        }

        if (massive == 0) {
            ql::B0m<TOutput, TMass, TScale>(res, xpi, musq, i);
        } else {
            res(i, 0) = ql::Constants<TOutput>::_zero();
            res(i, 1) = ql::Constants<TOutput>::_zero();
            res(i, 2) = ql::Constants<TOutput>::_zero();
            return;
        }

        const TScale scalefac2 = scalefac * scalefac;
        res(i, 0) /= scalefac2;
        res(i, 1) /= scalefac2;
        res(i, 2) /= scalefac2;
    }
#endif

}
