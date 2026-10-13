//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//          Stefan Hoeche   : shoeche@fnal.gov
//          Max Knobbe      : mknobbe@fnal.gov
//
// Dispatch for box integrals with 2 internal masses: classifies which two
// legs carry the masses and whether they are adjacent or opposite, builds
// Y, and calls the matching leaf from box/leaves/ (re-permuting Y via the
// shared Ycalc helper for the sub-cases that need it). Contains B2ma
// (adjacent), B2mo (opposite), the top-level B2m dispatcher, and (for
// selective compilation) a pruned BO() for massive == 2 only.

#pragma once

#include "box/box_common.h"
#include "box/classify.h"
#include "box/leaves/BIN2.h"
#include "box/leaves/B11.h"
#include "box/leaves/B12.h"
#include "box/leaves/B13.h"
#include "box/leaves/B14.h"
#include "box/leaves/B15.h"


namespace ql
{

    /*!
    * This function trigger the topologies with 2-offshell external lines (adjacent).
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param xpi array with masses and momenta squared
    * \param mu2 is the square of the scale mu
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B2ma(
        const Kokkos::View<TOutput* [3]>& res,
        const Kokkos::Array<TMass, 13>& xpi,
        const TScale& mu2,
        const int i) {

        Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Y;
        Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Yalt;

        Y[0][0] = xpi[0];
        Y[1][1] = xpi[1];
        Y[2][2] = xpi[2];
        Y[3][3] = xpi[3];
        Y[0][1] = Y[1][0] = ql::Constants<TMass>::_half() * (xpi[0] + xpi[1] - xpi[4]);
        Y[0][2] = Y[2][0] = ql::Constants<TMass>::_half() * (xpi[0] + xpi[2] - xpi[8]);
        Y[0][3] = Y[3][0] = ql::Constants<TMass>::_half() * (xpi[0] + xpi[3] - xpi[7]);
        Y[1][2] = Y[2][1] = ql::Constants<TMass>::_half() * (xpi[1] + xpi[2] - xpi[5]);
        Y[1][3] = Y[3][1] = ql::Constants<TMass>::_half() * (xpi[1] + xpi[3] - xpi[9]);
        Y[2][3] = Y[3][2] = ql::Constants<TMass>::_half() * (xpi[2] + xpi[3] - xpi[6]);

        ql::Ycalc<TOutput, TMass, TScale>(Y, Yalt, 2, ql::iszero<TOutput, TMass, TScale>(xpi[2]));

        const bool zY01 = ql::iszero<TOutput, TMass, TScale>(Y[0][1]);
        const bool zY12 = ql::iszero<TOutput, TMass, TScale>(Y[1][2]);
        const bool zY03 = ql::iszero<TOutput, TMass, TScale>(Y[0][3]);

        if (zY01 && zY12 && zY03)
            ql::B11<TOutput, TMass, TScale>(res, Y, mu2, i);
        else if (zY01 && zY12 && !zY03)
            ql::B12<TOutput, TMass, TScale>(res, Y, mu2, i);
        else if (ql::iszero<TOutput, TMass, TScale>(Yalt[0][1]) && ql::iszero<TOutput, TMass, TScale>(Yalt[1][2]) && !ql::iszero<TOutput, TMass, TScale>(Yalt[0][3]))
            ql::B12<TOutput, TMass, TScale>(res, Yalt, mu2, i);
        else if (zY01 && !zY12 && !zY03)
            ql::B13<TOutput, TMass, TScale>(res, Y, mu2, i);
        else
            ql::BIN2<TOutput, TMass, TScale>(res, Y, i);
    }

    /*!
    * This function trigger the topologies with 2-offshell external lines (opposite).
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param xpi array with masses and momenta squared
    * \param mu2 is the square of the scale mu
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B2mo(
        const Kokkos::View<TOutput* [3]>& res,
        const Kokkos::Array<TMass, 13>& xpi,
        const TScale& mu2,
        const int i) {

        Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Y;

        Y[0][0] = xpi[0];
        Y[1][1] = xpi[1];
        Y[2][2] = xpi[2];
        Y[3][3] = xpi[3];
        Y[0][1] = Y[1][0] = ql::Constants<TMass>::_half() * (xpi[0] + xpi[1] - xpi[4]);
        Y[0][2] = Y[2][0] = ql::Constants<TMass>::_half() * (xpi[0] + xpi[2] - xpi[8]);
        Y[0][3] = Y[3][0] = ql::Constants<TMass>::_half() * (xpi[0] + xpi[3] - xpi[7]);
        Y[1][2] = Y[2][1] = ql::Constants<TMass>::_half() * (xpi[1] + xpi[2] - xpi[5]);
        Y[1][3] = Y[3][1] = ql::Constants<TMass>::_half() * (xpi[1] + xpi[3] - xpi[9]);
        Y[2][3] = Y[3][2] = ql::Constants<TMass>::_half() * (xpi[2] + xpi[3] - xpi[6]);

        const bool zY00 = ql::iszero<TOutput, TMass, TScale>(Y[0][0]);
        const bool zY22 = ql::iszero<TOutput, TMass, TScale>(Y[2][2]);
        const bool zY01 = ql::iszero<TOutput, TMass, TScale>(Y[0][1]);
        const bool zY03 = ql::iszero<TOutput, TMass, TScale>(Y[0][3]);
        const bool zY12 = ql::iszero<TOutput, TMass, TScale>(Y[1][2]);
        const bool zY23 = ql::iszero<TOutput, TMass, TScale>(Y[2][3]);

        if (zY00 && zY22 && zY01 && zY03 && zY12 && zY23)
            ql::B14<TOutput, TMass, TScale>(res, Y, mu2, i);
        else if (zY00 && zY22 && zY01 && zY03)
            ql::B15<TOutput, TMass, TScale>(res, Y, mu2, i);
        else if (zY00 && zY22 && zY12 && zY23) {
            Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Yalt;
            ql::Ycalc<TOutput, TMass, TScale>(Y, Yalt, 2, ql::iszero<TOutput, TMass, TScale>(xpi[2]));
            ql::B15<TOutput, TMass, TScale>(res, Yalt, mu2, i);
        } else {
            Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Yadj;
            const int swap23[4] = {0, 2, 1, 3};
            for (int j = 0; j < 4; j++)
                for (int k = 0; k < 4; k++)
                    Yadj[j][k] = Y[swap23[j]][swap23[k]];

            ql::BIN2<TOutput, TMass, TScale>(res, Yadj, i);
        }
    }

    /*!
    * This function trigger the topologies with 2 internal masses.
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param xpi array with masses and momenta squared
    * \param mu2 is the square of the scale mu
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B2m(
        const Kokkos::View<TOutput* [3]>& res,
        const Kokkos::Array<TMass, 13>& xpi,
        const TScale& mu2,
        const int i) {

        const ClassifyB2m c = ql::classify_mass_b2m<TOutput, TMass, TScale>(xpi);
        const int jdiff = c.jdiff;

        Kokkos::Array<TMass, 13> xpiout;

        if (jdiff == 1 || jdiff == 2) {
            for (size_t j = 0; j < 13; j++) {
                xpiout[ql::swap_b2m(j, c.jsort2)] = xpi[j];
            }
        } else if (jdiff == 3) {
            for (size_t j = 0; j < 13; j++) {
                xpiout[ql::swap_b2m(j, 0)] = xpi[j];
            }
        }

        if (jdiff == 2) {
            ql::B2mo<TOutput, TMass, TScale>(res, xpiout, mu2, i);
        } else {
            ql::B2ma<TOutput, TMass, TScale>(res, xpiout, mu2, i);
        }
    }

#ifndef QCDLOOP_BOX_FULL_DISPATCH
    /*!
    * Pruned BO for massive == 2 (2 internal masses).
    * Includes only the B2m dispatch path.
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

        if (massive == 2) {
            ql::B2m<TOutput, TMass, TScale>(res, xpi, musq, i);
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
