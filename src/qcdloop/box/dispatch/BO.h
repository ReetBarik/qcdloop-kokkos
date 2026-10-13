//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//          Stefan Hoeche   : shoeche@fnal.gov
//          Max Knobbe      : mknobbe@fnal.gov
//
// Top-level box dispatch: classifies the internal-mass count and calls the
// matching mass-count group from box/dispatch/{B0m,B1m,B2m,B3m,B4m}.h, each
// of which in turn classifies and calls its leaf(s) from box/leaves/. This
// is the general, full-dispatch entry point every call site that doesn't
// statically know its integral should use (via boxGPU.h).

#pragma once

// Suppress the individual pruned BO() definitions from the five group
// headers below -- this file provides the one real, full BO().
#define QCDLOOP_BOX_FULL_DISPATCH

#include "box/classify.h"
#include "box/dispatch/B0m.h"
#include "box/dispatch/B1m.h"
#include "box/dispatch/B2m.h"
#include "box/dispatch/B3m.h"
#include "box/dispatch/B4m.h"


namespace ql
{

    /*!
    * \brief box integral
    *
    * Full dispatch function for box integrals. Computes the box integral
    * \f$ I_4^{D=4-2\epsilon}(p_1^2,p_2^2,p_3^2,p_4^2;s_{12},s_{23};m_1^2,m_2^2,m_3^2,m_4^2) \f$
    * which is defined as:
    * \f[
    * I_4^{D=4-2\epsilon} = \mu^{2\epsilon} \int \frac{d^Dk}{i \pi^{D/2}} \frac{1}{\prod_i (k^2_i-m_i^2+i\epsilon)}
    * \f]
    *
    * The result is a Laurent expansion about \f$ \epsilon \f$
    * \f[
    * I_4^{D=4-2\epsilon} = c_0 + c_1 \epsilon^{-1} + c_2 \epsilon^{-2}
    * \f]
    *
    * Implementation of the formulae of Denner et al. \cite Denner:1991qq,
    * 't Hooft and Veltman \cite tHooft:1978xw, Bern et al. \cite Bern:1993kr.
    *
    * \param res output object res[i,0,1,2] the coefficients in the Laurent series
    * \param mu2 is the square of the scale mu (per element)
    * \param m are the squares of the masses of the internal lines [batch][4]
    * \param p are the four-momentum squared of the external lines [batch][6]
    * \param i element index
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void BO(
        const Kokkos::View<TOutput* [3]>& res,      // Output view
        const Kokkos::View<TScale*>& mu2,           // Scale parameter (per element)
        const Kokkos::View<TMass* [4]>& m,          // Masses view [batch][4]
        const Kokkos::View<TScale* [6]>& p,         // Momenta view [batch][6]
        const int i) {                              // Element index

        // Compute scalefac for this element
        const TScale scalefac = ql::Max(
            ql::kAbs(p(i, 4)),
            ql::Max(ql::kAbs(p(i, 5)),
            ql::Max(ql::kAbs(p(i, 0)),
            ql::Max(ql::kAbs(p(i, 1)),
            ql::Max(ql::kAbs(p(i, 2)),
            ql::kAbs(p(i, 3)))))));

        // Compute xpi array for this element
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

        // Compute musq for this element
        const TScale musq = mu2(i) / scalefac;

        // Count number of internal masses
        const int massive = ql::classify_massive<TOutput, TMass, TScale>(xpi);

        // Check Cayley elements
        const TMass y13 = xpi[0] + xpi[2] - xpi[8];
        const TMass y24 = xpi[1] + xpi[3] - xpi[9];

        if (ql::iszero<TOutput, TMass, TScale>(y13) ||
            ql::iszero<TOutput, TMass, TScale>(y24)) {
            res(i, 0) = ql::Constants<TOutput>::_zero();
            res(i, 1) = ql::Constants<TOutput>::_zero();
            res(i, 2) = ql::Constants<TOutput>::_zero();
            return;
        }

        // Call appropriate B function based on massive count
        if (massive == 0) {
            ql::B0m<TOutput, TMass, TScale>(res, xpi, musq, i);
        } else if (massive == 1) {
            ql::B1m<TOutput, TMass, TScale>(res, xpi, musq, i);
        } else if (massive == 2) {
            ql::B2m<TOutput, TMass, TScale>(res, xpi, musq, i);
        } else if (massive == 3) {
            ql::B3m<TOutput, TMass, TScale>(res, xpi, musq, i);
        } else if (massive == 4) {
            ql::B4m<TOutput, TMass, TScale>(res, xpi, i);
        }

        // Normalize results
        const TScale scalefac2 = scalefac * scalefac;
        res(i, 0) /= scalefac2;
        res(i, 1) /= scalefac2;
        res(i, 2) /= scalefac2;
    }


}
