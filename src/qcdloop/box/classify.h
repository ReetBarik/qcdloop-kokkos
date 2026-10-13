//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//          Stefan Hoeche   : shoeche@fnal.gov
//          Max Knobbe      : mknobbe@fnal.gov
//
// Standalone classification helpers for the box dispatch layer (box/dispatch/).
// These decide which leaf integral a given (xpi, massive-count) belongs to and
// what permutation is needed to reach the leaf's canonical input ordering.
// KOKKOS_INLINE_FUNCTION expands to __device__ __host__ inline, so these are
// callable from host code (e.g. a future host-side bucketing pass) with no
// extra work.

#pragma once

#include "kokkosUtils.h"


namespace ql
{

    // massive == 0 group (B0m): index-permutation table used to bring the
    // off-shell leg(s) into canonical position before Y is built.
    KOKKOS_INLINE_FUNCTION
    constexpr int swap_b0m(int i, int j) {
        constexpr int val[13][4] = {
            {3, 2, 1, 0},
            {0, 3, 2, 1},
            {1, 0, 3, 2},
            {2, 1, 0, 3},
            {7, 6, 5, 4},
            {4, 7, 6, 5},
            {5, 4, 7, 6},
            {6, 5, 4, 7},
            {9, 8, 9, 8},
            {8, 9, 8, 9},
            {10, 10, 10, 10},
            {12, 11, 12, 11},
            {11, 12, 11, 12},
        };
        return val[i][j];
    }

    KOKKOS_INLINE_FUNCTION
    constexpr int jsort_b0m(int i) {
        constexpr int val[4] = {3,0,1,2};
        return val[i];
    }

    // Classification result for the massive == 0 (B0m) group: how many legs
    // are off-shell, and which leg index(es) carry the on-shell/off-shell
    // markers the dispatcher needs to pick a permutation. This only depends
    // on which of xpi[4..7] are exactly zero -- a runtime fact even though
    // the *count* class (massive == 0) is already known to the caller.
    struct ClassifyB0m {
        int offshell;
        int jsort0;
        int jsort1;
        int jsort2;
    };

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    ClassifyB0m classify_offshell_b0m(const Kokkos::Array<TMass, 13>& xpi) {
        ClassifyB0m c{0, 0, 0, 0};
        for (int j = 0; j < 4; j++) {
            if (!ql::iszero<TOutput, TMass, TScale>(xpi[j + 4])) {
                c.offshell += 1;
                if (c.jsort1 == 0) c.jsort1 = j + 1;
                else c.jsort2 = j + 1;
            } else
                c.jsort0 = j;
        }
        return c;
    }

    // massive == 1 group (B1m): index-permutation table used to bring the
    // single massive internal line into canonical position before Y is built.
    KOKKOS_INLINE_FUNCTION
    constexpr int swap_b1m(int i, int j) {
        constexpr int val[13][5] = {
            {3, 2, 1, 0, 2},
            {0, 3, 2, 1, 1},
            {1, 0, 3, 2, 0},
            {2, 1, 0, 3, 3},
            {7, 6, 5, 4, 5},
            {4, 7, 6, 5, 4},
            {5, 4, 7, 6, 7},
            {6, 5, 4, 7, 6},
            {9, 8, 9, 8, 8},
            {8, 9, 8, 9, 9},
            {10, 10, 10, 10, 10},
            {12, 11, 12, 11, 11},
            {11, 12, 11, 12, 12},
        };
        return val[i][j];
    }

    // Classification for the massive == 1 (B1m) group: which leg index
    // carries the nonzero internal mass. This only depends on which of
    // xpi[0..3] is nonzero -- a runtime fact even though the *count* class
    // (massive == 1) is already known to the caller.
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    int classify_mass_b1m(const Kokkos::Array<TMass, 13>& xpi) {
        int jsort = 0;
        for (int j = 0; j < 4; j++) {
            if (!ql::iszero<TOutput, TMass, TScale>(xpi[j])) {
                jsort = j;
            }
        }
        return jsort;
    }

    // massive == 2 group (B2m): index-permutation table used to bring the
    // two massive internal lines into canonical position before Y is built.
    KOKKOS_INLINE_FUNCTION
    constexpr int swap_b2m(int i, int j) {
        constexpr int val[13][5] = {
            {3, 2, 1, 0, 2},
            {0, 3, 2, 1, 1},
            {1, 0, 3, 2, 0},
            {2, 1, 0, 3, 3},
            {7, 6, 5, 4, 5},
            {4, 7, 6, 5, 4},
            {5, 4, 7, 6, 7},
            {6, 5, 4, 7, 6},
            {9, 8, 9, 8, 8},
            {8, 9, 8, 9, 9},
            {10, 10, 10, 10, 10},
            {11, 11, 11, 11, 11},
            {12, 12, 12, 12, 12},
        };
        return val[i][j];
    }

    // Classification for the massive == 2 (B2m) group: which two leg
    // indices carry the nonzero internal masses, and their cyclic
    // separation (jdiff). jdiff == 2 means the masses sit on opposite
    // (diagonal) legs; jdiff == 1 or 3 means they are cyclically adjacent
    // (3 arises from the wraparound edge between legs 0 and 3). This only
    // depends on which two of xpi[0..3] are nonzero -- a runtime fact even
    // though the *count* class (massive == 2) is already known to the
    // caller. The permutation-index choice and the actual permutation
    // stay in the dispatch body (box/dispatch/B2m.h), since jdiff == 3 is
    // permuted by a literal 0 rather than by jsort2 -- a caller-specific
    // wrinkle, not a pure classification fact.
    struct ClassifyB2m {
        int jsort1;
        int jsort2;
        int jdiff;
    };

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    ClassifyB2m classify_mass_b2m(const Kokkos::Array<TMass, 13>& xpi) {
        int jsort1 = -1, jsort2 = -1;
        for (int j = 0; j < 4; j++) {
            if (!ql::iszero<TOutput, TMass, TScale>(xpi[j])) {
                if (jsort1 == -1) {
                    jsort1 = j;
                } else {
                    jsort2 = j;
                }
            }
        }
        return ClassifyB2m{jsort1, jsort2, jsort2 - jsort1};
    }

}
