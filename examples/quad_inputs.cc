// Compiled only with host g++. Do not include Kokkos headers here.
#include "quad_inputs.h"

#include <cstdlib>

// GCC's __float128 (and libquadmath, which backs the `q` literal suffix
// and quadmath.h) is only ported to targets that need it: x86_64,
// powerpc64, s390x, ... AArch64 has no libquadmath at all -- __float128
// is not even a recognized type there (verified on gcc/11.2.0
// aarch64-unknown-linux-gnu during the GB300/Blackwell bring-up,
// 2026-10-15: "'__float128' does not name a type"). It doesn't need one
// though: AArch64's ABI already makes `long double` IEEE-754 binary128
// (sizeof(long double)==16, confirmed on the same toolchain), the exact
// format __float128 provides elsewhere. So on aarch64 this file uses
// `long double` + the standard `L` literal suffix instead; every other
// architecture's codegen is untouched (Quad is still exactly __float128,
// QUAD_LIT(x) still expands to the same `xq` token).
#if defined(__aarch64__)
using Quad = long double;
#define QUAD_LIT(x) x##L
#else
extern "C" {
#include <quadmath.h>
}
using Quad = __float128;
#define QUAD_LIT(x) x##q
#endif

static DdLimbs q_to_limbs(Quad q) {
    double hi = (double)q;
    double lo = (double)(q - (Quad)hi);
    return DdLimbs{hi, lo};
}

// Same successive split QuadFloat(double) uses, starting from the quad
// value so the later words are not discarded.
static QuadWords q_to_words(Quad q) {
    Quad r = q;
    float w[4];
    for (int i = 0; i < 4; ++i) {
        float c = (float)r;
        w[i] = c;
        r -= (Quad)c;
    }
    return QuadWords{w[0], w[1], w[2], w[3]};
}

DdLimbs dd_mu2() {
    return q_to_limbs(QUAD_LIT(91.2) * QUAD_LIT(91.2));
}

DdLimbs dd_mass_10() {
    return q_to_limbs(QUAD_LIT(10.0));
}

DdLimbs dd_mass_4p9_sq() {
    return q_to_limbs(QUAD_LIT(4.9) * QUAD_LIT(4.9));
}

DdLimbs dd_mass_50_sq() {
    return q_to_limbs(QUAD_LIT(50.0) * QUAD_LIT(50.0));
}

DdLimbs dd_rand(double min, double max) {
    Quad lo = min;
    Quad hi = max;
    return q_to_limbs(lo + std::rand() * 1.0 / RAND_MAX * (hi - lo));
}

DdLimbs dd_rands(double min, double max) {
    Quad lo = min;
    Quad hi = max;
    Quad r1 = lo + std::rand() * 1.0 / RAND_MAX * (hi - lo);
    Quad draw = std::rand() * 1.0 / RAND_MAX;
    if (draw < 0.5)
        return q_to_limbs(-r1);
    else
        return q_to_limbs(r1);
}

QuadWords qw_mu2() {
    return q_to_words(QUAD_LIT(91.2) * QUAD_LIT(91.2));
}

QuadWords qw_mass_10() {
    return q_to_words(QUAD_LIT(10.0));
}

QuadWords qw_mass_4p9_sq() {
    return q_to_words(QUAD_LIT(4.9) * QUAD_LIT(4.9));
}

QuadWords qw_mass_50_sq() {
    return q_to_words(QUAD_LIT(50.0) * QUAD_LIT(50.0));
}

QuadWords qw_rand(double min, double max) {
    Quad lo = min;
    Quad hi = max;
    return q_to_words(lo + std::rand() * 1.0 / RAND_MAX * (hi - lo));
}

QuadWords qw_rands(double min, double max) {
    Quad lo = min;
    Quad hi = max;
    Quad r1 = lo + std::rand() * 1.0 / RAND_MAX * (hi - lo);
    Quad draw = std::rand() * 1.0 / RAND_MAX;
    if (draw < 0.5)
        return q_to_words(-r1);
    else
        return q_to_words(r1);
}
