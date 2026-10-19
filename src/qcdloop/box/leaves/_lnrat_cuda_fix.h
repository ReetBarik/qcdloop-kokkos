//
// QCDLoop + Kokkos 2025
//
// A100 (sm_80), nvcc/ptxas: ql::BO()'s full 21-leaf dispatch kernel compiles
// TripleFloat (tf) box code to ~255 registers/thread (the hardware ceiling on
// every current NVIDIA GPU, not a LaunchBounds-tunable budget). Under that
// pressure, ptxas has a silent value-corruption defect: a ql::Lnrat() result
// already correct in isolation comes back wrong once a caller combines it
// with several siblings (compute-sanitizer reports zero errors -- this is a
// miscompilation, not UB in our code). Confirmed CUDA-only; HIP, SYCL, and
// host builds of the same source are unaffected. Same class of bug already
// hit and fixed for QuadFloat in the sister repo kokkos-extended-precision-
// demo/xpmath-kokkos, commit b773ad7 ("A100 sm_80 ptxas QF complex atan").
//
// Plain __noinline__/__attribute__((noinline)) does NOT work here -- ptxas
// does not honor it once the kernel is already pinned against the register
// ceiling (confirmed both for Lnrat() itself and, separately, for an entire
// leaf function wrapped in it: cuobjdump shows byte-identical register/spill
// counts and no outlined function symbol either way). Reordering a leaf's
// own arithmetic to shrink its live-register footprint doesn't help either,
// since the ~255-register pressure comes from the textual union of all 21
// leaves ql::BO() can reach, not any single leaf's own slice of it. The only
// mechanism that reliably survives ptxas's inlining is an indirect call
// through a `volatile` function pointer (ptxas cannot devirtualize/reinline
// through it) *combined with* round-tripping the result through volatile
// memory before it's combined with other values -- each alone is
// insufficient, only together do they force ptxas to materialize the result
// instead of keeping it live in registers across the combining arithmetic.
//
// Used by box/leaves/B2.h, B3.h, B4.h -- factored out here (rather than
// defined in each) because box/dispatch/B0m.h #includes all three leaves
// into one translation unit regardless of which is dispatched at runtime,
// and redefining the same templates in more than one would violate ODR.
#pragma once

#include "kokkosUtils.h"

namespace ql
{

    // NOTE: the confirmed-working experiment (examples/box_leaves_fix/) left
    // this unconditional, so it also paid a volatile memory round-trip's
    // cost on HIP/SYCL/host, and (being backend-agnostic) would also have
    // paid it for dd/ff/qf on CUDA. Gated here to CUDA+tf specifically so it
    // is a true identity no-op everywhere else, matching the "only tf on
    // CUDA needs this" scope established above.
    template<typename T>
    KOKKOS_INLINE_FUNCTION T volatile_roundtrip_fix(T const& x) {
#if defined(__CUDA_ARCH__) && defined(XPMATH_BACKEND_tf)
        volatile unsigned char buf[sizeof(T)];
        const unsigned char* src = reinterpret_cast<const unsigned char*>(&x);
        for (unsigned b = 0; b < sizeof(T); ++b) buf[b] = src[b];
        T out;
        unsigned char* dst = reinterpret_cast<unsigned char*>(&out);
        for (unsigned b = 0; b < sizeof(T); ++b) dst[b] = buf[b];
        return out;
#else
        return x;
#endif
    }

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_FUNCTION TOutput lnrat_callee_fix(TMass const& x, TMass const& y) {
        return ql::Lnrat<TOutput, TMass, TScale>(x, y);
    }

    template<typename TOutput, typename TMass, typename TScale>
    using LnratFnFixT = TOutput (*)(TMass const&, TMass const&);

    // Only tf (TripleFloat) was ever observed to hit this defect -- dd/ff/qf
    // were confirmed clean on the exact same A100/CUDA build (see
    // HANDOFF_a100_tf_nvcc_nan.md / memory project_a100_tf_nvcc_nan_bug.md).
    // Gate on XPMATH_BACKEND_tf too (not just __CUDACC__) so dd/ff/qf never
    // pay this workaround's indirect-call cost on CUDA either -- each
    // backend is its own TU with exactly one XPMATH_BACKEND_* macro set
    // (see kokkosMaths_xp.h), so this is a compile-time, not runtime, choice.
#if defined(__CUDACC__) && defined(XPMATH_BACKEND_tf)
    // static: an inline __device__ variable must have internal linkage when
    // compiled in whole-program mode (-rdc=false, the default here).
    template<typename TOutput, typename TMass, typename TScale>
    __device__ static inline const volatile LnratFnFixT<TOutput, TMass, TScale> kLnratIndirectFix =
        &lnrat_callee_fix<TOutput, TMass, TScale>;
#endif

    // Drop-in replacement for ql::Lnrat() inside box/leaves/{B2,B3,B4}.h:
    // only the CUDA+tf device path routes through the indirect-call
    // workaround above; every other path (host, HIP, SYCL, or CUDA with
    // dd/ff/qf) is a plain call with no added cost. Always pair with
    // volatile_roundtrip_fix() on the result.
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput lnrat_fix(TMass const& x, TMass const& y) {
#if defined(__CUDA_ARCH__) && defined(XPMATH_BACKEND_tf)
        return kLnratIndirectFix<TOutput, TMass, TScale>(x, y);
#else
        return ql::Lnrat<TOutput, TMass, TScale>(x, y);
#endif
    }

}
