# Box leaf/dispatch decomposition: validation and code-size measurements

Status: Steps 6-7 of the box leaf/dispatch decomposition task. Companion to
`~/xpmath-kokkos/repro/REPORT.md`, which investigated why `ql::BO()`'s
unconditional dispatch over all 21 box integrals makes the compiled kernel
code size vastly exceed GPU icache capacity. This refactor splits `BO()`'s
dispatch into `src/qcdloop/box/leaves/*.h` (21 near-verbatim leaf headers)
and `src/qcdloop/box/dispatch/{B0m,B1m,B2m,B3m,B4m,BO}.h` (classification +
composition), and adds `examples/boxGPU_test_direct.cc`, which calls each
leaf directly instead of going through `BO()`'s 5-way branch. This document
measures whether that split actually reduces per-kernel compiled code size.

## Step 6: consolidated validation

All numeric output is bit-for-bit identical to the original pre-refactor
tree (`feff43f`), Serial Kokkos, 100k-point batches:

| build | vs. `feff43f` | rows compared | result |
|---|---|---|---|
| `boxGPU_test` (double, routed through `BO()`) | identical | 2,100,000 | 0 diff lines |
| `boxGPU_test_direct` (double, direct leaf calls) | identical | 2,100,000 | 0 diff lines |
| `boxGPU_test_dd` (DoubleDouble, routed through `BO()`) | identical | 210,000 | 0 diff lines |
| `boxGPU_test_ff` (FloatFloat, routed through `BO()`) | n/a (confirmatory) | 210,000 | runs clean, no NaN/Inf |

This reconfirms, at the end of the full decomposition, what every
intermediate commit (`22c37d7`, `d91b6d1`, `013540b`, `eefc590`, `f22dc9e`,
`b0d61d9`, `0df23b0`) already validated incrementally: the leaf/dispatch
split changed nothing about the arithmetic, only where the code for it
lives.

## Step 7: compiled code-size measurement

Methodology matches REPORT.md §2.1: `llvm-objdump --offloading` to extract
the embedded GCN code object, `llvm-size`/`llvm-nm --print-size -C` for
whole-binary and per-symbol sizes (HIP); `cuobjdump`/`nvcc` toolchain for
CUDA. Builds only — no GPU queue time, nothing launched.

### HIP / gfx90a (ROCm 7.0.2, MI250 target)

**Whole binary device `.text`:**

| build | double | DoubleDouble (`dd`) |
|---|---|---|
| `boxGPU_test` (routed through `BO()`) | 3,550,714 B | 14,493,493 B |
| `boxGPU_test_direct` (direct leaf calls) | 3,767,361 B | *(no dd variant — see gap below)* |

The direct-call binary's **whole-binary** total is not smaller — it's
slightly larger. This is expected and is exactly the distinction REPORT.md
warned not to conflate: whole-binary size is the *sum* of 21 independently
compiled, now mostly-inlined leaf bodies (no more cross-section sharing of
one `BO()` copy), not what's resident for any *one* kernel launch. The
number that matters for the icache-pressure failure mode is what a single
kernel actually reaches:

**`ql::` out-of-line symbol census** (functions the compiler kept separate
rather than inlining — i.e., what's actually shared/reachable across call
sites), double:

| build | out-of-line `ql::` symbols | total bytes | notable absences/presences |
|---|---|---|---|
| `boxGPU_test` | 12 | 3,443,872 B | `ql::BO` present (836,760 B) — the universal entry point every one of the 21 kernels calls into, which itself reaches `B0m`-`B4m` (inlined into `BO`) and keeps `BIN1`/`BIN2`/`BIN3`/`BIN4` as separate callees |
| `boxGPU_test_direct` | 11 | 2,083,100 B | `ql::BO` and `ql::BIN1` **no longer exist as symbols at all** — both fully inlined into their single-caller kernel, since the compiler's cost heuristic now only has one call site to weigh instead of a universally-shared one |

The practical conclusion: in `boxGPU_test`, **every one of the 21 section
kernels reaches the same ~3.44 MB of `ql::` code**, regardless of which of
the 21 integrals it is actually computing, because `BO()`'s branch is
unconditional and the compiler cannot prove which arm runs. In
`boxGPU_test_direct`, 18 of the 21 sections (`BIN0`, `B1`-`B10`, `B11`-`B16`
except `B11`'s noinline-forced helper) now have their entire leaf fully
inlined into a single, section-specific kernel with no other box-integral
code reachable from it at all. The remaining 3 sections (`BIN2`, `BIN3`,
`BIN4` — the ones literally named for those leaves) still reach one
out-of-line leaf each, but **only their own**, not the other two's and
not the other 18 leaves':

| leaf | size (both binaries — unchanged, as expected) | reachable from, before | reachable from, after |
|---|---|---|---|
| `BIN2` | 494,432 B | all 21 kernels | only the `BIN2` kernel |
| `BIN3` | 743,328 B | all 21 kernels | only the `BIN3` kernel |
| **`BIN4`** | **711,156 B** | all 21 kernels | only the `BIN4` kernel |

`BIN4` is called out specifically per the task's instruction: its own size
is **unchanged** (711,156 B in both binaries) — exactly as expected, since
REPORT.md found its size is dominated by its own intrinsic complex
error-free-transformation arithmetic, not dispatch-tree residency, and this
refactor never touches a leaf's own math. What changes is not `BIN4`'s
size but *who else's kernel is forced to carry it*: before, all 21 kernels
paid the 711,156 B `BIN4` cost (plus `BIN2`'s, `BIN3`'s, and everything
else's) whether or not they needed it; after, only the one kernel that
actually computes `BIN4` does.

**DoubleDouble (`dd`), HIP:** only the `BO()`-routed `boxGPU_test_dd` could
be measured — see the gap noted below. Its numbers (14,493,493 B whole
binary, BIN4 = 4,121,452 B, 31 out-of-line `ql::` symbols totaling
14,273,432 B) match REPORT.md's pre-refactor v0.3.0 baseline figures
(14,492,981 B / 4,119,404 B) to within noise (<0.1%), confirming this
refactor is a pure reshuffling that changed nothing about the `dd`
compiled footprint on the path that already existed.

**Gap, flagged honestly:** `examples/boxGPU_test_direct.cc` is double-only
(Step 5 explicitly scoped it that way). There is no `dd`-instantiated
direct-leaf driver, so the dd-specific "all 21 kernels reach the whole
14.3 MB tree, vs. one kernel reaches only its own leaf" comparison that
would be the single most relevant number for the actual MI250 hang (which
was observed on `dd`, not `double`) **has not been measured**. The
double results make the qualitative mechanism clear and there's no reason
to expect a different *qualitative* outcome for `dd` (the dispatch
structure is type-independent), but the *absolute* per-leaf sizes for `dd`
would almost certainly be larger than double's in absolute terms (per
REPORT.md, `dd` arithmetic is 4-5x costlier per operation), so this is not
something to extrapolate numerically — it needs its own measurement,
which would require extending the direct driver to the `dd`/`ff`/`qf`/`tf`
backends (natural follow-on, not attempted here to stay in scope).

### CUDA / sm_80 (CUDA 12.9.1, A100 target)

Attempted the same double `boxGPU_test` build (BO-routed) as a starting
point. `ptxas` ran for **over 30 minutes of continuous 100% CPU** before
being killed to stay within this step's time budget. This was not
expected from REPORT.md's own numbers — REPORT.md's completed CUDA double
baseline (4,956,800 B/kernel) didn't flag double as pathologically slow,
only `dd` was (~100 minutes). Possible explanations, **not independently
diagnosed**: environment/toolchain variance on this node vs. REPORT.md's
original run, or the cumulative effect of this refactor's extra levels of
`#include` indirection increasing `ptxas`'s input IR size even for
`double`. Either way, **no CUDA measurement was obtained in this pass** —
`boxGPU_test_direct` and any `dd`/CUDA build were not attempted given the
double BO-routed build alone exceeded the time budget. This is a real gap,
flagged rather than papered over: a dedicated follow-up with a longer time
allowance (or `-rdc=true`, per REPORT.md §4's suggestion for making
`noinline`/out-of-lining actually work on CUDA) is needed before any CUDA
conclusion can be drawn for this refactor specifically.

## Verdict

Does this leaf/dispatch split "plausibly remove the need for the existing
slice-based runtime workaround for the lighter integrals"? **Not
confirmable from `.text` size alone** — this measurement shows a real,
large reduction in *reachable* code per kernel on HIP/double (from ~3.44 MB
universally to a few KB for 18 of 21 integrals, and to one leaf's own size,
~500-740 KB, for the remaining 3), which is the structural quantity
REPORT.md identified as the root cause of the icache-pressure failure
mode. But:

- The actual hang was observed on **`dd`**, not `double`, and no `dd`
  direct-driver measurement exists yet (gap above).
- Even the best HIP/double per-kernel number here (a few KB to ~740 KB) is
  compared against REPORT.md's own order-of-magnitude framing (32-64 KB
  realistic L1 icache) — the 3 still-large leaves (`BIN2`/`BIN3`/`BIN4`)
  remain well above that line on their own, even fully isolated from the
  other 20.
- No CUDA data was obtained this pass.
- **Only running the actual unsliced batch on real MI250 hardware** (per
  REPORT.md §8 and this task's own Step 10) can answer whether the hang
  itself is fixed, reduced, or unaffected. This document does not claim
  that answer and explicitly defers it, per the task's standing
  instruction to ask before consuming Cobalt queue time.

## Next steps this document recommends, not performed here

1. Extend `boxGPU_test_direct.cc` (or add a sibling) to cover `dd` (the
   type that actually hangs), to get the comparison that matters most.
2. Retry CUDA measurement with a longer time budget, or `-rdc=true`, on a
   quieter window.
3. Step 8 (host-side bucketed dispatch) and Step 9 (final write-up) are
   separate, already-scoped follow-ons to this measurement pass.
4. Real-hardware validation (Step 10) needs the user's explicit go-ahead
   before any Cobalt queue time is spent.
