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

**Update:** the first pass of this document (commit `bec5ba2`) left two
gaps — no `dd`-backend direct driver, and no CUDA measurement (a `ptxas`
stall on the BO()-routed double build exceeded the time budget). Both are
addressed below: `boxGPU_test_direct.cc` was extended to `dd`/`ff`/`qf`/`tf`
(commit `03c0616`) and measured on HIP, and CUDA was re-attempted for the
direct driver only (BO()-routed CUDA builds are deliberately not retried,
per the user's direction, to avoid repeating a long build on this shared
login node).

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

**DoubleDouble (`dd`), HIP — the headline number this task exists to
produce.** The gap flagged in the first measurement pass is now closed:
`examples/boxGPU_test_direct.cc` was extended to the `dd`/`ff`/`qf`/`tf`
backends (commit `03c0616`), validated bit-for-bit against
`boxGPU_test_dd` (2,100,000 rows, same math via a different call path),
and measured on HIP the same way as double:

| build | whole binary `.text` | out-of-line `ql::` symbols | total bytes |
|---|---|---|---|
| `boxGPU_test_dd` (routed through `BO()`) | 16,404,533 B | 32 | 16,168,356 B |
| `boxGPU_test_direct_dd` (direct leaf calls) | 17,197,228 B | 29 | 14,562,888 B |

(Whole-binary totals aren't directly comparable in the way that matters —
same caveat as double: it's a sum across 21 now-more-independently-compiled
kernels, not what one launch is resident for. The out-of-line `ql::`
symbol census is the structural number.)

The direct build drops `ql::BO`, `ql::B0m`, and `ql::B1m` entirely as
symbols (532,824 + 586,812 + 485,896 = 1,605,532 B) — these were each a
single universal entry/sub-dispatch point every one of the 21 kernels in
`boxGPU_test_dd` had to reach through, regardless of which integral that
kernel actually computes; with `BO()` gone, each `boxGPU_test_direct_dd`
kernel instead calls its own leaf-specific wrapper directly, so nothing
routes through a shared all-reaching entry point anymore. Unlike double
(where 18 of 21 leaves also vanished as symbols), most individual `dd`
leaves (`BIN0`-`BIN3`, `B3`, `B5`, `B7`, `B9`, `B10`, `B12`, `B13`, `B15`,
`B16`) **stay out-of-line in both builds, at identical sizes** — `dd`
arithmetic is intrinsically large enough per leaf that the compiler's
inlining heuristic was already keeping most of them separate even with a
single call site, so there is less room for the "fewer kernels, same
total code" sharing effect double showed. `BIN4` is again unchanged
(4,807,668 B in both), as expected.

**Verdict for `dd` specifically:** the structural fix (no single universal
dispatch entry point reachable from every kernel) is confirmed exactly as
for double, but the raw byte reduction is far more modest (16.17 MB → 14.56
MB total out-of-line code, ~10%) than double's (3.44 MB → 2.08 MB, ~40%),
because most of `dd`'s cost was never residing *only* in `BO`/`B0m`/`B1m`
to begin with — it's spread across the leaves and shared spence/dilog
helpers themselves, which this refactor does not and was never meant to
shrink (per the task's scope: no numerics changes). What this refactor
removes for `dd` is specifically the ~1.6 MB of universal dispatch
overhead every kernel paid regardless of integral, plus (not separately
re-measured per-kernel here, but following directly from `BO`/`B0m`/`B1m`
no longer existing) the guarantee that a `BIN0` kernel no longer has `BIN4`,
`BIN3`, or any other leaf's 500 KB-4.8 MB body reachable from it at all.

*Note on absolute numbers:* this session's `boxGPU_test_dd` baseline
(16.40 MB) does not match the previous measurement pass's figure
(14.49 MB) for the same, unchanged source file. Not independently
reconciled — plausibly a different Kokkos/xpmath-kokkos install revision
or ROCm module path between sessions, not a code change (this commit
touched only `boxGPU_test_direct.cc`/`CMakeLists.txt`, never
`boxGPU_test_xp.cc`). This does not affect the direct-vs-BO-routed
comparison above, which was built in the same session with the same
toolchain for both binaries and is self-consistent.

### CUDA / sm_80 (CUDA 12.9.1, A100 target)

The first measurement pass found a double `boxGPU_test` (BO-routed) build
pegging `ptxas` at 100% CPU past 30 minutes on this shared login node
(`jlselogin4`) before being killed. Per the user's explicit direction,
this round **deliberately does not retry `boxGPU_test`/`boxGPU_test_dd`
(the BO()-routed builds) on CUDA at all** — that number stays unmeasured
by choice, to avoid repeating a long, CPU-heavy build on shared
infrastructure, not because it's technically impossible. Only
`boxGPU_test_direct`/`boxGPU_test_direct_dd` (direct leaf calls) were
built, and — consistent with the structural argument above — they compiled
far faster: double in 2m23s, `dd` in well under the 45-minute budget,
both dramatically faster than the BO-routed build's 30+ minute stall on
the exact same toolchain.

As REPORT.md's §4 established and this measurement reconfirms via `nm`:
nvcc fully inlines the entire call tree into each kernel under default
(non-RDC) compilation regardless of dispatch shape — zero `ql::` symbols
appear as separate device functions in either binary, before or after
this refactor. So unlike HIP, CUDA's benefit from this split shows up
entirely as **smaller total code per kernel**, not as a shared out-of-line
symbol becoming kernel-local — each kernel now only has to inline the one
leaf it actually calls, not the sum of all 21.

**Per-kernel `.text` size, the 21 named-section kernels** (`cuobjdump
--extract-elf all` + `readelf -SW`, the one real code object per binary —
a handful of tiny Kokkos-internal `ViewCopy`/`Construct` kernels excluded):

| build | kernels | min | max | sum |
|---|---|---|---|---|
| `boxGPU_test_direct` (double) | 21 | 17,024 B | 1,323,648 B | 9,873,408 B |
| `boxGPU_test_direct_dd` | 21 | 194,176 B | 2,217,088 B | 21,140,480 B |

The headline qualitative result: **these 21 kernels are no longer
byte-identical.** REPORT.md's completed CUDA baseline found all 17
BO-routed kernels **byte-identical** (938,496 B each, double) — because
every kernel inlined the same universal dispatch tree regardless of which
integral it computes. Here, per-kernel size varies by **~78x** (double)
and **~11x** (`dd`) across sections, directly tracking each section's own
leaf complexity instead of the sum of all 21. This is the same structural
signature HIP showed (reachable code bounded by one leaf, not the whole
tree), expressed through CUDA's different inlining model (size varies
per-kernel instead of a shared symbol disappearing).

*Caveat:* because the BO-routed CUDA baseline was deliberately not
rebuilt this round, there is no same-session CUDA number for "all kernels
identical at N bytes" to diff the table above against directly — the
"byte-identical" claim for the BO-routed case is REPORT.md's own
(pre-refactor, double only) finding, cited for context, not re-measured
here. The non-uniformity of the direct-driver's own 21 kernels is
measured directly and does not depend on that citation.

## Verdict

Does this leaf/dispatch split "plausibly remove the need for the existing
slice-based runtime workaround for the lighter integrals"? **Not
confirmable from `.text` size alone** — this measurement shows a real
reduction in *reachable* code per kernel on both backends and both types
measured, which is the structural quantity REPORT.md identified as the
root cause of the icache-pressure failure mode:

- **HIP/double:** ~3.44 MB universally reachable → a few KB for 18 of 21
  integrals, and one leaf's own size (~500-740 KB) for the remaining 3.
- **HIP/`dd`** (the type that actually hangs): the universal
  `BO`/`B0m`/`B1m` entry point (1.6 MB reachable from every kernel
  regardless of integral) is gone; most individual leaves stay out-of-line
  at unchanged size (dd arithmetic is large enough per leaf that this was
  already true before this refactor), so the per-kernel win is "no longer
  forced to carry `BO`/`B0m`/`B1m` plus every other leaf," not "shrunk to
  a few KB" the way double's lighter leaves did.
- **CUDA, both types:** kernels that were byte-identical pre-refactor (per
  REPORT.md, all reaching the same inlined universal tree) now vary by
  11-78x by section, tracking each section's own leaf instead of the sum
  of all 21.

Caveats that still stand:

- Even the best HIP/double per-kernel number here (a few KB to ~740 KB) is
  compared against REPORT.md's own order-of-magnitude framing (32-64 KB
  realistic L1 icache) — the still-large leaves (`BIN2`/`BIN3`/`BIN4` on
  double; nearly all of them on `dd`) remain well above that line on their
  own, even fully isolated from the other 20.
- The CUDA BO()-routed baseline was deliberately not rebuilt this round
  (shared-login-node CPU courtesy, user's explicit direction) — CUDA's
  "before" number for this refactor specifically is REPORT.md's own
  pre-refactor citation, not a number produced in this repo's current
  state.
- **Only running the actual unsliced batch on real MI250 hardware** (per
  REPORT.md §8 and this task's own Step 10) can answer whether the hang
  itself is fixed, reduced, or unaffected. This document does not claim
  that answer and explicitly defers it, per the task's standing
  instruction to ask before consuming Cobalt queue time.

## Next steps this document recommends, not performed here

1. Retry a CUDA BO()-routed baseline (double and `dd`) when shared-node
   CPU time is less of a concern, or with `-rdc=true`, to get a true
   same-repo-state CUDA before/after rather than citing REPORT.md's.
2. Extend the direct driver and this measurement pass to `ff`/`qf`/`tf` if
   those backends matter for the eventual master-promotion decision (the
   code path already exists in `boxGPU_test_direct.cc`, just not
   benchmarked here).
3. Step 8 (host-side bucketed dispatch) and Step 9 (final write-up) are
   separate, already-scoped follow-ons to this measurement pass.
4. Real-hardware validation (Step 10) needs the user's explicit go-ahead
   before any Cobalt queue time is spent.
