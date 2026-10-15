#pragma once
// Shared launch policy for the GPU example drivers.
//
// History: Kokkos's RangePolicy picks a block size from N and the device's
// multiprocessor count alone (get_preferred_blocksize_for_range()); above
// N = 104*512 on gfx90a that heuristic chose 1024 threads/block, which the
// box kernels' register counts (148-192) could not satisfy, so the workgroup
// never got scheduled and Kokkos::fence() hung forever (see
// HANDOFF_mi250_box_hang_rootcause.md). That was first fixed by pinning
// Kokkos::LaunchBounds<256,1> on a RangePolicy.
//
// A later attempt replaced that with Kokkos::TeamPolicy + Kokkos::AUTO (no
// explicit LaunchBounds), on the premise that each backend's own occupancy
// logic would pick a schedulable size for the kernel actually being
// launched. That premise turned out to be false in a way that matters:
// reading Kokkos's own backend sources (Kokkos_HIP_BlockSize_Deduction.hpp,
// Kokkos_SYCL_TeamPolicy.hpp) shows AUTO is three different heuristics of
// very different quality. CUDA's is genuinely register-aware (it queries
// cudaFuncAttributes.numRegs against the real device register file). HIP's
// and SYCL's are not -- they reason only from shared-memory usage and, with
// no explicit LaunchBounds, HIP's compiles the kernel assuming it might be
// launched at the hardware's max block size (1024), which starves the
// compiler's register budget. Measured on gfx90a: the tf box kernels got
// 396 registers/thread under LaunchBounds<256,1> vs. only 212 under bare
// AUTO -- a big enough difference in compiled code that it pushed some
// already-delicate branch-cut regulator evaluations (kokkosMaths_xp_constants.h's
// kept_sign/1e-35 logic, needed for the float-word xpmath backends) into
// NaN for a meaningful fraction of BIN4 rows. Not a hang this time -- silently
// wrong physics output, which is worse. See NOTES_xpmath_source_changes.md /
// git history around this header for the full investigation.
//
// Kokkos's own SYCL backend reaches for the identical fix under the same
// circumstances: Kokkos_SYCL_TeamPolicy.hpp hardcodes a 256 register-safety
// fallback for SYCL-on-NVIDIA (marked "FIXME_SYCL Avoid requesting too many
// registers"), it just doesn't apply it everywhere it should (e.g. native
// Intel GPU targets get no such guard at all). So this isn't us picking a
// worse solution than what Kokkos offers -- Kokkos's own maintainers reach
// for the same conservative constant when a backend's AUTO can't be trusted.
// 256 here is not derived from gfx90a's specific register file size; it's a
// small, generically-safe compile-time ceiling (comfortably below the
// hardware block-size max on every current GPU backend Kokkos supports),
// used only to give the compiler a sane register budget to compile against.
// Kokkos::AUTO still picks the actual runtime team size within that ceiling,
// using whatever shared-memory/occupancy logic each backend provides.
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <string>

namespace ql {

using TeamPolicy =
    Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace, Kokkos::LaunchBounds<256, 1>>;
using TeamMember = TeamPolicy::member_type;

namespace detail {
// Not load-bearing for correctness -- just how many teams we ask for before
// Kokkos::AUTO picks a team size; the grid-stride loop below covers
// [begin, end) exactly once regardless of the team/league sizes actually
// chosen.
constexpr int kMaxLeagueSize = 1 << 16;
}

template <class F>
void launch_range(std::string const& label, int begin, int end, F const& f) {
    const int n = end - begin;
    if (n <= 0) return;
    const int league_size = std::max(1, std::min(n, detail::kMaxLeagueSize));
    Kokkos::parallel_for(
        label, TeamPolicy(league_size, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamMember& team) {
            const int team_size = team.team_size();
            const int stride = team.league_size() * team_size;
            const int start = begin + team.league_rank() * team_size + team.team_rank();
            for (int i = start; i < end; i += stride) {
                f(i);
            }
        });
}

template <class F>
void launch_range(int begin, int end, F const& f) {
    launch_range("ql::launch_range", begin, end, f);
}

} // namespace ql
