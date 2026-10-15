#pragma once
// Shared launch-bounds policy for the GPU example drivers.
//
// Kokkos picks a RangePolicy's block size from N and the device's
// multiprocessor count alone (get_preferred_blocksize_for_range()); above
// N = 104*512 on gfx90a that heuristic chooses 1024 threads/block, which the
// box kernels' register counts (148-192) cannot satisfy, so the workgroup
// never gets scheduled and Kokkos::fence() hangs forever. Pinning explicit
// LaunchBounds bypasses the heuristic and makes the compiler stamp a
// satisfiable .max_flat_workgroup_size instead. Launch configuration only;
// it does not change what any kernel computes. See
// HANDOFF_mi250_box_hang_rootcause.md for the full investigation.
#include <Kokkos_Core.hpp>

namespace ql {

using LaunchPolicy =
    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::LaunchBounds<256, 1>>;

template <class F>
void launch_range(int begin, int end, F const& f) {
    Kokkos::parallel_for(LaunchPolicy(begin, end), f);
}

} // namespace ql
