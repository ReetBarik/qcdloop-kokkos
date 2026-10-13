//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//          Stefan Hoeche   : shoeche@fnal.gov
//          Max Knobbe      : mknobbe@fnal.gov
//
// Meta-header: the generic, full-dispatch entry point for box integrals.
// General users include this header to get ql::BO(), which dispatches
// across all mass counts (see box/dispatch/BO.h for its implementation).
//
// For selective compilation, include individual dispatch group headers
// instead (or an individual box/leaves/*.h header for a single integral
// known statically):
//   box/dispatch/B0m.h  (0 internal masses: BIN0, B1-B5)
//   box/dispatch/B1m.h  (1 internal mass:   BIN1, B6-B10)
//   box/dispatch/B2m.h  (2 internal masses: BIN2, B11-B15)
//   box/dispatch/B3m.h  (3 internal masses: BIN3, B16)
//   box/dispatch/B4m.h  (4 internal masses: BIN4)

#pragma once

#include "box/dispatch/BO.h"
