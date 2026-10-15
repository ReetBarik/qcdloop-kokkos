<table border="0" cellspacing="0" cellpadding="0">
<tr>
<td valign="middle"><img src="https://raw.githubusercontent.com/ReetBarik/qcdloop-kokkos/master/extra/logo.png" alt="Logo QCDLoop" /></td>
<td width="32">&nbsp;</td>
<td valign="middle"><img src="https://raw.githubusercontent.com/ReetBarik/qcdloop-kokkos/master/extra/kokkos_text.svg" alt="Kokkos" width="405" height="85" /></td>
</tr>
</table>

# QCDLoop + Kokkos

This repository is a [Kokkos](https://kokkos.org) port of the serial one-loop scalar Feynman integral library [QCDLoop](https://github.com/scarrazza/qcdloop). The integrals are the same tadpole, bubble, triangle, and box families. Each driver evaluates one integral on a batch of phase-space points in double precision. The serial library, its Fortran wrapper, and its releases stay in that repository. The library description is at <https://qcdloop.web.cern.ch>.

If you use these integrals in a publication, please cite [arXiv:0712.1851](http://arxiv.org/abs/0712.1851) and [arXiv:1605.03181](http://arxiv.org/abs/1605.03181). The original library is maintained by Stefano Carrazza.

## Download

```Shell
git clone https://github.com/ReetBarik/qcdloop-kokkos.git
```

## Installation

The build needs C++17, a Kokkos installation on `CMAKE_PREFIX_PATH`, and `libquadmath`. It always produces five double-precision drivers and does not install a library:

- `tadpoleGPU_test`
- `bubbleGPU_test`
- `triangleGPU_test`
- `boxGPU_test`
- `boxGPU_test_direct`

By default it also builds the four extended-precision box drivers described
below, which additionally need `xpmath_kokkos`
(<https://github.com/ReetBarik/xpmath-kokkos>) on `CMAKE_PREFIX_PATH` and a
Kokkos install `>= 5.1` (`xpmath_kokkos` itself requires it). For a
double-only build against an older Kokkos, or without `xpmath_kokkos`
installed, pass `-DQCDLOOP_ENABLE_XPMATH=OFF`: the five drivers above build
with zero dependency on `xpmath_kokkos` either way.

```Shell
# double-only, any Kokkos version, no xpmath_kokkos needed
cmake -S . -B build -DCMAKE_CXX_STANDARD=17 -DCMAKE_PREFIX_PATH=/path/to/kokkos \
      -DQCDLOOP_ENABLE_XPMATH=OFF
cmake --build build -j

# default: also builds the dd/ff/qf/tf box drivers, needs Kokkos >= 5.1
cmake -S . -B build -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_PREFIX_PATH="/path/to/kokkos;/path/to/xpmath-kokkos-install"
cmake --build build -j
```

When the Kokkos compiler is `hipcc` or `nvcc`, set `DD_HOST_CXX` to a host `g++` before configuring. `examples/quad_inputs.cc` includes `quadmath.h`, which those compilers reject, so CMake compiles that file with the host compiler and links the object in.

On [JLSE](https://www.jlse.anl.gov) the `scripts/prepare_*.sh` and `scripts/build_QCDLoops_Kokkos_*.sh` pairs load the modules and build both Kokkos (pinned to 5.1.0, so the xpmath drivers below are always available alongside the double ones) and these drivers. Source the prepare script, then the matching build script. Binaries land in `build_<arch>/`.

Each build script looks for an `xpmath_kokkos` install at
`$XPMATH_KOKKOS_PREFIX` (default `$HOME/xpmath-kokkos-install`) and passes
`-DQCDLOOP_ENABLE_XPMATH=ON` automatically when it finds one, `OFF`
otherwise — nothing to set by hand either way.

| Machine | Prepare | Build | Binaries |
|---|---|---|---|
| CPU (Kokkos Serial) | `scripts/prepare_cpu.sh` | `scripts/build_QCDLoops_Kokkos_cpu.sh` | `build_cpu/` |
| NVIDIA A100 | `scripts/prepare_a100.sh` | `scripts/build_QCDLoops_Kokkos_a100.sh` | `build_a100/` |
| AMD MI250 | `scripts/prepare_mi250.sh` | `scripts/build_QCDLoops_Kokkos_mi250.sh` | `build_mi250/` |
| Intel PVC | `scripts/prepare_pvc.sh` | `scripts/build_QCDLoops_Kokkos_pvc.sh` | `build_pvc/` |

## Usage

```Shell
./build/<topology>GPU_test <mode> [batch_size]
```

`<topology>` is `tadpole`, `bubble`, `triangle`, or `box`.

- `mode` is required. `0` prints timing rows (`Target Integral,Batch size,Time`). `1` prints one accuracy row per point (`Target Integral,Test ID,mu2,ms,ps,Coeff 1,Coeff 2,Coeff 3`).
- `batch_size` is optional and defaults to `1000000`. It is the number of phase-space points in the batch.

In accuracy mode, each coefficient is a decimal complex pair (`%.16e`).

## Extended precision

This branch also builds four box drivers on the same integral, one precision each. By default, configure needs `xpmath_kokkos` on `CMAKE_PREFIX_PATH` in addition to Kokkos (`>= 5.1`); pass `-DQCDLOOP_ENABLE_XPMATH=OFF` to skip them (see Installation above).

- `boxGPU_test_ff` — FloatFloat
- `boxGPU_test_tf` — TripleFloat
- `boxGPU_test_qf` — QuadFloat
- `boxGPU_test_dd` — DoubleDouble

They read the same quad kinematics as the double drivers. DoubleDouble keeps both limbs. FloatFloat, TripleFloat, and QuadFloat keep successive float words of that value.

```Shell
./build/boxGPU_test_<precision> <mode> [batch_size] [integral]
```

`<precision>` is `ff`, `tf`, `qf`, or `dd`. `integral` is optional. Each binary launches the whole batch in a single Kokkos `parallel_for` with explicit `LaunchBounds<256,1>` (`examples/ql_launch.h`) — a wider, default-bounds launch can pick an unschedulable workgroup size on some GPUs and hang. There is no batch-splitting at this level.

## Contact

- Reet Barik (rbarik@anl.gov)
- Taylor Childers (jchilders@anl.gov)
- Stefan Hoeche (shoeche@fnal.gov)
- Max Knobbe (mknobbe@fnal.gov)
