################################################################
# QCDLoop + Kokkos build — AMD MI250 (HIP, gfx90a)
#
# Usage: source scripts/prepare_mi250.sh
#        source scripts/build_QCDLoops_Kokkos_mi250.sh <repo-root>
#
# <repo-root> must contain QCDLoop's CMakeLists.txt (the build is
# configured against it). Defaults to $PWD. Kokkos is cloned into
# <repo-root>/kokkos; its build/install trees and the QCDLoop build
# dir are all suffixed with "mi250" so several architectures can share
# one checkout without colliding.
#
# The --gcc-toolchain flag never reached cmake in the old script (see commit msg).
################################################################

ARCH_TAG=mi250

export TARGET_DIR=$1
if [ "$#" -ne 1 ]; then
   export TARGET_DIR=$(pwd -LP)
fi
START_DIR=$(pwd -LP)
echo "Installing in path: $TARGET_DIR  (arch: $ARCH_TAG)"

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR" || return 1

export LOGDIR=$TARGET_DIR/build_logs
mkdir -p "$LOGDIR"

##############
## SETTINGS ##
##############

# Compiler
CC=$(which hipcc)
CXX=$(which hipcc)
if [ -z "$CC" ] || [ -z "$CXX" ]; then
   echo "ERROR: hipcc/hipcc not on PATH — did you source scripts/prepare_mi250.sh?" >&2
   return 1
fi

# MPI Related settings
MPI_ENABLED=0
MPI_CC=NONE
MPI_CXX=NONE
if which mpicxx > /dev/null 2>&1; then
   echo Enabling MPI
   MPI_ENABLED=1
   MPI_CC=$(which mpicc)
   MPI_CXX=$(which mpicxx)
fi

# KOKKOS Related settings
#
# 5.1.0, not 4.7.01: the dd/ff/qf/tf xpmath box backends link
# xpmath_kokkos (https://github.com/ReetBarik/xpmath-kokkos), which
# requires Kokkos >= 5.1 (find_dependency(Kokkos 5.1 REQUIRED) in its
# CMake config). Kokkos >= 5.1 itself requires C++20, so the Kokkos
# build below uses CMAKE_CXX_STANDARD=20 -- that is Kokkos's own
# build standard, separate from QCDLoop's, which stays C++17 (set via
# -DCMAKE_CXX_STANDARD=17 in the QCDLoop cmake invocation further down).
KOKKOS_TAG=5.1.0
KOKKOS_BUILD=Release
KOKKOS_URL=https://github.com/kokkos/kokkos.git
KOKKOS_ENABLED=Kokkos_ENABLE_HIP
KOKKOS_ARCH_FLAG=Kokkos_ARCH_VEGA90A

# Backend-specific extras. Kept as an array so values containing spaces or
# "=" survive; the old single-string form word-split incorrectly.
EXTRA_FLAGS=(
   -DCMAKE_CXX_COMPILER="$CXX"
   -DCMAKE_CXX_FLAGS=--gcc-toolchain=/soft/compilers/gcc/13.3.0/x86_64-suse-linux
)

# xpmath_kokkos is header-only and only needs some Kokkos >= 5.1 on
# CMAKE_PREFIX_PATH at consume time, so one install serves every
# architecture. Point XPMATH_KOKKOS_PREFIX at it to also configure the
# dd/ff/qf/tf box backends; leave it unset/missing for a double-only
# build (QCDLoop's CMakeLists.txt then skips find_package(xpmath_kokkos)
# entirely via -DQCDLOOP_ENABLE_XPMATH=OFF, set automatically below).
XPMATH_KOKKOS_PREFIX=${XPMATH_KOKKOS_PREFIX:-$HOME/xpmath-kokkos-install}

KOKKOS_SUBDIR=kokkos-$KOKKOS_TAG-$ARCH_TAG/$KOKKOS_BUILD

####################
## install Kokkos ##
####################
echo "Installing Kokkos BACKEND=$KOKKOS_ENABLED ARCH=$KOKKOS_ARCH_FLAG"

KOKKOS_CMAKE_ARGS=(
   -DCMAKE_INSTALL_PREFIX="install/$KOKKOS_SUBDIR"
   -DCMAKE_BUILD_TYPE="$KOKKOS_BUILD"
   -DCMAKE_CXX_STANDARD=20
   -D$KOKKOS_ENABLED=ON
)
if [ "$KOKKOS_ARCH_FLAG" != "NONE" ]; then
   KOKKOS_CMAKE_ARGS+=( -D$KOKKOS_ARCH_FLAG=ON )
fi
KOKKOS_CMAKE_ARGS+=( "${EXTRA_FLAGS[@]}" )

{
   { [ -d kokkos ] || git clone "$KOKKOS_URL" -b "$KOKKOS_TAG"; } &&
   cd kokkos &&
   cmake -S . -B "build/$KOKKOS_SUBDIR" "${KOKKOS_CMAKE_ARGS[@]}" &&
   make -C "build/$KOKKOS_SUBDIR" -j install
} 1> "$LOGDIR/kokkos.$ARCH_TAG.stdout.txt" 2> "$LOGDIR/kokkos.$ARCH_TAG.stderr.txt"
KOKKOS_RC=$?
if [ $KOKKOS_RC -ne 0 ]; then
   echo "ERROR: Kokkos build failed (exit $KOKKOS_RC)." >&2
   echo "       see $LOGDIR/kokkos.$ARCH_TAG.stderr.txt" >&2
   cd "$START_DIR"
   return $KOKKOS_RC
fi

cd "$TARGET_DIR/kokkos" || return 1
KOKKOS_HOME=$PWD/install/$KOKKOS_SUBDIR

if [ -f "$XPMATH_KOKKOS_PREFIX/lib64/cmake/xpmath_kokkos/xpmath_kokkosConfig.cmake" ]; then
   QCDLOOP_ENABLE_XPMATH=ON
   XPMATH_PREFIX_PATH_ENTRY="$XPMATH_KOKKOS_PREFIX:"
   echo "xpmath_kokkos found at $XPMATH_KOKKOS_PREFIX -- building dd/ff/qf/tf box backends too"
else
   QCDLOOP_ENABLE_XPMATH=OFF
   XPMATH_PREFIX_PATH_ENTRY=""
   echo "xpmath_kokkos not found at $XPMATH_KOKKOS_PREFIX -- double-only build"
fi

{
   echo "export KOKKOS_HOME=$KOKKOS_HOME"
   echo "export CMAKE_PREFIX_PATH=$KOKKOS_HOME/lib64/cmake/Kokkos:${XPMATH_PREFIX_PATH_ENTRY}\$CMAKE_PREFIX_PATH"
   echo 'export CPATH=$KOKKOS_HOME/include:$CPATH'
   echo 'export PATH=$KOKKOS_HOME/bin:$PATH'
   echo 'export LD_LIBRARY_PATH=$KOKKOS_HOME/lib64:$LD_LIBRARY_PATH'
} > setup.$ARCH_TAG.sh
source setup.$ARCH_TAG.sh

######################
## install QCDLoops ##
######################
cd "$TARGET_DIR" || return 1
ulimit -s 131072
export LD_LIBRARY_PATH=$TARGET_DIR/build_$ARCH_TAG:$LD_LIBRARY_PATH
mkdir -p "build_$ARCH_TAG"
cd "build_$ARCH_TAG" || return 1
cmake -DCMAKE_INSTALL_PREFIX="$TARGET_DIR" \
      -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_C_COMPILER="$CC" \
      -DCMAKE_CXX_COMPILER="$CXX" \
      -DCMAKE_CXX_FLAGS="-g" \
      -DQCDLOOP_ENABLE_XPMATH="$QCDLOOP_ENABLE_XPMATH" \
      .. &&
make -j32
QCDLOOP_RC=$?
cd "$TARGET_DIR"
if [ $QCDLOOP_RC -ne 0 ]; then
   echo "ERROR: QCDLoop build failed (exit $QCDLOOP_RC)." >&2
   return $QCDLOOP_RC
fi
echo "Done. Binaries in $TARGET_DIR/build_$ARCH_TAG"
