#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<USAGE
Usage: ${0##*/} [--ndk <path>] [--build-dir <dir>] [--api-level <level>]

Options:
  --ndk <path>        Path to the Android NDK root. Defaults to ANDROID_NDK_HOME/ROOT.
  --build-dir <dir>   Build directory to use. Defaults to build-android-a53.
  --api-level <lvl>   Android API level to target. Defaults to 24.
  -h, --help          Show this help message.

Configures and builds the mecanum MPC solver for an Android Cortex-A53 target
(FTC Control Hub) with arm64-v8a and BLASFEO NEON kernels.
USAGE
}

NDK_PATH=""
BUILD_DIR="build-android-a53"
ANDROID_API="24"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ndk)
            shift || { echo "Missing value for --ndk" >&2; exit 1; }
            NDK_PATH="$1"
            ;;
        --ndk=*)
            NDK_PATH="${1#*=}"
            ;;
        --build-dir)
            shift || { echo "Missing value for --build-dir" >&2; exit 1; }
            BUILD_DIR="$1"
            ;;
        --build-dir=*)
            BUILD_DIR="${1#*=}"
            ;;
        --api-level)
            shift || { echo "Missing value for --api-level" >&2; exit 1; }
            ANDROID_API="$1"
            ;;
        --api-level=*)
            ANDROID_API="${1#*=}"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift || true
done

# Resolve NDK path
if [[ -z "${NDK_PATH}" ]]; then
    if [[ -n "${ANDROID_NDK_HOME:-}" ]]; then
        NDK_PATH="${ANDROID_NDK_HOME}"
    elif [[ -n "${ANDROID_NDK_ROOT:-}" ]]; then
        NDK_PATH="${ANDROID_NDK_ROOT}"
    fi
fi

if [[ -z "${NDK_PATH}" ]]; then
    echo "Android NDK path not specified. Use --ndk or set ANDROID_NDK_HOME." >&2
    exit 1
fi

if [[ ! -d "${NDK_PATH}" ]]; then
    echo "Android NDK path does not exist: ${NDK_PATH}" >&2
    exit 1
fi

TOOLCHAIN_FILE="${NDK_PATH}/build/cmake/android.toolchain.cmake"
if [[ ! -f "${TOOLCHAIN_FILE}" ]]; then
    echo "android.toolchain.cmake not found under ${NDK_PATH}." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${BUILD_DIR}" != /* ]]; then
    BUILD_DIR="${REPO_ROOT}/${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"

if command -v nproc >/dev/null 2>&1; then
    PARALLEL_JOBS="$(nproc)"
else
    PARALLEL_JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
fi

OPT_FLAGS="-O3 -mcpu=cortex-a53 -mtune=cortex-a53 -fomit-frame-pointer -fdata-sections -ffunction-sections -fvisibility=hidden -pipe -flto"
C_FLAGS="${OPT_FLAGS}"
CXX_FLAGS="${OPT_FLAGS} -fvisibility-inlines-hidden"

cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM="android-${ANDROID_API}" \
    -DANDROID_STL=c++_static \
    -DANDROID_ARM_NEON=TRUE \
    -DANDROID_PIE=TRUE \
    -DCMAKE_BUILD_TYPE=Release \
    -DBLASFEO_TARGET=ARMV8A_ARM_CORTEX_A53 \
    -DMPC_USE_NEON=ON \
    -DCMAKE_C_FLAGS="${C_FLAGS}" \
    -DCMAKE_CXX_FLAGS="${CXX_FLAGS}"

cmake --build "${BUILD_DIR}" --parallel "${PARALLEL_JOBS}"

cat <<INFO

Build completed. Android arm64-v8a artifacts in:
  ${BUILD_DIR}
INFO
