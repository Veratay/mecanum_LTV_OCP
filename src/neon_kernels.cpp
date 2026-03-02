#ifdef MPC_USE_NEON

#include "neon_kernels.h"
#include <arm_neon.h>

// ---------------------------------------------------------------------------
// y = A * x,  A is m x n column-major
// A[i + m*j] is element (i,j).
// y[i] = sum_j A[i + m*j] * x[j]
//
// We use a column-oriented approach: for each column j, scatter-add
// A[:,j] * x[j] into y.  This gives unit-stride access to A.
// ---------------------------------------------------------------------------
void neon_gemv_colmajor(int m, int n, const double* A, const double* x, double* y)
{
    // Zero out y
    {
        int i = 0;
        for (; i + 1 < m; i += 2) {
            vst1q_f64(y + i, vdupq_n_f64(0.0));
        }
        for (; i < m; ++i) {
            y[i] = 0.0;
        }
    }

    // Accumulate column by column
    for (int j = 0; j < n; ++j) {
        const double* col = A + (long)m * j;
        float64x2_t xj = vdupq_n_f64(x[j]);
        int i = 0;
        for (; i + 1 < m; i += 2) {
            float64x2_t yi = vld1q_f64(y + i);
            float64x2_t ai = vld1q_f64(col + i);
            yi = vfmaq_f64(yi, ai, xj);
            vst1q_f64(y + i, yi);
        }
        for (; i < m; ++i) {
            y[i] += col[i] * x[j];
        }
    }
}

// ---------------------------------------------------------------------------
// Solve L*y = b  (forward substitution)
// L is n x n lower triangular, column-major: L[i,j] = L[i + n*j], j <= i.
//
// Column-oriented forward substitution:
//   Copy b into y.
//   For j = 0 .. n-1:
//     y[j] /= L[j + n*j]
//     y[i] -= L[i + n*j] * y[j]   for i = j+1 .. n-1
//
// The inner update is a unit-stride AXPY on column j of L, which is
// SIMD-friendly.
// ---------------------------------------------------------------------------
void neon_trsv_lower_colmajor(int n, const double* L, const double* b, double* y)
{
    // Copy b into y
    for (int i = 0; i < n; ++i) {
        y[i] = b[i];
    }

    for (int j = 0; j < n; ++j) {
        const double* Lj = L + (long)n * j;  // column j of L
        y[j] /= Lj[j];                       // L[j,j]

        double yj_val = y[j];
        float64x2_t yj = vdupq_n_f64(yj_val);

        int i = j + 1;
        // Align i to even index if needed for clean SIMD
        if ((i & 1) && i < n) {
            y[i] -= Lj[i] * yj_val;
            ++i;
        }
        for (; i + 1 < n; i += 2) {
            float64x2_t yi  = vld1q_f64(y + i);
            float64x2_t lij = vld1q_f64(Lj + i);
            yi = vfmsq_f64(yi, lij, yj);   // yi -= lij * yj
            vst1q_f64(y + i, yi);
        }
        for (; i < n; ++i) {
            y[i] -= Lj[i] * yj_val;
        }
    }
}

// ---------------------------------------------------------------------------
// Solve L^T * x = y  (backward substitution with transposed lower factor)
// L is n x n lower triangular, column-major.
// L^T[i,j] = L[j,i] = L[j + n*i].  L^T is upper triangular.
//
// Column-oriented backward substitution on L^T:
//   Copy y into x.
//   For j = n-1 .. 0:
//     x[j] /= L[j + n*j]          (diagonal of L^T is same as L)
//     x[i] -= L[j + n*i] * x[j]   for i = 0 .. j-1
//
// But L[j + n*i] for varying i with fixed j is stride-n (bad for SIMD).
//
// Alternative: row-oriented backward substitution (standard upper trsv):
//   Copy y into x.
//   For i = n-1 .. 0:
//     sum = 0
//     for j = i+1 .. n-1: sum += L^T[i,j] * x[j] = L[j + n*i] * x[j]
//     x[i] = (y[i] - sum) / L[i + n*i]
//
// Here L[j + n*i] for varying j with fixed i is column i of L starting at
// row j -- this IS unit-stride in memory!  So we vectorize over j.
// ---------------------------------------------------------------------------
void neon_trsv_upper_trans_colmajor(int n, const double* L, const double* y, double* x)
{
    // Copy y into x
    for (int i = 0; i < n; ++i) {
        x[i] = y[i];
    }

    for (int i = n - 1; i >= 0; --i) {
        // Accumulate sum = sum_j L[j + n*i] * x[j] for j = i+1..n-1
        // L column i starts at L + n*i.  We read L[j + n*i] for j = i+1..n-1.
        const double* Li = L + (long)n * i;  // column i of L

        float64x2_t acc = vdupq_n_f64(0.0);
        double sum = 0.0;

        int j = i + 1;
        // Align j to even if needed
        if ((j & 1) && j < n) {
            sum += Li[j] * x[j];
            ++j;
        }
        for (; j + 1 < n; j += 2) {
            float64x2_t lj = vld1q_f64(Li + j);
            float64x2_t xj = vld1q_f64(x + j);
            acc = vfmaq_f64(acc, lj, xj);
        }
        for (; j < n; ++j) {
            sum += Li[j] * x[j];
        }

        // Horizontal add the NEON accumulator
        sum += vaddvq_f64(acc);

        x[i] = (x[i] - sum) / Li[i];
    }
}

// ---------------------------------------------------------------------------
// y = clip(x, lo, hi), returns number of elements that were clipped.
// ---------------------------------------------------------------------------
int neon_clip_and_count(int n, const double* x, double lo, double hi, double* y)
{
    float64x2_t vlo = vdupq_n_f64(lo);
    float64x2_t vhi = vdupq_n_f64(hi);
    int count = 0;

    int i = 0;
    for (; i + 1 < n; i += 2) {
        float64x2_t xi = vld1q_f64(x + i);
        float64x2_t clamped = vmaxq_f64(vminq_f64(xi, vhi), vlo);
        vst1q_f64(y + i, clamped);

        // Compare to detect clipping: if clamped != xi, element was clipped
        uint64x2_t eq = vceqq_f64(clamped, xi);
        // eq lanes are all-ones if equal, all-zeros if not
        // Count not-equal lanes
        uint64_t lane0 = vgetq_lane_u64(eq, 0);
        uint64_t lane1 = vgetq_lane_u64(eq, 1);
        if (lane0 == 0) ++count;
        if (lane1 == 0) ++count;
    }
    for (; i < n; ++i) {
        double v = x[i];
        if (v < lo) { y[i] = lo; ++count; }
        else if (v > hi) { y[i] = hi; ++count; }
        else { y[i] = v; }
    }

    return count;
}

#endif // MPC_USE_NEON
