#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>

#include "cholesky.h"
#include "blas_dispatch.h"

// ---------------------------------------------------------------------------
// Simple deterministic LCG pseudo-random number generator
// ---------------------------------------------------------------------------
static uint64_t lcg_state = 0;

static void lcg_seed(uint64_t seed) { lcg_state = seed; }

// Returns a double in (-1, 1)
static double lcg_rand() {
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    // Take upper 32 bits, map to (-1, 1)
    int32_t s = static_cast<int32_t>(lcg_state >> 33);
    return static_cast<double>(s) / static_cast<double>(1LL << 31);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static int g_tests_run    = 0;
static int g_tests_passed = 0;

static void check(bool cond, const char* name) {
    g_tests_run++;
    if (cond) {
        g_tests_passed++;
        std::printf("  [PASS] %s\n", name);
    } else {
        std::printf("  [FAIL] %s\n", name);
    }
}

// Max absolute element of a vector
static double max_abs(int len, const double* v) {
    double m = 0.0;
    for (int i = 0; i < len; i++) {
        double a = std::fabs(v[i]);
        if (a > m) m = a;
    }
    return m;
}

// Build an n x n SPD matrix: A = B' * B + eps * I
// B is n x n with pseudo-random entries.  A and B must be pre-allocated.
static void make_spd(int n, double* A, double* B, double eps) {
    // Fill B with random values
    for (int i = 0; i < n * n; i++) B[i] = lcg_rand();

    // A = B' * B  (column-major: A_ij = sum_k B_ki * B_kj)
    mpc_linalg::gemm_atb(n, n, n, B, n, B, n, A, n);

    // A += eps * I
    for (int i = 0; i < n; i++) A[i + i * n] += eps;
}

// Compute max |L*L' - A| for lower-triangular L (n x n column-major)
static double reconstruct_error(int n, const double* L, const double* A) {
    // Compute C = L * L' using gemm.
    // L' stored explicitly in a temporary, then call gemm.
    double* Lt = static_cast<double*>(std::calloc(n * n, sizeof(double)));
    double* C  = static_cast<double*>(std::calloc(n * n, sizeof(double)));

    // Build L' (upper triangular) from L (lower triangular)
    for (int j = 0; j < n; j++)
        for (int i = 0; i <= j; i++)
            Lt[i + j * n] = L[j + i * n];   // Lt(i,j) = L(j,i)

    mpc_linalg::gemm(n, n, n, L, Lt, C);

    // Difference
    double maxerr = 0.0;
    for (int i = 0; i < n * n; i++) {
        double e = std::fabs(C[i] - A[i]);
        if (e > maxerr) maxerr = e;
    }

    std::free(Lt);
    std::free(C);
    return maxerr;
}

// ---------------------------------------------------------------------------
// Test: factor + reconstruct for a given size
// ---------------------------------------------------------------------------
static void test_factor_reconstruct(int n) {
    char label[128];
    double* A = static_cast<double*>(std::calloc(n * n, sizeof(double)));
    double* B = static_cast<double*>(std::calloc(n * n, sizeof(double)));
    double* L = static_cast<double*>(std::calloc(n * n, sizeof(double)));

    make_spd(n, A, B, 1e-3);

    int ret = cholesky_factor(n, A, L);
    std::snprintf(label, sizeof(label), "cholesky_factor returns 0 (n=%d)", n);
    check(ret == 0, label);

    double err = reconstruct_error(n, L, A);
    std::snprintf(label, sizeof(label),
                  "L*L' == A  max_err=%.3e (n=%d)", err, n);
    check(err < 1e-12, label);

    std::free(A);
    std::free(B);
    std::free(L);
}

// ---------------------------------------------------------------------------
// Test: solve A*x = b, verify residual
// ---------------------------------------------------------------------------
static void test_solve(int n) {
    char label[128];
    double* A = static_cast<double*>(std::calloc(n * n, sizeof(double)));
    double* B = static_cast<double*>(std::calloc(n * n, sizeof(double)));
    double* L = static_cast<double*>(std::calloc(n * n, sizeof(double)));
    double* b = static_cast<double*>(std::calloc(n, sizeof(double)));
    double* x = static_cast<double*>(std::calloc(n, sizeof(double)));
    double* Ax = static_cast<double*>(std::calloc(n, sizeof(double)));
    double* res = static_cast<double*>(std::calloc(n, sizeof(double)));

    make_spd(n, A, B, 1e-3);
    for (int i = 0; i < n; i++) b[i] = lcg_rand();

    int ret = cholesky_factor(n, A, L);
    check(ret == 0, "factor for solve test");

    cholesky_solve(n, L, b, x);

    // Ax = A * x
    mpc_linalg::gemv(n, n, A, x, Ax);

    // res = Ax - b
    for (int i = 0; i < n; i++) res[i] = Ax[i] - b[i];

    double err = max_abs(n, res);
    std::snprintf(label, sizeof(label),
                  "solve residual max|A*x - b|=%.3e (n=%d)", err, n);
    check(err < 1e-10, label);

    std::free(A);
    std::free(B);
    std::free(L);
    std::free(b);
    std::free(x);
    std::free(Ax);
    std::free(res);
}

// ---------------------------------------------------------------------------
// Test: in-place factor + in-place solve
// ---------------------------------------------------------------------------
static void test_inplace(int n) {
    char label[128];
    double* A_orig = static_cast<double*>(std::calloc(n * n, sizeof(double)));
    double* A_work = static_cast<double*>(std::calloc(n * n, sizeof(double)));
    double* B      = static_cast<double*>(std::calloc(n * n, sizeof(double)));
    double* b      = static_cast<double*>(std::calloc(n, sizeof(double)));
    double* x      = static_cast<double*>(std::calloc(n, sizeof(double)));
    double* Ax     = static_cast<double*>(std::calloc(n, sizeof(double)));
    double* res    = static_cast<double*>(std::calloc(n, sizeof(double)));

    make_spd(n, A_orig, B, 1e-3);
    for (int i = 0; i < n; i++) b[i] = lcg_rand();

    // Copy A for in-place factorization
    std::memcpy(A_work, A_orig, n * n * sizeof(double));

    int ret = cholesky_factor_inplace(n, A_work);
    std::snprintf(label, sizeof(label),
                  "cholesky_factor_inplace returns 0 (n=%d)", n);
    check(ret == 0, label);

    // Zero the strict upper triangle (dpotrf leaves it untouched)
    for (int j = 1; j < n; j++)
        for (int i = 0; i < j; i++)
            A_work[i + j * n] = 0.0;

    // Verify L*L' = A_orig using the lower triangle now stored in A_work
    double err = reconstruct_error(n, A_work, A_orig);
    std::snprintf(label, sizeof(label),
                  "inplace L*L' == A  max_err=%.3e (n=%d)", err, n);
    check(err < 1e-12, label);

    // In-place solve: x_inout starts as b, ends as x
    std::memcpy(x, b, n * sizeof(double));
    cholesky_solve_inplace(n, A_work, x);

    mpc_linalg::gemv(n, n, A_orig, x, Ax);
    for (int i = 0; i < n; i++) res[i] = Ax[i] - b[i];

    err = max_abs(n, res);
    std::snprintf(label, sizeof(label),
                  "inplace solve residual max|A*x - b|=%.3e (n=%d)", err, n);
    check(err < 1e-10, label);

    std::free(A_orig);
    std::free(A_work);
    std::free(B);
    std::free(b);
    std::free(x);
    std::free(Ax);
    std::free(res);
}

// ---------------------------------------------------------------------------
// Test: factoring a non-SPD matrix returns nonzero
// ---------------------------------------------------------------------------
static void test_non_spd() {
    const int n = 6;
    double A[36];
    double L[36];

    // Start with identity
    std::memset(A, 0, sizeof(A));
    for (int i = 0; i < n; i++) A[i + i * n] = 1.0;

    // Make one diagonal negative -> not positive definite
    A[2 + 2 * n] = -1.0;

    int ret = cholesky_factor(n, A, L);
    check(ret != 0, "cholesky_factor returns nonzero for non-SPD matrix");

    // Also test in-place variant
    double A2[36];
    std::memset(A2, 0, sizeof(A2));
    for (int i = 0; i < n; i++) A2[i + i * n] = 1.0;
    A2[2 + 2 * n] = -1.0;

    ret = cholesky_factor_inplace(n, A2);
    check(ret != 0, "cholesky_factor_inplace returns nonzero for non-SPD matrix");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::printf("=== Cholesky factorization and solve tests ===\n\n");

    // Use a fixed seed for reproducibility
    lcg_seed(42);

    std::printf("--- Factor + reconstruct (6x6) ---\n");
    test_factor_reconstruct(6);

    std::printf("\n--- Factor + reconstruct (20x20) ---\n");
    test_factor_reconstruct(20);

    std::printf("\n--- Solve A*x = b (6x6) ---\n");
    test_solve(6);

    std::printf("\n--- Solve A*x = b (20x20) ---\n");
    test_solve(20);

    std::printf("\n--- In-place factor + solve (6x6) ---\n");
    test_inplace(6);

    std::printf("\n--- In-place factor + solve (20x20) ---\n");
    test_inplace(20);

    std::printf("\n--- Non-SPD detection ---\n");
    test_non_spd();

    std::printf("\n=== Summary: %d / %d tests passed ===\n",
                g_tests_passed, g_tests_run);

    if (g_tests_passed == g_tests_run) {
        std::printf("All Cholesky tests passed\n");
        return 0;
    } else {
        std::printf("SOME TESTS FAILED\n");
        return 1;
    }
}
