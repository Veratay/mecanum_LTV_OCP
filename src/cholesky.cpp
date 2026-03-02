#include "cholesky.h"
#include "blas_dispatch.h"
#include <cstring>

extern "C" {
    void dpotrf_(const char* uplo, const int* n, double* A, const int* lda, int* info);
    void dpotrs_(const char* uplo, const int* n, const int* nrhs, const double* A,
                 const int* lda, double* B, const int* ldb, int* info);
}

int cholesky_factor(int n, const double* A, double* L_out) {
    // Copy A to L_out
    mpc_linalg::copy(n * n, A, L_out);

    // Compute Cholesky factorization (lower triangular)
    int info = 0;
    dpotrf_("L", &n, L_out, &n, &info);

    // Zero out the strict upper triangle for cleanliness
    for (int j = 1; j < n; ++j) {
        for (int i = 0; i < j; ++i) {
            L_out[i + j * n] = 0.0;
        }
    }

    return info;
}

void cholesky_solve(int n, const double* L, const double* b, double* x) {
    // Copy b to x
    mpc_linalg::copy(n, b, x);

    // Solve L * L' * x = b using the precomputed factor
    int one = 1;
    int info = 0;
    dpotrs_("L", &n, &one, L, &n, x, &n, &info);
}

int cholesky_factor_inplace(int n, double* A) {
    int info = 0;
    dpotrf_("L", &n, A, &n, &info);
    return info;
}

void cholesky_solve_inplace(int n, const double* L, double* x_inout) {
    int one = 1;
    int info = 0;
    dpotrs_("L", &n, &one, L, &n, x_inout, &n, &info);
}
