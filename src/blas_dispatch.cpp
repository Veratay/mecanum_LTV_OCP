#include "blas_dispatch.h"

#include <cstring>

#ifdef MPC_USE_NEON
#include "neon_kernels.h"
#endif

// Fortran BLAS declarations (provided by BLASFEO with FORTRAN_BLAS_API=ON)
extern "C" {
void dgemv_(const char* trans, const int* m, const int* n,
            const double* alpha, const double* A, const int* lda,
            const double* x, const int* incx,
            const double* beta, double* y, const int* incy);

void dsymv_(const char* uplo, const int* n,
            const double* alpha, const double* A, const int* lda,
            const double* x, const int* incx,
            const double* beta, double* y, const int* incy);

void dgemm_(const char* transa, const char* transb,
            const int* m, const int* n, const int* k,
            const double* alpha, const double* A, const int* lda,
            const double* B, const int* ldb,
            const double* beta, double* C, const int* ldc);

void daxpy_(const int* n, const double* alpha,
            const double* x, const int* incx,
            double* y, const int* incy);

double ddot_(const int* n, const double* x, const int* incx,
             const double* y, const int* incy);

void dcopy_(const int* n, const double* x, const int* incx,
            double* y, const int* incy);

// BLASFEO doesn't provide dtrsv; use dtrsm with nrhs=1 instead
void dtrsm_(const char* side, const char* uplo, const char* transa,
            const char* diag, const int* m, const int* n,
            const double* alpha, const double* A, const int* lda,
            double* B, const int* ldb);
}

namespace mpc_linalg {

static const double ONE  = 1.0;
static const double ZERO = 0.0;
static const int    INC1 = 1;

void gemv(int m, int n, const double* A, const double* x, double* y) {
#ifdef MPC_USE_NEON
    if (n <= 120) {
        neon_gemv_colmajor(m, n, A, x, y);
        return;
    }
#endif
    dgemv_("N", &m, &n, &ONE, A, &m, x, &INC1, &ZERO, y, &INC1);
}

void gemv_trans(int m, int n, const double* A, const double* x, double* y) {
    dgemv_("T", &m, &n, &ONE, A, &m, x, &INC1, &ZERO, y, &INC1);
}

void trsv_lower(int n, const double* L, const double* b, double* y) {
#ifdef MPC_USE_NEON
    if (n <= 120) {
        neon_trsv_lower_colmajor(n, L, b, y);
        return;
    }
#endif
    std::memcpy(y, b, static_cast<std::size_t>(n) * sizeof(double));
    // dtrsm: solve L * Y = B where Y and B are n x 1
    int nrhs = 1;
    dtrsm_("L", "L", "N", "N", &n, &nrhs, &ONE, L, &n, y, &n);
}

void trsv_upper_trans(int n, const double* L, const double* y, double* x) {
#ifdef MPC_USE_NEON
    if (n <= 120) {
        neon_trsv_upper_trans_colmajor(n, L, y, x);
        return;
    }
#endif
    std::memcpy(x, y, static_cast<std::size_t>(n) * sizeof(double));
    // dtrsm: solve L' * X = Y where X and Y are n x 1
    int nrhs = 1;
    dtrsm_("L", "L", "T", "N", &n, &nrhs, &ONE, L, &n, x, &n);
}

void axpy(int n, double alpha, const double* x, double* y) {
    daxpy_(&n, &alpha, x, &INC1, y, &INC1);
}

double dot(int n, const double* x, const double* y) {
    return ddot_(&n, x, &INC1, y, &INC1);
}

void copy(int n, const double* x, double* y) {
    dcopy_(&n, x, &INC1, y, &INC1);
}

void gemm(int m, int n, int k, const double* A, const double* B, double* C) {
    dgemm_("N", "N", &m, &n, &k, &ONE, A, &m, B, &k, &ZERO, C, &m);
}

void gemm_full(int m, int n, int k, double alpha, const double* A, int lda,
               const double* B, int ldb, double beta, double* C, int ldc) {
    dgemm_("N", "N", &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

void gemm_atb(int m, int n, int k, const double* A, int lda,
              const double* B, int ldb, double* C, int ldc) {
    dgemm_("T", "N", &m, &n, &k, &ONE, A, &lda, B, &ldb, &ZERO, C, &ldc);
}

void gemm_atb_full(int m, int n, int k, double alpha, const double* A, int lda,
                   const double* B, int ldb, double beta, double* C, int ldc) {
    dgemm_("T", "N", &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

void symv(int n, const double* A, const double* x, double* y) {
    dsymv_("U", &n, &ONE, A, &n, x, &INC1, &ZERO, y, &INC1);
}

void symv_full(int n, double alpha, const double* A, const double* x,
               double beta, double* y) {
    dsymv_("U", &n, &alpha, A, &n, x, &INC1, &beta, y, &INC1);
}

void scal(int n, double alpha, double* x) {
    // BLASFEO doesn't provide dscal; simple loop
    for (int i = 0; i < n; ++i)
        x[i] *= alpha;
}

}  // namespace mpc_linalg
