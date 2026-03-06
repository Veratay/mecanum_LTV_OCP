#include <cstdio>
#include <cmath>
#include <cstring>

extern "C" {
void dgemm_(const char* transa, const char* transb,
            const int* m, const int* n, const int* k,
            const double* alpha, const double* A, const int* lda,
            const double* B, const int* ldb,
            const double* beta, double* C, const int* ldc);
}

static bool has_nan(const double* x, int n) {
    for (int i = 0; i < n; i++)
        if (std::isnan(x[i])) return true;
    return false;
}

int main() {
    // Test: identity * identity = identity (6x6)
    double I6[36] = {};
    for (int i = 0; i < 6; i++) I6[i + 6*i] = 1.0;

    double C[36] = {};
    int m = 6, n = 6, k = 6;
    double alpha = 1.0, beta = 0.0;
    int lda = 6, ldb = 6, ldc = 6;

    std::printf("Test 1: I6 * I6 (6x6 x 6x6)\n");
    dgemm_("N", "N", &m, &n, &k, &alpha, I6, &lda, I6, &ldb, &beta, C, &ldc);
    std::printf("  NaN in result: %s\n", has_nan(C, 36) ? "YES" : "no");

    // Test: diagonal Q * dense Phi (6x6 x 6x6)
    double Q[36] = {};
    Q[0+6*0] = 10; Q[1+6*1] = 10; Q[2+6*2] = 1;
    Q[3+6*3] = 0.1; Q[4+6*4] = 0.1; Q[5+6*5] = 0.01;

    // Phi = A^k for some typical A (identity + small perturbation)
    double Phi[36] = {};
    for (int i = 0; i < 6; i++) Phi[i + 6*i] = 1.0;
    Phi[0 + 6*3] = 0.05;  // px += vx*dt
    Phi[1 + 6*4] = 0.05;  // py += vy*dt
    Phi[2 + 6*5] = 0.05;  // theta += omega*dt

    std::printf("Test 2: Q * Phi (6x6 x 6x6, diagonal Q)\n");
    std::memset(C, 0, sizeof(C));
    dgemm_("N", "N", &m, &n, &k, &alpha, Q, &lda, Phi, &ldb, &beta, C, &ldc);
    std::printf("  NaN in result: %s\n", has_nan(C, 36) ? "YES" : "no");
    for (int r = 0; r < 6; r++) {
        std::printf("  row %d:", r);
        for (int c = 0; c < 6; c++)
            std::printf(" %8.4f", C[r + 6*c]);
        std::printf("\n");
    }

    // Test: same dimensions but 6x4 (should work per previous findings)
    int n4 = 4;
    double B64[24] = {};
    for (int i = 0; i < 24; i++) B64[i] = 0.1 * (i + 1);
    double C64[24] = {};

    std::printf("Test 3: Q * B (6x6 x 6x4)\n");
    dgemm_("N", "N", &m, &n4, &k, &alpha, Q, &lda, B64, &ldb, &beta, C64, &ldc);
    std::printf("  NaN in result: %s\n", has_nan(C64, 24) ? "YES" : "no");

    // Test 4: repeated 6x6 calls (check if it's intermittent)
    std::printf("Test 4: 100 repeated Q*Phi calls\n");
    int nan_count = 0;
    for (int iter = 0; iter < 100; iter++) {
        std::memset(C, 0, sizeof(C));
        dgemm_("N", "N", &m, &n, &k, &alpha, Q, &lda, Phi, &ldb, &beta, C, &ldc);
        if (has_nan(C, 36)) nan_count++;
    }
    std::printf("  NaN count: %d/100\n", nan_count);

    // Test 5: 6x6 with lda=8 (panel-aligned)
    std::printf("Test 5: Q_padded * Phi_padded (lda=8)\n");
    double Q8[48] = {};  // 8x6
    double Phi8[48] = {};
    double C8[48] = {};
    for (int c = 0; c < 6; c++)
        for (int r = 0; r < 6; r++) {
            Q8[r + 8*c] = Q[r + 6*c];
            Phi8[r + 8*c] = Phi[r + 6*c];
        }
    int lda8 = 8;
    dgemm_("N", "N", &m, &n, &k, &alpha, Q8, &lda8, Phi8, &lda8, &beta, C8, &lda8);
    std::printf("  NaN in result: %s\n", has_nan(C8, 48) ? "YES" : "no");

    return 0;
}
