#pragma once

#ifdef MPC_USE_NEON

// y = A * x,  A is m x n column-major
void neon_gemv_colmajor(int m, int n, const double* A, const double* x, double* y);

// Solve L*y = b, L is n x n lower triangular column-major
void neon_trsv_lower_colmajor(int n, const double* L, const double* b, double* y);

// Solve L'*x = y, L is n x n lower triangular column-major
void neon_trsv_upper_trans_colmajor(int n, const double* L, const double* y, double* x);

// y = clip(x, lo, hi), returns number of clipped elements
int neon_clip_and_count(int n, const double* x, double lo, double hi, double* y);

#endif
