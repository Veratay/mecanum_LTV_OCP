#pragma once

// Cholesky factorization: A = L * L'
// A is n x n symmetric positive definite, column-major
// L is written to the lower triangle of L_out (n x n column-major)
// Returns 0 on success, positive value k if leading minor k is not positive definite
int cholesky_factor(int n, const double* A, double* L_out);

// Solve A * x = b using precomputed Cholesky factor L (A = L * L')
// b is input, x is output (both length n)
void cholesky_solve(int n, const double* L, const double* b, double* x);

// In-place Cholesky factorization (overwrites lower triangle of A)
// Returns 0 on success
int cholesky_factor_inplace(int n, double* A);

// Solve using in-place factor: L * L' * x = b
// Uses LAPACK dpotrs. x_inout is b on input, x on output.
void cholesky_solve_inplace(int n, const double* L, double* x_inout);
