#pragma once

#include "mpc_types.h"

// Solve box-constrained QP: min 0.5 U' H U + g' U  s.t.  V_min <= U <= V_max
// H is n x n SPD, L is its Cholesky factor, g is the gradient vector
// U_warm is the warm-start (typically unconstrained solution)
// Result written to workspace.U, returns number of iterations
int box_qp_solve(const double* H, const double* L, const double* g,
                 double V_min, double V_max, int n, int max_iter,
                 BoxQPWorkspace& workspace);

// Unconstrained solve: U = -H^{-1} g via Cholesky
void unconstrained_solve(const double* L, const double* g, int n, double* U);

// Check if U is feasible (all elements in [V_min, V_max])
bool is_feasible(const double* U, int n, double V_min, double V_max);

// Clip U to [V_min, V_max], return number of clipped elements
int clip_to_bounds(double* U, int n, double V_min, double V_max);
