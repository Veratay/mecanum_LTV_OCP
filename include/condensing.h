#pragma once

#include "mpc_types.h"

// Build state transition matrices Phi[k] and input-to-state matrices Gamma
// Phi_blocks: array of (N+1) matrices, each NX*NX, column-major
// Gamma: ((N+1)*NX) x (N*NU) matrix, block lower-triangular, column-major
void build_prediction_matrices(const double* A_list, const double* B_list, int N,
                               double* Phi_blocks, double* Gamma);

// Form the condensed Hessian: H = Gamma' * Q_bar * Gamma + R_bar
// H is (N*NU) x (N*NU), column-major
// Exploits block-diagonal structure of Q_bar
void form_hessian(const double* Gamma, const double* Q, const double* Qf,
                  const double* R, int N, double* H);

// Form gradient matrices: g = F * e0 + f_const
// F is (N*NU) x NX, column-major
// f_const is (N*NU) vector
void form_gradient_matrices(const double* Gamma, const double* Phi_blocks,
                            const double* Q, const double* Qf,
                            const double* x_ref_consistent, int N,
                            double* F, double* f_const);

// Full condensation pipeline: given A_list, B_list, config, and consistent reference,
// produce a PrecomputedWindow
void condense_window(const double* A_list, const double* B_list,
                     const double* x_ref_consistent, const double* u_ref,
                     const MPCConfig& config, PrecomputedWindow& window);
