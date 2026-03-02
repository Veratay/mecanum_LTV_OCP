#pragma once

#include "mpc_types.h"

// Cubic Hermite heading interpolation
// Returns theta at time t given boundary conditions at t_k and t_k1
double theta_interp(double t, double t_k, double t_k1,
                    double theta_k, double omega_k,
                    double theta_k1, double omega_k1);

// Derivative of the cubic Hermite interpolant (for verification)
double omega_interp(double t, double t_k, double t_k1,
                    double theta_k, double omega_k,
                    double theta_k1, double omega_k1);

// Interpolation data passed to the ODE integrator
struct InterpData {
    double t_k, t_k1;
    double theta_k, omega_k;
    double theta_k1, omega_k1;
    const ModelParams* params;
};

// Augmented ODE RHS: Psi_dot = M(t) * Psi
// Psi is N_AUG x N_AUG (10x10), column-major
// Uses structured multiplication (only top 6 rows of output are nonzero)
void augmented_rhs(double t, const double* Psi, const InterpData& interp,
                   double* Psi_dot);

// Single RK4 step for the 10x10 matrix ODE
void rk4_matrix_step(const double* Psi, double t, double h,
                     const InterpData& interp, double* Psi_next);

// Full exact discretization for one interval
// Given two reference nodes, compute A_k (6x6) and B_k (6x4)
// n_substeps: number of RK4 substeps (default 50)
void exact_discretize(const RefNode& ref_k, const RefNode& ref_k1,
                      const ModelParams& params,
                      double A_k[NX * NX], double B_k[NX * NU],
                      int n_substeps = 50);

// Compute affine offset: c_k = x_ref_{k+1} - A_k * x_ref_k - B_k * u_ref_k
void compute_affine_offset(const double A_k[NX * NX], const double B_k[NX * NU],
                           const RefNode& ref_k, const RefNode& ref_k1,
                           double c_k[NX]);
