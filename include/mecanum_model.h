#pragma once

#include "mpc_types.h"

// Compute the 3x4 mecanum Jacobian J_mec and store in params.J_mec
// Maps wheel duty cycles to body-frame [Fx, Fy, tau] at stall (zero speed)
void compute_mecanum_jacobian(ModelParams& params);

// Compute continuous-time LTV matrices Ac(theta), Bc(theta)
// Ac is 6x6, Bc is 6x4, both column-major
void continuous_dynamics(double theta, const ModelParams& params,
                         double Ac[NX * NX], double Bc[NX * NU]);
