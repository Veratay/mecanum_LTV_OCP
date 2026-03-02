#pragma once

#include "mpc_types.h"

// Main online MPC solve function
// Given a precomputed window and current state x0, solve for optimal control
QPSolution mpc_solve_online(const PrecomputedWindow& window, const double x0[NX],
                            const MPCConfig& config, BoxQPWorkspace& workspace);
