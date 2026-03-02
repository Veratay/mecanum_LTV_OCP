#pragma once

#include <cstdint>
#include <cstring>

// Fixed problem dimensions
constexpr int NX = 6;      // state: [px, py, theta, vx, vy, omega]
constexpr int NU = 4;      // input: [V1, V2, V3, V4]
constexpr int N_MAX = 30;  // maximum horizon length
constexpr int N_AUG = NX + NU;  // augmented system dimension (10)

// Robot parameters
struct ModelParams {
    double mass;              // kg
    double inertia;           // kg*m^2
    double damping_linear;    // N*s/m
    double damping_angular;   // N*m*s/rad
    double wheel_radius;      // m
    double lx, ly;            // half-distances center to wheel (m)
    double motor_kv;          // V -> force constant (N/V through kinematic chain)
    double J_mec[3 * NU];    // 3x4 mecanum Jacobian, column-major
};

// MPC tuning
struct MPCConfig {
    int N;                    // horizon length (<= N_MAX)
    double Q[NX * NX];       // state cost, column-major
    double R[NU * NU];       // input cost, column-major
    double Qf[NX * NX];      // terminal cost, column-major
    double V_min;             // voltage lower bound (scalar, same for all wheels)
    double V_max;             // voltage upper bound
    double dt;                // control timestep (s)
};

// Reference trajectory node
struct RefNode {
    double x_ref[NX];        // reference state
    double u_ref[NU];        // reference input (feedforward)
    double theta;             // heading (explicit copy of x_ref[2])
    double omega;             // angular velocity (explicit copy of x_ref[5])
    double t;                 // timestamp
};

// Precomputed QP data for one MPC window (output of offline phase)
struct PrecomputedWindow {
    double H[N_MAX * NU * N_MAX * NU];     // condensed Hessian, column-major
    double L[N_MAX * NU * N_MAX * NU];     // lower Cholesky factor
    double F[N_MAX * NU * NX];             // gradient linear term: g = F*e0 + f_const
    double f_const[N_MAX * NU];            // gradient constant term
    double x_ref_0[NX];                    // reference state at window start
    int N;                                  // horizon length
    int n_vars;                             // N * NU
};

// Online QP solution
struct QPSolution {
    double U[N_MAX * NU];       // full optimized input sequence
    double u0[NU];              // first control: [V1, V2, V3, V4]
    int n_active;               // number of active constraints at solution
    int n_iterations;           // active-set iterations taken
    double solve_time_ns;       // timing
};

// Preallocated workspace for box QP solver (no heap allocation online)
struct BoxQPWorkspace {
    double grad[N_MAX * NU];
    double U[N_MAX * NU];
    double rhs[N_MAX * NU];
    double H_ff[N_MAX * NU * N_MAX * NU];
    double L_ff[N_MAX * NU * N_MAX * NU];
    double temp[N_MAX * NU];
    int free_idx[N_MAX * NU];
    int clamped_idx[N_MAX * NU];
};

// Serialization header
struct MPCFileHeader {
    uint32_t magic;       // 0x4D504351 ("MPCQ")
    uint32_t version;     // 1
    uint32_t n_windows;
    uint32_t N;           // horizon
    uint32_t nx;
    uint32_t nu;
    double V_min;
    double V_max;
};

constexpr uint32_t MPC_FILE_MAGIC = 0x4D504351;
constexpr uint32_t MPC_FILE_VERSION = 1;
