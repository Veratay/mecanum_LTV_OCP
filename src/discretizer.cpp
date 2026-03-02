#include "discretizer.h"
#include "mecanum_model.h"
#include "blas_dispatch.h"

#include <cblas.h>
#include <cstring>
#include <cmath>

// ---------------------------------------------------------------------------
// Cubic Hermite heading interpolation
// ---------------------------------------------------------------------------
double theta_interp(double t, double t_k, double t_k1,
                    double theta_k, double omega_k,
                    double theta_k1, double omega_k1)
{
    const double h = t_k1 - t_k;
    const double s = (t - t_k) / h;
    const double s2 = s * s;
    const double s3 = s2 * s;

    return (2.0*s3 - 3.0*s2 + 1.0) * theta_k
         + (s3 - 2.0*s2 + s) * h * omega_k
         + (-2.0*s3 + 3.0*s2) * theta_k1
         + (s3 - s2) * h * omega_k1;
}

// ---------------------------------------------------------------------------
// Derivative of the cubic Hermite interpolant
// ---------------------------------------------------------------------------
double omega_interp(double t, double t_k, double t_k1,
                    double theta_k, double omega_k,
                    double theta_k1, double omega_k1)
{
    const double h = t_k1 - t_k;
    const double s = (t - t_k) / h;
    const double s2 = s * s;
    // ds/dt = 1/h
    const double ds = 1.0 / h;

    // d/dt theta = d/ds theta * ds/dt
    const double dtheta_ds =
          (6.0*s2 - 6.0*s) * theta_k
        + (3.0*s2 - 4.0*s + 1.0) * h * omega_k
        + (-6.0*s2 + 6.0*s) * theta_k1
        + (3.0*s2 - 2.0*s) * h * omega_k1;

    return dtheta_ds * ds;
}

// ---------------------------------------------------------------------------
// Augmented ODE RHS: Psi_dot = M(t) * Psi
// ---------------------------------------------------------------------------
void augmented_rhs(double t, const double* Psi, const InterpData& interp,
                   double* Psi_dot)
{
    // Interpolate heading at current time
    const double theta = theta_interp(t, interp.t_k, interp.t_k1,
                                      interp.theta_k, interp.omega_k,
                                      interp.theta_k1, interp.omega_k1);

    // Get continuous-time matrices
    double Ac[NX * NX];
    double Bc[NX * NU];
    continuous_dynamics(theta, *interp.params, Ac, Bc);

    // Zero entire output (10x10 = 100 doubles)
    std::memset(Psi_dot, 0, N_AUG * N_AUG * sizeof(double));

    // Psi_dot[0:6, :] = Ac(6x6) * Psi[0:6, :] + Bc(6x4) * Psi[6:10, :]
    // Bottom 4 rows of Psi_dot remain zero.

    // Ac(6x6) * Psi_top(6x10) -> Psi_dot_top(6x10)
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                6, N_AUG, 6,
                1.0, Ac, NX,
                Psi, N_AUG,
                0.0, Psi_dot, N_AUG);

    // += Bc(6x4) * Psi_bottom(4x10) -> accumulate into Psi_dot_top(6x10)
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                6, N_AUG, 4,
                1.0, Bc, NX,
                Psi + 6, N_AUG,
                1.0, Psi_dot, N_AUG);
}

// ---------------------------------------------------------------------------
// Single RK4 step for the 10x10 matrix ODE
// ---------------------------------------------------------------------------
void rk4_matrix_step(const double* Psi, double t, double h,
                     const InterpData& interp, double* Psi_next)
{
    constexpr int SZ = N_AUG * N_AUG;  // 100

    double k1[SZ], k2[SZ], k3[SZ], k4[SZ], temp[SZ];

    // k1 = rhs(t, Psi)
    augmented_rhs(t, Psi, interp, k1);

    // temp = Psi + (h/2)*k1
    const double h2 = h * 0.5;
    for (int i = 0; i < SZ; ++i)
        temp[i] = Psi[i] + h2 * k1[i];

    // k2 = rhs(t + h/2, temp)
    augmented_rhs(t + h2, temp, interp, k2);

    // temp = Psi + (h/2)*k2
    for (int i = 0; i < SZ; ++i)
        temp[i] = Psi[i] + h2 * k2[i];

    // k3 = rhs(t + h/2, temp)
    augmented_rhs(t + h2, temp, interp, k3);

    // temp = Psi + h*k3
    for (int i = 0; i < SZ; ++i)
        temp[i] = Psi[i] + h * k3[i];

    // k4 = rhs(t + h, temp)
    augmented_rhs(t + h, temp, interp, k4);

    // Psi_next = Psi + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    const double h6 = h / 6.0;
    for (int i = 0; i < SZ; ++i)
        Psi_next[i] = Psi[i] + h6 * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
}

// ---------------------------------------------------------------------------
// Full exact discretization for one interval
// ---------------------------------------------------------------------------
void exact_discretize(const RefNode& ref_k, const RefNode& ref_k1,
                      const ModelParams& params,
                      double A_k[NX * NX], double B_k[NX * NU],
                      int n_substeps)
{
    constexpr int SZ = N_AUG * N_AUG;

    // Initialize Psi = I_10
    double Psi[SZ];
    std::memset(Psi, 0, SZ * sizeof(double));
    for (int i = 0; i < N_AUG; ++i)
        Psi[i + N_AUG * i] = 1.0;

    // Set up interpolation data
    InterpData interp;
    interp.t_k     = ref_k.t;
    interp.t_k1    = ref_k1.t;
    interp.theta_k  = ref_k.theta;
    interp.omega_k  = ref_k.omega;
    interp.theta_k1 = ref_k1.theta;
    interp.omega_k1 = ref_k1.omega;
    interp.params   = &params;

    // Integrate with n_substeps RK4 steps
    const double dt = (ref_k1.t - ref_k.t) / n_substeps;
    double Psi_next[SZ];

    for (int step = 0; step < n_substeps; ++step) {
        const double t = ref_k.t + step * dt;
        rk4_matrix_step(Psi, t, dt, interp, Psi_next);
        std::memcpy(Psi, Psi_next, SZ * sizeof(double));
    }

    // Extract A_k = Psi[0:6, 0:6]
    for (int j = 0; j < NX; ++j)
        for (int i = 0; i < NX; ++i)
            A_k[i + NX * j] = Psi[i + N_AUG * j];

    // Extract B_k = Psi[0:6, 6:10]
    for (int j = 0; j < NU; ++j)
        for (int i = 0; i < NX; ++i)
            B_k[i + NX * j] = Psi[i + N_AUG * (j + NX)];
}

// ---------------------------------------------------------------------------
// Compute affine offset: c_k = x_ref_{k+1} - A_k * x_ref_k - B_k * u_ref_k
// ---------------------------------------------------------------------------
void compute_affine_offset(const double A_k[NX * NX], const double B_k[NX * NU],
                           const RefNode& ref_k, const RefNode& ref_k1,
                           double c_k[NX])
{
    // c_k = -A_k * x_ref_k
    double Ax[NX];
    mpc_linalg::gemv(NX, NX, A_k, ref_k.x_ref, Ax);

    // c_k = x_ref_{k+1} - Ax
    for (int i = 0; i < NX; ++i)
        c_k[i] = ref_k1.x_ref[i] - Ax[i];

    // c_k -= B_k * u_ref_k
    double Bu[NX];
    mpc_linalg::gemv(NX, NU, B_k, ref_k.u_ref, Bu);
    for (int i = 0; i < NX; ++i)
        c_k[i] -= Bu[i];
}
