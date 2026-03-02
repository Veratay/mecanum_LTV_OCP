// test_discretizer.cpp -- standalone tests for exact discretizer
//
// Tests: cubic Hermite heading interpolation, exact discretization accuracy
// at omega=0 and omega=3 rad/s, and affine offset consistency.

#include "discretizer.h"
#include "mecanum_model.h"
#include "blas_dispatch.h"

#include <cstdio>
#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static ModelParams make_params()
{
    ModelParams p{};
    p.mass            = 10.0;
    p.inertia         = 0.5;
    p.damping_linear  = 2.0;
    p.damping_angular = 0.3;
    p.wheel_radius    = 0.05;
    p.lx              = 0.15;
    p.ly              = 0.15;
    p.motor_kv        = 0.5;
    compute_mecanum_jacobian(p);
    return p;
}

static double vec_max_abs(const double* v, int n)
{
    double mx = 0.0;
    for (int i = 0; i < n; ++i) {
        double a = std::fabs(v[i]);
        if (a > mx) mx = a;
    }
    return mx;
}

// ---------------------------------------------------------------------------
// Test 1: Heading interpolant boundaries and mid-point
// ---------------------------------------------------------------------------
static bool test_heading_interpolant()
{
    std::printf("Test 1: Heading interpolant ... ");

    const double t_k     = 0.0;
    const double t_k1    = 0.02;
    const double theta_k  = 0.5;
    const double omega_k  = 1.0;
    const double theta_k1 = 0.52;
    const double omega_k1 = 1.1;

    bool ok = true;

    // theta_interp at boundaries
    {
        double val = theta_interp(t_k, t_k, t_k1, theta_k, omega_k, theta_k1, omega_k1);
        double err = std::fabs(val - theta_k);
        if (err > 1e-14) {
            std::printf("\n  theta_interp(t_k) error = %.3e ", err);
            ok = false;
        }
    }
    {
        double val = theta_interp(t_k1, t_k, t_k1, theta_k, omega_k, theta_k1, omega_k1);
        double err = std::fabs(val - theta_k1);
        if (err > 1e-14) {
            std::printf("\n  theta_interp(t_k1) error = %.3e ", err);
            ok = false;
        }
    }

    // omega_interp at boundaries
    {
        double val = omega_interp(t_k, t_k, t_k1, theta_k, omega_k, theta_k1, omega_k1);
        double err = std::fabs(val - omega_k);
        if (err > 1e-14) {
            std::printf("\n  omega_interp(t_k) error = %.3e ", err);
            ok = false;
        }
    }
    {
        double val = omega_interp(t_k1, t_k, t_k1, theta_k, omega_k, theta_k1, omega_k1);
        double err = std::fabs(val - omega_k1);
        if (err > 1e-14) {
            std::printf("\n  omega_interp(t_k1) error = %.3e ", err);
            ok = false;
        }
    }

    // Mid-point: theta should be between theta_k and theta_k1 (for monotone case)
    {
        double t_mid = 0.5 * (t_k + t_k1);
        double val = theta_interp(t_mid, t_k, t_k1, theta_k, omega_k, theta_k1, omega_k1);
        if (val < std::fmin(theta_k, theta_k1) - 1e-10 ||
            val > std::fmax(theta_k, theta_k1) + 1e-10) {
            std::printf("\n  theta_interp(mid) = %.6f not between [%.6f, %.6f] ",
                        val, theta_k, theta_k1);
            ok = false;
        }
    }

    // Verify omega_interp is the numerical derivative of theta_interp
    {
        double t_test = t_k + 0.3 * (t_k1 - t_k);
        double eps = 1e-8;
        double th_plus  = theta_interp(t_test + eps, t_k, t_k1, theta_k, omega_k, theta_k1, omega_k1);
        double th_minus = theta_interp(t_test - eps, t_k, t_k1, theta_k, omega_k, theta_k1, omega_k1);
        double num_deriv = (th_plus - th_minus) / (2.0 * eps);
        double analytic  = omega_interp(t_test, t_k, t_k1, theta_k, omega_k, theta_k1, omega_k1);
        double err = std::fabs(num_deriv - analytic);
        if (err > 1e-5) {
            std::printf("\n  omega_interp vs numerical derivative error = %.3e ", err);
            ok = false;
        }
    }

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Test 2: Discretization accuracy at omega=0 (straight line)
// ---------------------------------------------------------------------------
static bool test_discretize_omega_zero()
{
    std::printf("Test 2: Discretization at omega=0 ... ");

    ModelParams params = make_params();
    const double dt = 0.02;

    // Reference nodes: origin, heading=0, zero angular velocity
    RefNode ref0{};
    std::memset(&ref0, 0, sizeof(ref0));
    ref0.theta = 0.0;
    ref0.omega = 0.0;
    ref0.t     = 0.0;

    RefNode ref1{};
    std::memset(&ref1, 0, sizeof(ref1));
    ref1.theta = 0.0;
    ref1.omega = 0.0;
    ref1.t     = dt;

    // Exact discretize
    double A_k[NX * NX], B_k[NX * NU];
    exact_discretize(ref0, ref1, params, A_k, B_k, 50);

    // Constant input
    double u[NU] = {1.0, 1.0, 1.0, 1.0};

    // x_exact = A_k * x0 + B_k * u,  where x0 = 0
    double x_exact[NX];
    mpc_linalg::gemv(NX, NU, B_k, u, x_exact);
    // (A_k * 0 = 0, so x_exact = B_k * u)

    // Fine RK4 simulation: x_dot = Ac*x + Bc*u at constant theta=0
    double Ac[NX * NX], Bc[NX * NU];
    continuous_dynamics(0.0, params, Ac, Bc);

    // Compute Bc*u once (constant)
    double Bcu[NX];
    mpc_linalg::gemv(NX, NU, Bc, u, Bcu);

    // Forward Euler with 10000 substeps
    const int N_fine = 10000;
    const double dt_fine = dt / N_fine;
    double x_fine[NX];
    std::memset(x_fine, 0, NX * sizeof(double));

    for (int i = 0; i < N_fine; ++i) {
        // k1 = Ac*x + Bc*u
        double k1[NX], k2[NX], k3[NX], k4[NX], tmp[NX];

        mpc_linalg::gemv(NX, NX, Ac, x_fine, k1);
        for (int j = 0; j < NX; ++j) k1[j] += Bcu[j];

        // k2
        for (int j = 0; j < NX; ++j) tmp[j] = x_fine[j] + 0.5 * dt_fine * k1[j];
        mpc_linalg::gemv(NX, NX, Ac, tmp, k2);
        for (int j = 0; j < NX; ++j) k2[j] += Bcu[j];

        // k3
        for (int j = 0; j < NX; ++j) tmp[j] = x_fine[j] + 0.5 * dt_fine * k2[j];
        mpc_linalg::gemv(NX, NX, Ac, tmp, k3);
        for (int j = 0; j < NX; ++j) k3[j] += Bcu[j];

        // k4
        for (int j = 0; j < NX; ++j) tmp[j] = x_fine[j] + dt_fine * k3[j];
        mpc_linalg::gemv(NX, NX, Ac, tmp, k4);
        for (int j = 0; j < NX; ++j) k4[j] += Bcu[j];

        for (int j = 0; j < NX; ++j)
            x_fine[j] += (dt_fine / 6.0) * (k1[j] + 2.0*k2[j] + 2.0*k3[j] + k4[j]);
    }

    // Compare
    double err[NX];
    for (int i = 0; i < NX; ++i) err[i] = x_exact[i] - x_fine[i];
    double max_err = vec_max_abs(err, NX);

    bool ok = max_err < 1e-9;
    if (!ok) {
        std::printf("\n  max error = %.3e (expected < 1e-9)", max_err);
        for (int i = 0; i < NX; ++i)
            std::printf("\n    x_exact[%d]=%.12e  x_fine[%d]=%.12e  err=%.3e",
                        i, x_exact[i], i, x_fine[i], err[i]);
    }
    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Test 3: Discretization accuracy at omega=3 rad/s (high rotation)
// ---------------------------------------------------------------------------
static bool test_discretize_omega_high()
{
    std::printf("Test 3: Discretization at omega=3 ... ");

    ModelParams params = make_params();
    const double dt = 0.02;

    // Start heading=0, omega=3, end heading=0.06, omega=3
    RefNode ref0{};
    std::memset(&ref0, 0, sizeof(ref0));
    ref0.x_ref[2] = 0.0;   // theta
    ref0.x_ref[5] = 3.0;   // omega
    ref0.theta = 0.0;
    ref0.omega = 3.0;
    ref0.t     = 0.0;

    RefNode ref1{};
    std::memset(&ref1, 0, sizeof(ref1));
    ref1.x_ref[2] = 0.06;
    ref1.x_ref[5] = 3.0;
    ref1.theta = 0.06;
    ref1.omega = 3.0;
    ref1.t     = dt;

    // Exact discretize with default 50 substeps
    double A_k[NX * NX], B_k[NX * NU];
    exact_discretize(ref0, ref1, params, A_k, B_k, 50);

    // Constant input
    double u[NU] = {1.0, 1.0, 1.0, 1.0};

    // x_exact = A_k * x0 + B_k * u, where x0 = ref0.x_ref
    double Ax0[NX], Bu[NX], x_exact[NX];
    mpc_linalg::gemv(NX, NX, A_k, ref0.x_ref, Ax0);
    mpc_linalg::gemv(NX, NU, B_k, u, Bu);
    for (int i = 0; i < NX; ++i)
        x_exact[i] = Ax0[i] + Bu[i];

    // Fine simulation with 10000 substeps, interpolating theta at each step
    const int N_fine = 10000;
    const double dt_fine = dt / N_fine;
    double x_fine[NX];
    std::memcpy(x_fine, ref0.x_ref, NX * sizeof(double));

    for (int i = 0; i < N_fine; ++i) {
        double t_i = i * dt_fine;

        // Interpolate theta at sub-step points
        auto get_Ac_Bc = [&](double t_sub, double* Ac_sub, double* Bc_sub) {
            double th = theta_interp(t_sub, ref0.t, ref1.t,
                                     ref0.theta, ref0.omega,
                                     ref1.theta, ref1.omega);
            continuous_dynamics(th, params, Ac_sub, Bc_sub);
        };

        // RK4 with varying theta
        double Ac_t[NX * NX], Bc_t[NX * NU];
        double k1[NX], k2[NX], k3[NX], k4[NX], tmp[NX], Bcu[NX];

        // k1 at t_i
        get_Ac_Bc(t_i, Ac_t, Bc_t);
        mpc_linalg::gemv(NX, NX, Ac_t, x_fine, k1);
        mpc_linalg::gemv(NX, NU, Bc_t, u, Bcu);
        for (int j = 0; j < NX; ++j) k1[j] += Bcu[j];

        // k2 at t_i + dt_fine/2
        for (int j = 0; j < NX; ++j) tmp[j] = x_fine[j] + 0.5 * dt_fine * k1[j];
        get_Ac_Bc(t_i + 0.5 * dt_fine, Ac_t, Bc_t);
        mpc_linalg::gemv(NX, NX, Ac_t, tmp, k2);
        mpc_linalg::gemv(NX, NU, Bc_t, u, Bcu);
        for (int j = 0; j < NX; ++j) k2[j] += Bcu[j];

        // k3 at t_i + dt_fine/2
        for (int j = 0; j < NX; ++j) tmp[j] = x_fine[j] + 0.5 * dt_fine * k2[j];
        mpc_linalg::gemv(NX, NX, Ac_t, tmp, k3);
        mpc_linalg::gemv(NX, NU, Bc_t, u, Bcu);
        for (int j = 0; j < NX; ++j) k3[j] += Bcu[j];

        // k4 at t_i + dt_fine
        for (int j = 0; j < NX; ++j) tmp[j] = x_fine[j] + dt_fine * k3[j];
        get_Ac_Bc(t_i + dt_fine, Ac_t, Bc_t);
        mpc_linalg::gemv(NX, NX, Ac_t, tmp, k4);
        mpc_linalg::gemv(NX, NU, Bc_t, u, Bcu);
        for (int j = 0; j < NX; ++j) k4[j] += Bcu[j];

        for (int j = 0; j < NX; ++j)
            x_fine[j] += (dt_fine / 6.0) * (k1[j] + 2.0*k2[j] + 2.0*k3[j] + k4[j]);
    }

    // Compare
    double err[NX];
    for (int i = 0; i < NX; ++i) err[i] = x_exact[i] - x_fine[i];
    double max_err = vec_max_abs(err, NX);

    bool ok = max_err < 1e-6;
    if (!ok) {
        std::printf("\n  max error = %.3e (expected < 1e-6)", max_err);
        for (int i = 0; i < NX; ++i)
            std::printf("\n    x_exact[%d]=%.12e  x_fine[%d]=%.12e  err=%.3e",
                        i, x_exact[i], i, x_fine[i], err[i]);
    }
    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Test 4: Affine offset is small for dynamically consistent reference
// ---------------------------------------------------------------------------
static bool test_affine_offset()
{
    std::printf("Test 4: Affine offset consistency ... ");

    ModelParams params = make_params();
    const double dt = 0.02;

    // Build ref0 with some nonzero state and input
    RefNode ref0{};
    std::memset(&ref0, 0, sizeof(ref0));
    ref0.x_ref[0] = 0.1;   // px
    ref0.x_ref[1] = 0.2;   // py
    ref0.x_ref[2] = 0.3;   // theta
    ref0.x_ref[3] = 0.5;   // vx
    ref0.x_ref[4] = -0.1;  // vy
    ref0.x_ref[5] = 1.0;   // omega
    ref0.u_ref[0] = 1.0;
    ref0.u_ref[1] = 0.5;
    ref0.u_ref[2] = 0.8;
    ref0.u_ref[3] = 0.3;
    ref0.theta = ref0.x_ref[2];
    ref0.omega = ref0.x_ref[5];
    ref0.t     = 0.0;

    // Simulate ref1 = A_k * ref0 + B_k * u_ref0  (dynamically consistent)
    // First, pick ref1's heading/omega for the interpolant.
    // For a truly consistent reference, we need to integrate the actual
    // nonlinear dynamics.  We'll use a fine simulation to generate ref1.

    // Fine simulation with 10000 substeps to get the "true" next state
    const int N_fine = 10000;
    const double dt_fine = dt / N_fine;
    double x_sim[NX];
    std::memcpy(x_sim, ref0.x_ref, NX * sizeof(double));

    for (int i = 0; i < N_fine; ++i) {
        double theta_now = x_sim[2];
        double Ac[NX * NX], Bc[NX * NU];
        continuous_dynamics(theta_now, params, Ac, Bc);

        double k1[NX], Bcu[NX];
        mpc_linalg::gemv(NX, NX, Ac, x_sim, k1);
        mpc_linalg::gemv(NX, NU, Bc, ref0.u_ref, Bcu);
        for (int j = 0; j < NX; ++j) k1[j] += Bcu[j];

        for (int j = 0; j < NX; ++j)
            x_sim[j] += dt_fine * k1[j];
    }

    // Set ref1 from the simulated state
    RefNode ref1{};
    std::memcpy(ref1.x_ref, x_sim, NX * sizeof(double));
    std::memcpy(ref1.u_ref, ref0.u_ref, NU * sizeof(double));
    ref1.theta = ref1.x_ref[2];
    ref1.omega = ref1.x_ref[5];
    ref1.t     = dt;

    // Now discretize using the exact discretizer
    double A_k[NX * NX], B_k[NX * NU];
    exact_discretize(ref0, ref1, params, A_k, B_k, 50);

    // Compute affine offset -- should be small since ref1 was generated
    // by integrating the same dynamics (up to linearization error)
    double c_k[NX];
    compute_affine_offset(A_k, B_k, ref0, ref1, c_k);

    double max_c = vec_max_abs(c_k, NX);

    // The offset won't be exactly zero because:
    //   - The exact discretizer linearizes around the heading interpolant
    //   - The fine sim integrates the full nonlinear dynamics
    // But it should be small for small dt.
    bool ok = max_c < 1e-4;
    if (!ok) {
        std::printf("\n  max |c_k| = %.3e (expected < 1e-4)", max_c);
        for (int i = 0; i < NX; ++i)
            std::printf("\n    c_k[%d] = %.6e", i, c_k[i]);
    }
    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    std::printf("=== test_discretizer ===\n");

    bool all_pass = true;
    all_pass &= test_heading_interpolant();
    all_pass &= test_discretize_omega_zero();
    all_pass &= test_discretize_omega_high();
    all_pass &= test_affine_offset();

    std::printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
