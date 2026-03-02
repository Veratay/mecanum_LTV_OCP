#include <cstdio>
#include <cmath>
#include <cstring>

#include "mpc_types.h"
#include "condensing.h"
#include "discretizer.h"
#include "mecanum_model.h"
#include "blas_dispatch.h"
#include "cholesky.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static int g_tests_run    = 0;
static int g_tests_passed = 0;

static void check(bool cond, const char* name) {
    g_tests_run++;
    if (cond) {
        g_tests_passed++;
        std::printf("  [PASS] %s\n", name);
    } else {
        std::printf("  [FAIL] %s\n", name);
    }
}

// Max absolute element of a vector
static double max_abs(int len, const double* v) {
    double m = 0.0;
    for (int i = 0; i < len; i++) {
        double a = std::fabs(v[i]);
        if (a > m) m = a;
    }
    return m;
}

// ---------------------------------------------------------------------------
// Setup: model, config, reference trajectory, discretization
// ---------------------------------------------------------------------------
static constexpr int N_HORIZON = 5;
static constexpr int NVARS = N_HORIZON * NU;  // 20

struct TestSetup {
    ModelParams params;
    MPCConfig config;
    RefNode refs[N_HORIZON + 1];

    double A_list[N_HORIZON * NX * NX];
    double B_list[N_HORIZON * NX * NU];

    // Consistent reference (forward-propagated from x_ref[0])
    double x_ref_consistent[(N_HORIZON + 1) * NX];
    double u_ref[N_HORIZON * NU];

    PrecomputedWindow window;
};

static void setup(TestSetup& ts) {
    // -- Model parameters --
    ModelParams& p = ts.params;
    p.mass             = 10.0;
    p.inertia          = 0.5;
    p.damping_linear   = 2.0;
    p.damping_angular  = 0.3;
    p.wheel_radius     = 0.05;
    p.lx               = 0.15;
    p.ly               = 0.15;
    p.motor_kv         = 0.5;
    compute_mecanum_jacobian(p);

    // -- MPC config --
    MPCConfig& c = ts.config;
    c.N     = N_HORIZON;
    c.dt    = 0.02;
    c.V_min = -12.0;
    c.V_max = 12.0;

    // Q = diag(10, 10, 5, 1, 1, 0.5)
    std::memset(c.Q, 0, sizeof(c.Q));
    double q_diag[NX] = {10.0, 10.0, 5.0, 1.0, 1.0, 0.5};
    for (int i = 0; i < NX; i++) c.Q[i + NX * i] = q_diag[i];

    // R = diag(0.1, 0.1, 0.1, 0.1)
    std::memset(c.R, 0, sizeof(c.R));
    for (int i = 0; i < NU; i++) c.R[i + NU * i] = 0.1;

    // Qf = 2*Q
    for (int i = 0; i < NX * NX; i++) c.Qf[i] = 2.0 * c.Q[i];

    // -- Reference trajectory: straight line along x-axis --
    // x_ref[k] = [k*0.01, 0, 0, 0.5, 0, 0]
    // u_ref = 0
    std::memset(ts.u_ref, 0, sizeof(ts.u_ref));

    for (int k = 0; k <= N_HORIZON; k++) {
        RefNode& r = ts.refs[k];
        r.x_ref[0] = k * 0.01;  // px
        r.x_ref[1] = 0.0;       // py
        r.x_ref[2] = 0.0;       // theta
        r.x_ref[3] = 0.5;       // vx
        r.x_ref[4] = 0.0;       // vy
        r.x_ref[5] = 0.0;       // omega
        r.theta = 0.0;
        r.omega = 0.0;
        r.t     = k * c.dt;
    }

    // -- Discretize --
    for (int k = 0; k < N_HORIZON; k++) {
        exact_discretize(ts.refs[k], ts.refs[k + 1], p,
                         ts.A_list + k * NX * NX,
                         ts.B_list + k * NX * NU,
                         50);
    }

    // -- Build consistent reference by forward propagation --
    // x_ref_consistent[0] = refs[0].x_ref
    std::memcpy(ts.x_ref_consistent, ts.refs[0].x_ref, NX * sizeof(double));

    for (int k = 0; k < N_HORIZON; k++) {
        const double* Ak = ts.A_list + k * NX * NX;
        const double* Bk = ts.B_list + k * NX * NU;
        double* xk   = ts.x_ref_consistent + k * NX;
        double* xk1  = ts.x_ref_consistent + (k + 1) * NX;

        // x_{k+1} = A_k * x_k + B_k * u_ref_k
        mpc_linalg::gemv(NX, NX, Ak, xk, xk1);
        // Add B_k * u_ref_k (u_ref is zero, but do it for correctness)
        double Bu[NX];
        mpc_linalg::gemv(NX, NU, Bk, ts.u_ref + k * NU, Bu);
        mpc_linalg::axpy(NX, 1.0, Bu, xk1);
    }

    // -- Condense --
    condense_window(ts.A_list, ts.B_list,
                    ts.x_ref_consistent, ts.u_ref,
                    ts.config, ts.window);
}

// ---------------------------------------------------------------------------
// Forward simulate: given x0, U (length NVARS), compute state trajectory
// x[0..N] each of length NX using A_list, B_list from TestSetup
// ---------------------------------------------------------------------------
static void forward_simulate(const TestSetup& ts, const double* x0,
                              const double* U, double* x_traj)
{
    std::memcpy(x_traj, x0, NX * sizeof(double));
    for (int k = 0; k < N_HORIZON; k++) {
        const double* Ak = ts.A_list + k * NX * NX;
        const double* Bk = ts.B_list + k * NX * NU;
        const double* uk = U + k * NU;
        double* xk  = x_traj + k * NX;
        double* xk1 = x_traj + (k + 1) * NX;

        mpc_linalg::gemv(NX, NX, Ak, xk, xk1);
        double Bu[NX];
        mpc_linalg::gemv(NX, NU, Bk, uk, Bu);
        mpc_linalg::axpy(NX, 1.0, Bu, xk1);
    }
}

// ---------------------------------------------------------------------------
// Compute the full stage cost given a state trajectory and input sequence
// cost = sum_{k=0}^{N-1} (x_k - x_ref_k)^T Q (x_k - x_ref_k)
//      + sum_{k=0}^{N-1} (u_k - u_ref_k)^T R (u_k - u_ref_k)
//      + (x_N - x_ref_N)^T Qf (x_N - x_ref_N)
// ---------------------------------------------------------------------------
static double compute_cost(const TestSetup& ts, const double* x_traj,
                           const double* U)
{
    double cost = 0.0;
    double e[NX], Qe[NX];

    // Stage costs (k = 0..N for state)
    // Note: k=0 term is constant w.r.t. U so doesn't affect gradients
    for (int k = 0; k <= N_HORIZON; k++) {
        for (int i = 0; i < NX; i++)
            e[i] = x_traj[k * NX + i] - ts.x_ref_consistent[k * NX + i];

        const double* Qk = (k < N_HORIZON) ? ts.config.Q : ts.config.Qf;
        mpc_linalg::gemv(NX, NX, Qk, e, Qe);
        cost += mpc_linalg::dot(NX, e, Qe);
    }

    // Input costs
    double eu[NU], Reu[NU];
    for (int k = 0; k < N_HORIZON; k++) {
        for (int i = 0; i < NU; i++)
            eu[i] = U[k * NU + i] - ts.u_ref[k * NU + i];
        mpc_linalg::gemv(NU, NU, ts.config.R, eu, Reu);
        cost += mpc_linalg::dot(NU, eu, Reu);
    }

    // Return 0.5 * cost to match the standard QP form: 0.5 * U^T H U + g^T U
    return 0.5 * cost;
}

// ---------------------------------------------------------------------------
// Compute the condensed cost: 0.5 * U^T * H * U + g^T * U + const_term
// where g = F * e0 + f_const
// ---------------------------------------------------------------------------
static double compute_condensed_cost(const TestSetup& ts, const double* x0,
                                     const double* U, double* grad_out)
{
    const PrecomputedWindow& w = ts.window;
    int nv = w.n_vars;

    // e0 = x0 - x_ref_0
    double e0[NX];
    for (int i = 0; i < NX; i++)
        e0[i] = x0[i] - w.x_ref_0[i];

    // g = F * e0 + f_const
    double g[N_MAX * NU];
    mpc_linalg::gemv(nv, NX, w.F, e0, g);
    mpc_linalg::axpy(nv, 1.0, w.f_const, g);

    // HU = H * U
    double HU[N_MAX * NU];
    mpc_linalg::gemv(nv, nv, w.H, U, HU);

    // cost = 0.5 * U^T * HU + g^T * U
    double cost = 0.5 * mpc_linalg::dot(nv, U, HU) + mpc_linalg::dot(nv, g, U);

    // gradient = H * U + g
    if (grad_out) {
        for (int i = 0; i < nv; i++)
            grad_out[i] = HU[i] + g[i];
    }

    return cost;
}

// ---------------------------------------------------------------------------
// Test 1: Cost gradient equivalence
// Verify that the gradient of the stagewise cost w.r.t. U matches the
// condensed gradient H*U + g via central finite differences.
// ---------------------------------------------------------------------------
static void test_cost_gradient_equivalence(const TestSetup& ts) {
    std::printf("--- Test 1: Cost gradient equivalence ---\n");

    // x0 = x_ref_0 + small perturbation
    double x0[NX];
    double x0_offsets[NX] = {0.01, -0.005, 0.003, 0.02, -0.01, 0.004};
    for (int i = 0; i < NX; i++)
        x0[i] = ts.x_ref_consistent[i] + x0_offsets[i];

    // U = small deterministic values near zero
    double U[NVARS];
    for (int i = 0; i < NVARS; i++)
        U[i] = 0.05 * std::sin(0.7 * i + 0.3);

    // Condensed gradient
    double grad_condensed[NVARS];
    compute_condensed_cost(ts, x0, U, grad_condensed);

    // Finite-difference gradient of the stagewise cost
    double grad_fd[NVARS];
    const double eps = 1e-7;

    for (int i = 0; i < NVARS; i++) {
        double U_plus[NVARS], U_minus[NVARS];
        std::memcpy(U_plus,  U, sizeof(U));
        std::memcpy(U_minus, U, sizeof(U));
        U_plus[i]  += eps;
        U_minus[i] -= eps;

        double x_traj_p[(N_HORIZON + 1) * NX];
        double x_traj_m[(N_HORIZON + 1) * NX];
        forward_simulate(ts, x0, U_plus,  x_traj_p);
        forward_simulate(ts, x0, U_minus, x_traj_m);

        double cost_p = compute_cost(ts, x_traj_p, U_plus);
        double cost_m = compute_cost(ts, x_traj_m, U_minus);

        grad_fd[i] = (cost_p - cost_m) / (2.0 * eps);
    }

    // Compare
    double max_err = 0.0;
    for (int i = 0; i < NVARS; i++) {
        double err = std::fabs(grad_condensed[i] - grad_fd[i]);
        if (err > max_err) max_err = err;
    }

    char label[256];
    std::snprintf(label, sizeof(label),
                  "gradient agreement: max|grad_condensed - grad_fd| = %.3e (tol 1e-5)",
                  max_err);
    check(max_err < 1e-5, label);
}

// ---------------------------------------------------------------------------
// Test 2: Hessian symmetry
// ---------------------------------------------------------------------------
static void test_hessian_symmetry(const TestSetup& ts) {
    std::printf("\n--- Test 2: Hessian symmetry ---\n");

    const PrecomputedWindow& w = ts.window;
    int nv = w.n_vars;

    double max_asym = 0.0;
    for (int c = 0; c < nv; c++) {
        for (int r = c + 1; r < nv; r++) {
            double diff = std::fabs(w.H[r + nv * c] - w.H[c + nv * r]);
            if (diff > max_asym) max_asym = diff;
        }
    }

    char label[256];
    std::snprintf(label, sizeof(label),
                  "max|H - H^T| = %.3e (tol 1e-14)", max_asym);
    check(max_asym < 1e-14, label);
}

// ---------------------------------------------------------------------------
// Test 3: Cholesky correctness: max|L*L^T - H| < 1e-12
// ---------------------------------------------------------------------------
static void test_cholesky_correctness(const TestSetup& ts) {
    std::printf("\n--- Test 3: Cholesky correctness ---\n");

    const PrecomputedWindow& w = ts.window;
    int nv = w.n_vars;

    // Compute L * L^T
    // Build L^T from lower-triangular L
    double Lt[NVARS * NVARS];
    double LLt[NVARS * NVARS];

    std::memset(Lt, 0, sizeof(Lt));
    for (int j = 0; j < nv; j++)
        for (int i = 0; i <= j; i++)
            Lt[i + nv * j] = w.L[j + nv * i];  // Lt(i,j) = L(j,i)

    mpc_linalg::gemm(nv, nv, nv, w.L, Lt, LLt);

    double max_err = 0.0;
    for (int i = 0; i < nv * nv; i++) {
        double err = std::fabs(LLt[i] - w.H[i]);
        if (err > max_err) max_err = err;
    }

    char label[256];
    std::snprintf(label, sizeof(label),
                  "max|L*L^T - H| = %.3e (tol 1e-12)", max_err);
    check(max_err < 1e-12, label);
}

// ---------------------------------------------------------------------------
// Test 4: Gradient matrices -- verify F*e0 + f_const gives correct gradient
// at U=0 by finite-differencing the cost w.r.t. x0
//
// At U=0, the condensed gradient is g = F*e0 + f_const.
// We verify F by checking that d(g)/d(x0) = F via finite differences.
// Specifically, for each component j of x0, we perturb x0[j] and check
// how the gradient at U=0 changes.
// ---------------------------------------------------------------------------
static void test_gradient_matrices(const TestSetup& ts) {
    std::printf("\n--- Test 4: Gradient matrix F via finite differences ---\n");

    const PrecomputedWindow& w = ts.window;
    int nv = w.n_vars;

    // Base x0 = x_ref_0 + small offset
    double x0_base[NX];
    double x0_offsets[NX] = {0.01, -0.005, 0.003, 0.02, -0.01, 0.004};
    for (int i = 0; i < NX; i++)
        x0_base[i] = w.x_ref_0[i] + x0_offsets[i];

    double U_zero[NVARS];
    std::memset(U_zero, 0, sizeof(U_zero));

    // Compute the gradient at U=0 for perturbed x0
    // grad(x0) = F * (x0 - x_ref_0) + f_const
    // d(grad)/d(x0_j) = F[:, j]

    // Finite-difference approximation of F
    double F_fd[NVARS * NX];  // column-major: F_fd(i, j) = F_fd[i + nv*j]
    const double eps = 1e-7;

    for (int j = 0; j < NX; j++) {
        double x0_p[NX], x0_m[NX];
        std::memcpy(x0_p, x0_base, sizeof(x0_base));
        std::memcpy(x0_m, x0_base, sizeof(x0_base));
        x0_p[j] += eps;
        x0_m[j] -= eps;

        // grad_p = H * 0 + g(x0_p) = F * (x0_p - x_ref_0) + f_const
        // But we can also do it via the stagewise cost finite diff to be
        // truly independent. Let's use the stagewise cost.
        double grad_p[NVARS], grad_m[NVARS];

        // Finite-diff the stagewise cost gradient at U=0 w.r.t. each U[i]
        for (int i = 0; i < NVARS; i++) {
            double Upi[NVARS], Umi[NVARS];
            std::memset(Upi, 0, sizeof(Upi));
            std::memset(Umi, 0, sizeof(Umi));
            Upi[i] = eps;
            Umi[i] = -eps;

            double xt_pp[(N_HORIZON + 1) * NX], xt_pm[(N_HORIZON + 1) * NX];
            double xt_mp[(N_HORIZON + 1) * NX], xt_mm[(N_HORIZON + 1) * NX];

            forward_simulate(ts, x0_p, Upi, xt_pp);
            forward_simulate(ts, x0_p, Umi, xt_pm);
            forward_simulate(ts, x0_m, Upi, xt_mp);
            forward_simulate(ts, x0_m, Umi, xt_mm);

            double cp = (compute_cost(ts, xt_pp, Upi) - compute_cost(ts, xt_pm, Umi)) / (2.0 * eps);
            double cm = (compute_cost(ts, xt_mp, Upi) - compute_cost(ts, xt_mm, Umi)) / (2.0 * eps);

            F_fd[i + nv * j] = (cp - cm) / (2.0 * eps);
        }
    }

    // Compare F_fd with w.F
    double max_err = 0.0;
    for (int j = 0; j < NX; j++) {
        for (int i = 0; i < nv; i++) {
            double err = std::fabs(F_fd[i + nv * j] - w.F[i + nv * j]);
            if (err > max_err) max_err = err;
        }
    }

    char label[256];
    std::snprintf(label, sizeof(label),
                  "max|F_fd - F| = %.3e (tol 1e-3)", max_err);
    check(max_err < 1e-3, label);

    // Also verify f_const: at x0 = x_ref_0, U=0 the stagewise gradient
    // should equal f_const (since F*0 + f_const = f_const)
    double grad_at_ref[NVARS];
    for (int i = 0; i < NVARS; i++) {
        double Upi[NVARS], Umi[NVARS];
        std::memset(Upi, 0, sizeof(Upi));
        std::memset(Umi, 0, sizeof(Umi));
        Upi[i] = eps;
        Umi[i] = -eps;

        double xt_p[(N_HORIZON + 1) * NX], xt_m[(N_HORIZON + 1) * NX];
        forward_simulate(ts, w.x_ref_0, Upi, xt_p);
        forward_simulate(ts, w.x_ref_0, Umi, xt_m);

        grad_at_ref[i] = (compute_cost(ts, xt_p, Upi) - compute_cost(ts, xt_m, Umi))
                          / (2.0 * eps);
    }

    double max_err_fc = 0.0;
    for (int i = 0; i < nv; i++) {
        double err = std::fabs(grad_at_ref[i] - w.f_const[i]);
        if (err > max_err_fc) max_err_fc = err;
    }

    std::snprintf(label, sizeof(label),
                  "f_const agreement: max|grad_fd(x_ref,U=0) - f_const| = %.3e (tol 1e-5)",
                  max_err_fc);
    check(max_err_fc < 1e-5, label);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::printf("=== QP condensation tests ===\n\n");

    TestSetup ts;
    setup(ts);

    test_cost_gradient_equivalence(ts);
    test_hessian_symmetry(ts);
    test_cholesky_correctness(ts);
    test_gradient_matrices(ts);

    std::printf("\n=== Summary: %d / %d tests passed ===\n",
                g_tests_passed, g_tests_run);

    if (g_tests_passed == g_tests_run) {
        std::printf("All condensation tests passed.\n");
        return 0;
    } else {
        std::printf("SOME TESTS FAILED.\n");
        return 1;
    }
}
