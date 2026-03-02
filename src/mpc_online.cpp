#include "mpc_online.h"
#include "blas_dispatch.h"
#include "box_qp_solver.h"
#include <time.h>

QPSolution mpc_solve_online(const PrecomputedWindow& window, const double x0[NX],
                            const MPCConfig& config, BoxQPWorkspace& workspace)
{
    const int n_vars = window.n_vars;

    // ---- start timing ----
    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    // Step 1: error e0 = x0 - x_ref_0
    double e0[NX];
    for (int i = 0; i < NX; ++i) {
        e0[i] = x0[i] - window.x_ref_0[i];
    }

    // Step 2: form gradient  g = F * e0 + f_const
    //   workspace.grad = F * e0
    mpc_linalg::gemv(n_vars, NX, window.F, e0, workspace.grad);
    //   workspace.grad += 1.0 * f_const
    mpc_linalg::axpy(n_vars, 1.0, window.f_const, workspace.grad);

    // Step 3: unconstrained solve via precomputed Cholesky
    //   negate gradient into temp
    for (int i = 0; i < n_vars; ++i) {
        workspace.temp[i] = -workspace.grad[i];
    }
    //   L * rhs = -grad   (forward substitution)
    mpc_linalg::trsv_lower(n_vars, window.L, workspace.temp, workspace.rhs);
    //   L' * U = rhs      (back substitution)
    mpc_linalg::trsv_upper_trans(n_vars, window.L, workspace.rhs, workspace.U);

    // Step 4: feasibility check
    bool feasible = is_feasible(workspace.U, n_vars, config.V_min, config.V_max);

    int n_iter = 0;
    if (!feasible) {
        // Active-set box QP solve (warm-started from the unconstrained solution in workspace.U)
        n_iter = box_qp_solve(window.H, window.L, workspace.grad,
                              config.V_min, config.V_max, n_vars, 10, workspace);
    }

    // ---- stop timing ----
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed_ns = (t_end.tv_sec - t_start.tv_sec) * 1e9
                      + (t_end.tv_nsec - t_start.tv_nsec);

    // Step 5: package solution
    QPSolution sol;

    // First NU elements -> u0
    for (int i = 0; i < NU; ++i) {
        sol.u0[i] = workspace.U[i];
    }

    // Full input sequence
    for (int i = 0; i < n_vars; ++i) {
        sol.U[i] = workspace.U[i];
    }

    sol.n_iterations = feasible ? 0 : n_iter;

    // Count active constraints (elements at bounds)
    int n_active = 0;
    for (int i = 0; i < n_vars; ++i) {
        if (workspace.U[i] <= config.V_min || workspace.U[i] >= config.V_max) {
            ++n_active;
        }
    }
    sol.n_active = n_active;
    sol.solve_time_ns = elapsed_ns;

    return sol;
}
