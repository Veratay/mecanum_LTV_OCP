#include "heading_lookup.h"
#include "condensing.h"
#include "blas_dispatch.h"
#include "cholesky.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <time.h>

// ---------------------------------------------------------------------------
// Angle wrapping to [-π, π]
// ---------------------------------------------------------------------------
static double angle_wrap(double a)
{
    a = std::fmod(a + M_PI, 2.0 * M_PI);
    if (a < 0.0) a += 2.0 * M_PI;
    return a - M_PI;
}

// ---------------------------------------------------------------------------
// Online B_d reconstruction: trig decomposition
// ---------------------------------------------------------------------------
void heading_lookup_build_B_list(const HeadingLookupData& data,
                                 const double* theta_list, int N,
                                 double* B_list)
{
    for (int k = 0; k < N; ++k) {
        double ct = std::cos(theta_list[k]);
        double st = std::sin(theta_list[k]);
        double* B_k = B_list + k * NX * NU;
        for (int i = 0; i < NX * NU; ++i) {
            B_k[i] = data.B_d0[i] + ct * data.B_dc[i] + st * data.B_ds[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Online B_d reconstruction: table interpolation
// ---------------------------------------------------------------------------
void heading_table_build_B_list(const HeadingTableData& table,
                                const double* theta_list, int N,
                                double* B_list)
{
    const int M = table.M;
    const double step = 2.0 * M_PI / M;

    for (int k = 0; k < N; ++k) {
        // Normalize theta to [0, 2π)
        double theta = std::fmod(theta_list[k], 2.0 * M_PI);
        if (theta < 0.0) theta += 2.0 * M_PI;

        // Find bracketing indices
        double idx_f = theta / step;
        int i0 = static_cast<int>(std::floor(idx_f)) % M;
        int i1 = (i0 + 1) % M;
        double alpha = idx_f - std::floor(idx_f);

        const double* B0 = table.B_d_table + i0 * NX * NU;
        const double* B1 = table.B_d_table + i1 * NX * NU;
        double* B_k = B_list + k * NX * NU;

        for (int i = 0; i < NX * NU; ++i) {
            B_k[i] = (1.0 - alpha) * B0[i] + alpha * B1[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Default heading schedule config from model params
// ---------------------------------------------------------------------------
HeadingScheduleConfig heading_schedule_config_from_params(const ModelParams& params)
{
    double lxy = params.lx + params.ly;
    double r = params.wheel_radius;

    HeadingScheduleConfig cfg;
    cfg.alpha_0 = 4.0 * lxy * params.stall_torque / (r * params.inertia);
    cfg.omega_max = params.free_speed * r / lxy;
    cfg.v_max = params.free_speed * r;
    cfg.heading_gain = 5.0;
    return cfg;
}

// ---------------------------------------------------------------------------
// Generate feasible heading schedule
// ---------------------------------------------------------------------------
void generate_heading_schedule(const double x0[NX], const RefNode* ref_window,
                               int N, double dt,
                               const HeadingScheduleConfig& sched_config,
                               double* theta_out)
{
    double theta = x0[2];
    double omega = x0[5];

    theta_out[0] = theta;

    for (int k = 0; k < N; ++k) {
        // Estimate field velocity from reference
        double vx = ref_window[k].x_ref[3];
        double vy = ref_window[k].x_ref[4];
        double v_field = std::sqrt(vx * vx + vy * vy);

        // Available angular acceleration with derating
        double abs_omega = std::fabs(omega);
        double headroom = 1.0 - abs_omega / sched_config.omega_max
                              - v_field / sched_config.v_max;
        if (headroom < 0.0) headroom = 0.0;
        double alpha_max = sched_config.alpha_0 * headroom;

        // Desired omega to track reference heading
        double e_heading = angle_wrap(ref_window[k + 1].x_ref[2] - theta);
        double omega_des = sched_config.heading_gain * e_heading;

        // Clamp angular acceleration
        double omega_next = std::clamp(omega_des,
                                        omega - alpha_max * dt,
                                        omega + alpha_max * dt);
        // Clamp omega magnitude
        omega_next = std::clamp(omega_next,
                                -sched_config.omega_max,
                                 sched_config.omega_max);

        theta = theta + omega_next * dt;
        omega = omega_next;
        theta_out[k + 1] = theta;
    }
}

// ---------------------------------------------------------------------------
// Common condensed solve implementation
// ---------------------------------------------------------------------------
static QPSolution heading_solve_condensed_impl(const double* A_d,
                                                const double* B_list,
                                                const RefNode* ref_window,
                                                const double* theta_sched,
                                                const double x0[NX],
                                                const MPCConfig& config,
                                                QpSolverType solver_type,
                                                SolverContext& ctx)
{
    const int N = config.N;

    // Build A_list as N copies of A_d
    double A_list[N_MAX * NX * NX];
    for (int k = 0; k < N; ++k)
        std::memcpy(A_list + k * NX * NX, A_d, NX * NX * sizeof(double));

    // Build consistent reference: x_{k+1} = A_d·x_k + B_k·u_ref_k
    double x_ref_consistent[(N_MAX + 1) * NX];
    std::memcpy(x_ref_consistent, ref_window[0].x_ref, NX * sizeof(double));

    double u_ref_stacked[N_MAX * NU];
    double temp_Ax[NX], temp_Bu[NX];

    for (int k = 0; k < N; ++k) {
        std::memcpy(u_ref_stacked + k * NU, ref_window[k].u_ref, NU * sizeof(double));

        const double* x_k = x_ref_consistent + k * NX;
        double* x_next = x_ref_consistent + (k + 1) * NX;
        const double* B_k = B_list + k * NX * NU;

        mpc_linalg::gemv(NX, NX, A_d, x_k, temp_Ax);
        mpc_linalg::gemv(NX, NU, B_k, ref_window[k].u_ref, temp_Bu);

        for (int i = 0; i < NX; ++i)
            x_next[i] = temp_Ax[i] + temp_Bu[i];
    }

    // Condense window using existing infrastructure
    PrecomputedWindow window;
    condense_window(A_list, B_list, x_ref_consistent, u_ref_stacked,
                    config, window);

    // Solve using existing solver dispatch
    return mpc_solve_with_solver(window, x0, config, solver_type, ctx);
}

// ---------------------------------------------------------------------------
// Trig decomposition solve
// ---------------------------------------------------------------------------
QPSolution heading_lookup_solve_condensed(const HeadingLookupData& data,
                                          const RefNode* ref_window,
                                          const double x0[NX],
                                          const MPCConfig& config,
                                          const HeadingScheduleConfig& sched_config,
                                          QpSolverType solver_type,
                                          SolverContext& ctx)
{
    const int N = config.N;

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    // Generate feasible heading schedule
    double theta_sched[N_MAX + 1];
    generate_heading_schedule(x0, ref_window, N, config.dt, sched_config, theta_sched);

    // Build B_list from trig decomposition
    double B_list[N_MAX * NX * NU];
    heading_lookup_build_B_list(data, theta_sched, N, B_list);

    QPSolution sol = heading_solve_condensed_impl(data.A_d, B_list, ref_window,
                                                   theta_sched, x0, config,
                                                   solver_type, ctx);

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    sol.solve_time_ns = (t_end.tv_sec - t_start.tv_sec) * 1e9
                      + (t_end.tv_nsec - t_start.tv_nsec);
    return sol;
}

// ---------------------------------------------------------------------------
// Table interpolation solve
// ---------------------------------------------------------------------------
QPSolution heading_table_solve_condensed(const HeadingTableData& table,
                                         const RefNode* ref_window,
                                         const double x0[NX],
                                         const MPCConfig& config,
                                         const HeadingScheduleConfig& sched_config,
                                         QpSolverType solver_type,
                                         SolverContext& ctx)
{
    const int N = config.N;

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    // Generate feasible heading schedule
    double theta_sched[N_MAX + 1];
    generate_heading_schedule(x0, ref_window, N, config.dt, sched_config, theta_sched);

    // Build B_list from table interpolation
    double B_list[N_MAX * NX * NU];
    heading_table_build_B_list(table, theta_sched, N, B_list);

    QPSolution sol = heading_solve_condensed_impl(table.A_d, B_list, ref_window,
                                                   theta_sched, x0, config,
                                                   solver_type, ctx);

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    sol.solve_time_ns = (t_end.tv_sec - t_start.tv_sec) * 1e9
                      + (t_end.tv_nsec - t_start.tv_nsec);
    return sol;
}
