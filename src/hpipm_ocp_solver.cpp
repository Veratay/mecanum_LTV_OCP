#ifdef MPC_USE_HPIPM

#include "qp_solvers.h"

#include <cstdlib>
#include <cstring>

// BLASFEO headers
extern "C" {
#include <blasfeo_d_aux.h>
}

// HPIPM OCP QP headers
extern "C" {
#include <hpipm_d_ocp_qp_dim.h>
#include <hpipm_d_ocp_qp.h>
#include <hpipm_d_ocp_qp_sol.h>
#include <hpipm_d_ocp_qp_ipm.h>
}

// ---------------------------------------------------------------------------
// Workspace lifecycle
// ---------------------------------------------------------------------------

void hpipm_ocp_workspace_init(HpipmOcpWorkspace& ws, int N)
{
    ws.N_alloc = N;

    // Compute memory sizes
    hpipm_size_t dim_size = d_ocp_qp_dim_memsize(N);
    void* dim_mem = std::malloc(dim_size);
    struct d_ocp_qp_dim dim;
    d_ocp_qp_dim_create(N, &dim, dim_mem);

    // Set dimensions: nx, nu, nb, ng, ns per stage
    int nx[N_MAX + 1], nu_arr[N_MAX + 1], nbx[N_MAX + 1], nbu[N_MAX + 1];
    int ng[N_MAX + 1], ns[N_MAX + 1];

    for (int k = 0; k <= N; ++k) {
        nx[k] = NX;
        nu_arr[k] = (k < N) ? NU : 0;
        nbx[k] = (k == 0) ? NX : 0;  // initial state constraint
        nbu[k] = (k < N) ? NU : 0;    // box constraints on u
        ng[k] = 0;
        ns[k] = 0;
    }

    d_ocp_qp_dim_set_all(nx, nu_arr, nbx, nbu, ng, ns, &dim);

    hpipm_size_t qp_size  = d_ocp_qp_memsize(&dim);
    hpipm_size_t sol_size = d_ocp_qp_sol_memsize(&dim);
    hpipm_size_t arg_size = d_ocp_qp_ipm_arg_memsize(&dim);

    // Need arg to compute ipm_ws size
    void* arg_mem = std::malloc(arg_size);
    struct d_ocp_qp_ipm_arg arg;
    d_ocp_qp_ipm_arg_create(&dim, &arg, arg_mem);
    d_ocp_qp_ipm_arg_set_default(SPEED, &arg);

    hpipm_size_t ipm_size = d_ocp_qp_ipm_ws_memsize(&dim, &arg);

    std::free(arg_mem);
    std::free(dim_mem);

    // Single allocation
    hpipm_size_t total = sizeof(struct d_ocp_qp_dim)
                       + sizeof(struct d_ocp_qp)
                       + sizeof(struct d_ocp_qp_sol)
                       + sizeof(struct d_ocp_qp_ipm_arg)
                       + sizeof(struct d_ocp_qp_ipm_ws)
                       + dim_size + qp_size + sol_size + arg_size + ipm_size
                       + 6 * 64;  // alignment padding

    ws.memory = std::malloc(total);
    ws.memory_size = static_cast<int>(total);
    std::memset(ws.memory, 0, total);
}

void hpipm_ocp_workspace_free(HpipmOcpWorkspace& ws)
{
    if (ws.memory) {
        std::free(ws.memory);
        ws.memory = nullptr;
    }
    ws.memory_size = 0;
    ws.N_alloc = 0;
}

// ---------------------------------------------------------------------------
// OCP QP solve
// ---------------------------------------------------------------------------
int hpipm_ocp_qp_solve(const double* A_list, const double* B_list,
                        const double* Q, const double* Qf, const double* R,
                        const double* x_ref_consistent, const double* u_ref,
                        const double x0[NX],
                        double u_min, double u_max, int N,
                        double* U_out,
                        HpipmOcpWorkspace& ws)
{
    if (ws.N_alloc != N) {
        hpipm_ocp_workspace_free(ws);
        hpipm_ocp_workspace_init(ws, N);
    }

    char* ptr = static_cast<char*>(ws.memory);
    auto align64 = [](char*& p) {
        p = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(p) + 63) & ~63ULL);
    };

    // Place structs
    auto* dim    = reinterpret_cast<struct d_ocp_qp_dim*>(ptr);     ptr += sizeof(*dim);
    auto* qp     = reinterpret_cast<struct d_ocp_qp*>(ptr);        ptr += sizeof(*qp);
    auto* sol    = reinterpret_cast<struct d_ocp_qp_sol*>(ptr);    ptr += sizeof(*sol);
    auto* arg    = reinterpret_cast<struct d_ocp_qp_ipm_arg*>(ptr); ptr += sizeof(*arg);
    auto* ipm_ws = reinterpret_cast<struct d_ocp_qp_ipm_ws*>(ptr); ptr += sizeof(*ipm_ws);

    // Dim memory
    align64(ptr);
    d_ocp_qp_dim_create(N, dim, ptr);
    ptr += d_ocp_qp_dim_memsize(N);

    // Set dimensions
    int nx[N_MAX + 1], nu_arr[N_MAX + 1], nbx[N_MAX + 1], nbu[N_MAX + 1];
    int ng_arr[N_MAX + 1], ns_arr[N_MAX + 1];

    for (int k = 0; k <= N; ++k) {
        nx[k] = NX;
        nu_arr[k] = (k < N) ? NU : 0;
        nbx[k] = (k == 0) ? NX : 0;
        nbu[k] = (k < N) ? NU : 0;
        ng_arr[k] = 0;
        ns_arr[k] = 0;
    }
    d_ocp_qp_dim_set_all(nx, nu_arr, nbx, nbu, ng_arr, ns_arr, dim);

    // QP memory
    align64(ptr);
    d_ocp_qp_create(dim, qp, ptr);
    ptr += d_ocp_qp_memsize(dim);

    // Sol memory
    align64(ptr);
    d_ocp_qp_sol_create(dim, sol, ptr);
    ptr += d_ocp_qp_sol_memsize(dim);

    // Arg memory
    align64(ptr);
    d_ocp_qp_ipm_arg_create(dim, arg, ptr);
    d_ocp_qp_ipm_arg_set_default(SPEED, arg);
    ptr += d_ocp_qp_ipm_arg_memsize(dim);

    // IPM workspace memory
    align64(ptr);
    d_ocp_qp_ipm_ws_create(dim, arg, ipm_ws, ptr);

    // --- Set problem data ---

    // Temporary vectors for gradient terms
    double q_k[NX], r_k[NU];
    double lb[NU], ub[NU];
    int idxbu[NU], idxbx[NX];

    for (int j = 0; j < NU; ++j) {
        lb[j] = u_min;
        ub[j] = u_max;
        idxbu[j] = j;
    }
    for (int i = 0; i < NX; ++i)
        idxbx[i] = i;

    // Per-stage data
    for (int k = 0; k < N; ++k) {
        const double* A_k = A_list + k * NX * NX;
        const double* B_k = B_list + k * NX * NU;
        const double* x_ref_k = x_ref_consistent + k * NX;
        const double* u_ref_k = u_ref + k * NU;

        d_ocp_qp_set_A(k, const_cast<double*>(A_k), qp);
        d_ocp_qp_set_B(k, const_cast<double*>(B_k), qp);

        // b_k = 0 (dynamics: x_{k+1} = A_k x_k + B_k u_k, no affine term
        // since we use consistent reference)
        double b_k[NX];
        std::memset(b_k, 0, NX * sizeof(double));
        d_ocp_qp_set_b(k, b_k, qp);

        // Stage cost: Q, R
        d_ocp_qp_set_Q(k, const_cast<double*>(Q), qp);
        d_ocp_qp_set_R(k, const_cast<double*>(R), qp);

        // Linear terms: q_k = -Q * x_ref_k, r_k = -R * u_ref_k
        for (int i = 0; i < NX; ++i) {
            q_k[i] = 0.0;
            for (int j = 0; j < NX; ++j)
                q_k[i] -= Q[i + NX * j] * x_ref_k[j];
        }
        d_ocp_qp_set_q(k, q_k, qp);

        for (int i = 0; i < NU; ++i) {
            r_k[i] = 0.0;
            for (int j = 0; j < NU; ++j)
                r_k[i] -= R[i + NU * j] * u_ref_k[j];
        }
        d_ocp_qp_set_r(k, r_k, qp);

        // Box constraints on u
        d_ocp_qp_set_idxbu(k, idxbu, qp);
        d_ocp_qp_set_lbu(k, lb, qp);
        d_ocp_qp_set_ubu(k, ub, qp);
    }

    // Terminal cost
    const double* x_ref_N = x_ref_consistent + N * NX;
    d_ocp_qp_set_Q(N, const_cast<double*>(Qf), qp);

    double qf_k[NX];
    for (int i = 0; i < NX; ++i) {
        qf_k[i] = 0.0;
        for (int j = 0; j < NX; ++j)
            qf_k[i] -= Qf[i + NX * j] * x_ref_N[j];
    }
    d_ocp_qp_set_q(N, qf_k, qp);

    // Initial state constraint: x_0 = x0
    d_ocp_qp_set_idxbx(0, idxbx, qp);
    d_ocp_qp_set_lbx(0, const_cast<double*>(x0), qp);
    d_ocp_qp_set_ubx(0, const_cast<double*>(x0), qp);

    // Solve
    d_ocp_qp_ipm_solve(qp, sol, arg, ipm_ws);

    // Extract solution (u_k for k=0..N-1)
    for (int k = 0; k < N; ++k) {
        double u_k[NU];
        d_ocp_qp_sol_get_u(k, sol, u_k);
        std::memcpy(U_out + k * NU, u_k, NU * sizeof(double));
    }

    int iter_count = 0;
    d_ocp_qp_ipm_get_iter(ipm_ws, &iter_count);

    return iter_count;
}

#endif // MPC_USE_HPIPM
