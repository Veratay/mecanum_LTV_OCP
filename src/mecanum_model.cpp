// mecanum_model.cpp -- Mecanum drivetrain model with DC motor torque-speed curve
//
// Implements the 3x4 mecanum Jacobian (duty cycle -> body wrench) and the
// continuous-time LTV state-space matrices Ac(theta), Bc(theta) used by
// the MPC linearization.
//
// Motor model (per wheel):
//   torque_i = stall_torque * (d_i  -  omega_wheel_i / free_speed)
//   force_i  = torque_i / wheel_radius
//
// Wheel speeds are derived from robot-frame (body) velocities:
//   1. Rotate world velocities to robot frame:
//        vx_body =  cos(theta)*vx_world + sin(theta)*vy_world
//        vy_body = -sin(theta)*vx_world + cos(theta)*vy_world
//   2. Mecanum inverse kinematics (in robot frame):
//        omega_wheel_i = (vx_body ± vy_body ± L*omega) / r

#include "mecanum_model.h"

#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
// compute_mecanum_jacobian
// ---------------------------------------------------------------------------
//
// Builds the 3x4 matrix J_mec that maps wheel duty cycles [d_FL, d_FR, d_RL, d_RR]
// to the body-frame wrench [Fx, Fy, tau] at zero wheel speed (stall condition).
//
// Each wheel's stall force is:  F_i = (stall_torque * d_i) / r
//
// The mecanum geometry sums these forces into body wrench:
//   Fx  = F_FL + F_FR + F_RL + F_RR                    (all push forward)
//   Fy  = -F_FL + F_FR + F_RL - F_RR                   (diagonal roller pattern)
//   tau = (-F_FL + F_FR - F_RL + F_RR) * (lx+ly)       (rotation about center)
//
// So J_mec = (stall_torque / r) * sign_matrix,  stored column-major.
//
void compute_mecanum_jacobian(ModelParams& params)
{
    const double r   = params.wheel_radius;
    const double ts  = params.stall_torque;
    const double lxy = params.lx + params.ly;

    const double s = ts / r;   // stall force per unit duty cycle per wheel

    // Column 0: FL wheel  (signs: +1, -1, -(lx+ly))
    params.J_mec[0 + 3 * 0] =  s;
    params.J_mec[1 + 3 * 0] = -s;
    params.J_mec[2 + 3 * 0] = -s * lxy;

    // Column 1: FR wheel  (signs: +1, +1, +(lx+ly))
    params.J_mec[0 + 3 * 1] =  s;
    params.J_mec[1 + 3 * 1] =  s;
    params.J_mec[2 + 3 * 1] =  s * lxy;

    // Column 2: RL wheel  (signs: +1, +1, -(lx+ly))
    params.J_mec[0 + 3 * 2] =  s;
    params.J_mec[1 + 3 * 2] =  s;
    params.J_mec[2 + 3 * 2] = -s * lxy;

    // Column 3: RR wheel  (signs: +1, -1, +(lx+ly))
    params.J_mec[0 + 3 * 3] =  s;
    params.J_mec[1 + 3 * 3] = -s;
    params.J_mec[2 + 3 * 3] =  s * lxy;
}

// ---------------------------------------------------------------------------
// continuous_dynamics
// ---------------------------------------------------------------------------
//
// Fills Ac (6x6) and Bc (6x4), both column-major, for the state:
//
//   x = [px, py, theta, vx_world, vy_world, omega]
//
// The dynamics follow the same chain as the Python reference
// (gen_mpc_native.py):
//
//   1. Rotate world velocities to robot frame:
//        vx_body =  cos(θ)·vx_w + sin(θ)·vy_w
//        vy_body = -sin(θ)·vx_w + cos(θ)·vy_w
//
//   2. Wheel angular velocities (robot-frame inverse kinematics):
//        ω_FL = (vx_body - vy_body - L·ω) / r
//        ω_FR = (vx_body + vy_body + L·ω) / r
//        ω_RL = (vx_body + vy_body - L·ω) / r
//        ω_RR = (vx_body - vy_body + L·ω) / r
//
//   3. Motor torque from torque-speed curve:
//        τ_i = τ_stall · (d_i − ω_wheel_i / ω_free)
//
//   4. Wheel contact force:
//        F_i = τ_i / r
//
//   5. Sum into body-frame wrench via mecanum geometry:
//        [Fx_body, Fy_body, τ_body]
//
//   6. Rotate forces back to world frame:
//        [Fx_w, Fy_w] = R(θ) · [Fx_body, Fy_body]
//
// Linearizing, this splits into:
//   - Input coupling (Bc): stall force J_mec rotated to world frame (same as before)
//   - Back-EMF damping (Ac): velocity-dependent torque reduction
//
// The back-EMF contributes an additional damping in body frame.  Two 1/r
// factors appear:
//   • ω_wheel = v_body / r        (step 2: kinematics)
//   • F = τ / r                    (step 4: torque-to-force)
//
// Combined per wheel: F_emf = -(τ_stall / ω_free) · v_body_combo / r / r
//
// Summing over 4 wheels, the body-frame damping matrix is diagonal:
//   D_body = (4 · τ_stall / (ω_free · r²)) · diag(1, 1, L²)
//
// Rotating the 2×2 force block to world frame:
//   D_world_force = R(θ) · (scalar · I₂) · R(-θ) = scalar · I₂
//   (isotropic — rotation has no effect)
//
// So the back-EMF simply adds to the existing viscous damping coefficients.
//
void continuous_dynamics(double theta, const ModelParams& params,
                         double Ac[NX * NX], double Bc[NX * NU])
{
    // ------------------------------------------------------------------
    // Zero-initialise both matrices
    // ------------------------------------------------------------------
    std::memset(Ac, 0, NX * NX * sizeof(double));
    std::memset(Bc, 0, NX * NU * sizeof(double));

    const double m    = params.mass;
    const double J    = params.inertia;
    const double d_l  = params.damping_linear;
    const double d_a  = params.damping_angular;
    const double r    = params.wheel_radius;
    const double lxy  = params.lx + params.ly;
    const double ts   = params.stall_torque;
    const double wf   = params.free_speed;

    const double ct = std::cos(theta);
    const double st = std::sin(theta);

    // ------------------------------------------------------------------
    // Ac  (column-major: element (row, col) at index [row + NX*col])
    // ------------------------------------------------------------------

    // Position kinematics
    Ac[0 + NX * 3] = 1.0;   // px_dot = vx_world
    Ac[1 + NX * 4] = 1.0;   // py_dot = vy_world
    Ac[2 + NX * 5] = 1.0;   // theta_dot = omega

    // Back-EMF damping from motor torque-speed relationship.
    //
    // Per-wheel back-EMF force (in body frame):
    //   F_emf_i = -(τ_stall / ω_free) · ω_wheel_i / r
    //           = -(τ_stall / ω_free) · v_body_combo_i / r²
    //                     ↑ torque-speed      ↑ kinematics  ↑ torque→force
    //
    // After summing 4 wheels (each contributes +vx_body to Fx):
    //   d_emf_lin = 4 · τ_stall / (ω_free · r²)
    //   d_emf_ang = 4 · L² · τ_stall / (ω_free · r²)
    //
    const double emf_per_wheel = ts / (wf * r * r);  // force damping per unit body vel per wheel
    const double d_emf_lin = 4.0 * emf_per_wheel;
    const double d_emf_ang = 4.0 * lxy * lxy * emf_per_wheel;

    // Total velocity damping (viscous friction + motor back-EMF)
    Ac[3 + NX * 3] = -(d_l + d_emf_lin) / m;
    Ac[4 + NX * 4] = -(d_l + d_emf_lin) / m;
    Ac[5 + NX * 5] = -(d_a + d_emf_ang) / J;

    // ------------------------------------------------------------------
    // Bc  (input coupling: J_mec rotated from body to world frame)
    // ------------------------------------------------------------------
    //
    // J_mec maps duty cycles to body-frame wrench at stall (step 5).
    // We rotate the force part to world frame (step 6):
    //   Bc(3, j) = (1/m) · ( cos(θ)·Fx_j − sin(θ)·Fy_j )
    //   Bc(4, j) = (1/m) · ( sin(θ)·Fx_j + cos(θ)·Fy_j )
    //   Bc(5, j) = (1/J) · τ_j
    //
    const double inv_m = 1.0 / m;
    const double inv_J = 1.0 / J;

    for (int j = 0; j < NU; ++j) {
        const double Fx_j  = params.J_mec[0 + 3 * j];
        const double Fy_j  = params.J_mec[1 + 3 * j];
        const double tau_j = params.J_mec[2 + 3 * j];

        Bc[3 + NX * j] = inv_m * (ct * Fx_j - st * Fy_j);
        Bc[4 + NX * j] = inv_m * (st * Fx_j + ct * Fy_j);
        Bc[5 + NX * j] = inv_J * tau_j;
    }
}
