// mecanum_model.cpp -- Mecanum drivetrain kinematic model
//
// Implements the 3x4 mecanum Jacobian (voltage -> body wrench) and the
// continuous-time LTV state-space matrices Ac(theta), Bc(theta) used by
// the MPC linearization.

#include "mecanum_model.h"

#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
// compute_mecanum_jacobian
// ---------------------------------------------------------------------------
//
// Builds the 3x4 matrix J_mec that maps wheel voltages [V_FL, V_FR, V_RL, V_RR]
// to the body-frame wrench [Fx, Fy, tau].
//
// Derivation (virtual-work duality):
//   The standard mecanum inverse kinematics maps body twist [vx, vy, omega]
//   to wheel angular velocities:
//
//     w_FL = (1/r)(vx - vy - (lx+ly)*omega)
//     w_FR = (1/r)(vx + vy + (lx+ly)*omega)
//     w_RL = (1/r)(vx + vy - (lx+ly)*omega)
//     w_RR = (1/r)(vx - vy + (lx+ly)*omega)
//
//   Written as omega_wheels = J_inv * twist, where J_inv is 4x3.
//   By virtual work:  twist^T * wrench = omega_wheels^T * torques
//                     => wrench = J_inv^T * torques
//
//   With a simplified DC motor model (torque = motor_kv * voltage):
//     wrench = J_inv^T * motor_kv * V
//
//   Therefore J_mec = motor_kv * J_inv^T, giving:
//
//     J_mec = (motor_kv / r) * [ 1   1   1   1  ]   <- Fx row
//                               [-1   1   1  -1  ]   <- Fy row
//                               [-(lx+ly)  (lx+ly)  -(lx+ly)  (lx+ly)]  <- tau row
//
//   Stored column-major: J_mec[row + 3*col].
//
void compute_mecanum_jacobian(ModelParams& params)
{
    const double r   = params.wheel_radius;
    const double kv  = params.motor_kv;
    const double lxy = params.lx + params.ly;   // half-track sum

    const double s = kv / r;   // common scale factor

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
// Fills the continuous-time LTV matrices Ac (6x6) and Bc (6x4), both
// column-major, for the state vector:
//
//   x = [px, py, theta, vx_world, vy_world, omega]
//
// Dynamics:
//   px_dot        = vx_world
//   py_dot        = vy_world
//   theta_dot     = omega
//   vx_world_dot  = -(d_lin / m) * vx_world  +  (1/m) * [R(theta) * F_body]_x
//   vy_world_dot  = -(d_lin / m) * vy_world  +  (1/m) * [R(theta) * F_body]_y
//   omega_dot     = -(d_ang / J) * omega      +  (1/J) * tau
//
// where F_body = J_mec[0:2, :] * V  and  tau = J_mec[2, :] * V,
// and R(theta) rotates body-frame forces into the world frame.
//
// Ac encodes the linear/damping part; Bc encodes the input coupling
// (with the heading-dependent rotation applied to the force rows).
//
void continuous_dynamics(double theta, const ModelParams& params,
                         double Ac[NX * NX], double Bc[NX * NU])
{
    // ------------------------------------------------------------------
    // Zero-initialise both matrices
    // ------------------------------------------------------------------
    std::memset(Ac, 0, NX * NX * sizeof(double));
    std::memset(Bc, 0, NX * NU * sizeof(double));

    // Shorthand
    const double m    = params.mass;
    const double J    = params.inertia;
    const double d_l  = params.damping_linear;
    const double d_a  = params.damping_angular;

    const double ct = std::cos(theta);
    const double st = std::sin(theta);

    // ------------------------------------------------------------------
    // Ac  (column-major: element (row, col) at index [row + NX*col])
    // ------------------------------------------------------------------

    // Position-to-velocity coupling (rows 0-2, cols 3-5):
    //   px_dot  = vx_world   -> Ac(0,3) = 1
    //   py_dot  = vy_world   -> Ac(1,4) = 1
    //   theta_dot = omega    -> Ac(2,5) = 1
    Ac[0 + NX * 3] = 1.0;
    Ac[1 + NX * 4] = 1.0;
    Ac[2 + NX * 5] = 1.0;

    // Velocity damping (rows 3-5, cols 3-5):
    //   vx_world_dot += -(d_lin/m) * vx_world
    //   vy_world_dot += -(d_lin/m) * vy_world
    //   omega_dot    += -(d_ang/J) * omega
    Ac[3 + NX * 3] = -d_l / m;
    Ac[4 + NX * 4] = -d_l / m;
    Ac[5 + NX * 5] = -d_a / J;

    // ------------------------------------------------------------------
    // Bc  (column-major: element (row, col) at index [row + NX*col])
    // ------------------------------------------------------------------
    //
    // For each wheel column j (0..3):
    //   Bc(3, j) = (1/m) * ( cos(theta)*J_mec(0,j) - sin(theta)*J_mec(1,j) )
    //   Bc(4, j) = (1/m) * ( sin(theta)*J_mec(0,j) + cos(theta)*J_mec(1,j) )
    //   Bc(5, j) = (1/J) * J_mec(2, j)
    //
    // J_mec is 3x4 column-major: element (r, c) at J_mec[r + 3*c].
    //
    const double inv_m = 1.0 / m;
    const double inv_J = 1.0 / J;

    for (int j = 0; j < NU; ++j) {
        const double Fx_j  = params.J_mec[0 + 3 * j];   // body-frame Fx per volt
        const double Fy_j  = params.J_mec[1 + 3 * j];   // body-frame Fy per volt
        const double tau_j = params.J_mec[2 + 3 * j];   // body-frame tau per volt

        // Rotate body-frame force into world frame and scale by 1/m
        Bc[3 + NX * j] = inv_m * (ct * Fx_j - st * Fy_j);
        Bc[4 + NX * j] = inv_m * (st * Fx_j + ct * Fy_j);

        // Torque does not require rotation
        Bc[5 + NX * j] = inv_J * tau_j;
    }
}
