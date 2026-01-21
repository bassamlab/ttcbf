# Copyright (c) 2026, Professorship for Adaptive Behavior of Autonomous Vehicles, University of the Bundeswehr Munich.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from typing import List, Tuple
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.lines import Line2D
from matplotlib import animation
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from common import CorridorTrajectoryHistory, cprint
from enum import Enum, auto
import pickle
from pathlib import Path
from tqdm.auto import tqdm


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "font.family": "serif",
        "text.usetex": True,  # Set to True if you have LaTeX installed
    }
)


class ClassKType(Enum):
    LINEAR = auto()
    EXP = auto()
    RATIONAL = auto()


class CBFMethod(Enum):
    HOCBF = auto()  # Higher-Order CBF
    PACBF = auto()  # Parameter-Adaptive CBF
    RACBF = auto()  # Relaxation-Adaptive CBF
    TTCBF = auto()  # Taylor-Truncated CBF
    aTTCBF = auto()  # Adaptive TTCBF


# ─────────────────────────── simulation settings ────────────────────────────
DT: float = 0.05  # [s] sampling period
HORIZON_HP: int = 2  # n-step horizon for TTCBF (Taylor)
DT_T = DT * HORIZON_HP  # Taylor step size
T_MAX: float = 4.0  # [s] maximum wall-clock horizon

CBF_METHOD = CBFMethod.HOCBF  # "HOCBF" or "TTCBF"
CLASS_K = ClassKType.LINEAR
LAMBDA_1: float = 3  # HOCBF λ₁
LAMBDA_2: float = 3  # HOCBF λ₂
LAMBDA_TTCBF: float = 0.2  # TTCBF λ  (0 < λ ≤ 1)
R_ROBOT: float = 2.0  # [m] robot radius
R_OBS: float = 4.0  # [m] waypoint-obstacle radius

IS_PERCEPTION_UNCERTAINTY = False  # Whether to add perception uncertainty
OBS_POS_UNCERTAINTY_BOUND = (
    0.5  # [m] uncertainty level of obstacle position, modeled by white noise
)

IS_SLACK_CBF = True

# Parameter-Adaptive CBF (PACBF)
P1_INIT = 0.2  # p1(0)  ≥ 0   (initial value of p1)
P2_CONST = 2.0  # p2(t)  ≜  λ2
P1_MAX = 20
P1_DESIRED = 0.1  # a desired samll value to avoid the p1 being too large. A CLF will be used to drive p1 to this value
W_SLACK_CLF_P1 = 0.01
W_SLACK_CLF_P2 = 0.01

LAMBDA_P1 = 10
W_NU_PACBF = 1.0  # quadratic cost weight ‖ν1‖²


# RACBF constants  ───────────────
K1_RACBF = 4.0  # α1(s)=k1·s  in ψ1
K2_RACBF = 4.0  # α2(s)=k2·s  in ψ2
R_INIT = 0.05  # r_i(0)  (initial relaxation gap)
R_TARGET = 0.05  # r*   (desired steady-state gap)
LAMBDA_R_1 = 2.0  # λ_r  in ṙ = ν + λ_r r
LAMBDA_R_2 = 2.0
C_CLF_PACBF = 4.0  # CLF parameter for PACBF
W_SLACK_CLF_R = 50  # weight on δ_r
W_NU_RACBF = 10.0  # weight on ν


# CLF parameters
V_REF: float = 10.0  # [m/s] desired cruising speed
C_CLF: float = 4.0  # CLF parameter

# Control bounds
U1_MAX: float = 2.0  # [rad/s] steering-rate bound
U2_MAX: float = 2.0  # [m/s²] acceleration bound

# QP objective weights
R_NOM: np.ndarray = np.array([1, 1])
W_SLACK_CLF: float = 100.0  # slack penalty (large → CLF *soft*)
W_SLACK_CBF: float = 1e6  # slack penalty for CBF (large → CBF *soft*)
W_LAMBDA: float = 500  # 500 works well

# ─────────────────────────── map-related data ────────────────────────────
# Number of points to place on the circle
NUM_INNER_OBS = 8
NUM_OUTER_OBS = 8
NUM_OBS_CONSIDER = (
    16  # The number of considered most nearing obstacles in CBF constraints
)
NUM_CBF_CONS = (
    2 + NUM_OBS_CONSIDER
)  # The number of CBF constraints (number of considered obstacle + two boundaries of the corridor)

CENTER = np.array([0, 0])  # Center of the circle
# Radius of the circle
R_CETERLINE = 40
CORRIDOR_WIDTH = 10
R_INNER = R_CETERLINE - CORRIDOR_WIDTH / 2
R_OUTER = R_CETERLINE + CORRIDOR_WIDTH / 2
inner_obs_angle_list = np.linspace(0.0, 2.0 * np.pi, NUM_INNER_OBS, endpoint=False)
x_coords_inner = CENTER[0] + R_INNER * np.cos(inner_obs_angle_list)
y_coords_outer = CENTER[1] + R_INNER * np.sin(inner_obs_angle_list)
inner_obs_xy_list = np.stack((x_coords_inner, y_coords_outer), axis=1)

OUTER_ANGLE_OFFSET = np.pi / NUM_OUTER_OBS
outer_obs_angle_list = (
    np.linspace(0.0, 2.0 * np.pi, NUM_OUTER_OBS, endpoint=False) + OUTER_ANGLE_OFFSET
)
x_coords_outer = CENTER[0] + R_OUTER * np.cos(outer_obs_angle_list)
y_coords_outer = CENTER[1] + R_OUTER * np.sin(outer_obs_angle_list)
outer_obs_xy_list = np.stack((x_coords_outer, y_coords_outer), axis=1)

all_obs_xy_list = np.concatenate((inner_obs_xy_list, outer_obs_xy_list), axis=0)
all_obs_angle_list = np.concatenate(
    (inner_obs_angle_list, outer_obs_angle_list), axis=0
)

# Get the sort order based on angle
sorted_indices = np.argsort(all_obs_angle_list)
# Apply the sort order to both angle and position lists
all_obs_angle_list = all_obs_angle_list[sorted_indices]
all_obs_xy_list = all_obs_xy_list[sorted_indices]


def get_colormap(num_colors, style="viridis"):
    colormap = plt.colormaps.get_cmap(style)  # viridis, plasma, magma, inferno, cividis
    num_colors_ = max(num_colors, 3)  # Minimum of 3 colors
    colormap = [
        colormap(i / (num_colors_ - 1)) for i in range(num_colors_)
    ]  # Generate colors

    return colormap


# ────────────────────────── I/O utilities ──────────────────────────
def save_history(hist: CorridorTrajectoryHistory, file_path: str | Path) -> None:
    """
    Serialize CorridorTrajectoryHistory to disk using pickle.

    Parameters
    ----------
    hist : CorridorTrajectoryHistory
        Simulation history to save.
    file_path : str or Path
        Target file path, e.g., "data/run_aTTCBF.pkl".
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "wb") as f:
        pickle.dump(hist, f, protocol=pickle.HIGHEST_PROTOCOL)

    cprint(f"[INFO] History saved to {file_path.resolve()}")


def load_history(file_path: str | Path) -> CorridorTrajectoryHistory:
    """
    Load CorridorTrajectoryHistory from disk.

    Parameters
    ----------
    file_path : str or Path
        Path to the pickle file.

    Returns
    -------
    CorridorTrajectoryHistory
    """
    file_path = Path(file_path)
    with open(file_path, "rb") as f:
        hist: CorridorTrajectoryHistory = pickle.load(f)

    print(f"[INFO] History loaded from {file_path.resolve()}")
    return hist

# ╔══════════════════════════════════════════════════════════════════╗
#  Helper functions
# ╚══════════════════════════════════════════════════════════════════╝
def wrap_to_pi(angle: float) -> float:
    """Wrap angle to (-π, π]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


# ──────────────────────── Unicycle dynamics (state 4×1) ─────────────────────
def unicycle_dynamics(state: np.ndarray, control: np.ndarray) -> np.ndarray:
    """
    f(x) + g(x)u with state x = [x, y, θ, v].

    Parameters
    ----------
    state   : ndarray (4,)
    control : ndarray (2,)  [u₁ = θ̇, u₂ = v̇]

    Returns
    -------
    ẋ : ndarray (4,)
    """
    x, y, theta, v = state
    u1, u2 = control
    return np.array([v * math.cos(theta), v * math.sin(theta), u1, u2], dtype=float)


def distance_cbf_terms(
    state: np.ndarray,
    u: cp.Variable,
    obs_pos: np.ndarray,
    R_sq: float,
) -> Tuple[float, float, float, List[float]]:
    """
    Distance-based CBF for a unicycle robot.

    Parameters
    ----------
    state   : array_like, shape (4,)
              [x, y, theta, v]
    control : array_like, shape (2,) or None
              [u1 = theta_dot, u2 = v_dot]; if None, uses symbolic form
    obs_pos : array_like, shape (2,)
              Obstacle centre [x_o, y_o]
    R_sq    : float
              Safety distance squared: (R_robot + R_obst)**2

    Returns
    -------
    h       : float
    dh      : float     (first time derivative)
    d2h     : float     (second time derivative; depends on control)
    """
    x, y, theta, v = map(float, state)
    x_o, y_o = map(float, obs_pos)

    # relative position
    dx, dy = x - x_o, y - y_o

    # 0) the CBF itself
    h = dx**2 + dy**2 - R_sq

    # 1) first derivative  ḣ = 2·(dx·v·cosθ + dy·v·sinθ)
    dh = 2.0 * (dx * v * math.cos(theta) + dy * v * math.sin(theta))

    # 2) second derivative  ḧ = 2v² + 2u₂(dx cosθ + dy sinθ)
    #                                  + 2vu₁(−dx sinθ + dy cosθ)

    d2h = (
        2.0 * v**2
        + 2.0 * v * u[0] * (-dx * math.sin(theta) + dy * math.cos(theta))
        + 2.0 * u[1] * (dx * math.cos(theta) + dy * math.sin(theta))
    )

    d2h_components = [
        2.0 * v**2,
        2.0 * v * (-dx * math.sin(theta) + dy * math.cos(theta)),
        2.0 * (dx * math.cos(theta) + dy * math.sin(theta)),
    ]

    return h, dh, d2h, d2h_components


# ────────────────────────── Way-point utilities ─────────────────────────────
def compute_clf_condition(
    u: cp.Variable,
    delta_theta: cp.Variable,
    c_clf: float,
    e: float,
):
    V = 0.5 * e**2
    dV = e * (-u)
    clf_condition = dV + c_clf * V <= delta_theta

    return clf_condition


# -- Nominal Controller --
def nominal_robot_controller(e_theta: float, e_v: float, kp_theta=1.0, kp_v=1.0):
    """
    Simple P controller with proper circular error handling.

    Parameters
    ----------
    e_theta : float, (-π, π], heading error (theta_desired - theta)
    e_v     : float, speed error (v_desired - v)
    kp_theta, kp_v : float
        Proportional gains for heading and speed.
    """
    # P control
    u1 = np.clip(kp_theta * e_theta, -U1_MAX, U1_MAX)
    u2 = np.clip(kp_v * e_v, -U2_MAX, U2_MAX)

    return np.array([u1, u2])


def argmink(arr, k):
    # Find the indices of the k smallest elements
    idx = np.argpartition(arr, k)[:k]
    return idx[np.argsort(arr[idx])]


# ╔══════════════════════════════════════════════════════════════════╗
#  Random restart
# ╚══════════════════════════════════════════════════════════════════╝
def random_init_state() -> np.ndarray:
    """Return a random start state on the centre-line."""
    # Create a random angle between 0 and 2pi:
    random_idx = np.random.randint(0, len(all_obs_angle_list))
    angle_random_obs = all_obs_angle_list[random_idx]
    angle0 = wrap_to_pi(angle_random_obs + np.pi / 28)

    x0 = CENTER[0] + R_CETERLINE * np.cos(angle0)
    y0 = CENTER[1] + R_CETERLINE * np.sin(angle0)
    theta0 = wrap_to_pi(angle0 + np.pi / 2)  # Perpendicular to the centerline
    v0 = 0.0

    # Check if the initial position collide with the obstacle
    obs = all_obs_xy_list[random_idx]
    assert np.linalg.norm(np.array([x0, y0]) - obs) - (R_OBS + R_ROBOT) >= 0

    return np.array([x0, y0, theta0, v0], dtype=float)


# ╔══════════════════════════════════════════════════════════════════╗
#  Main simulation loop
# ╚══════════════════════════════════════════════════════════════════╝
def simulate(
    param: None,
    P2_CONST=P2_CONST,
    CBF_METHOD=CBF_METHOD,
    T_MAX=T_MAX,
    is_terminate_on_collision=False,
    CLASS_K=CLASS_K,
    IS_SLACK_CBF=IS_SLACK_CBF,
) -> None:
    hist = CorridorTrajectoryHistory()

    # Set random seend for reproducibility
    random.seed(0)
    np.random.seed(0)

    coll_bound = 0
    coll_obs = 0
    qp_infeas = 0

    # initial state
    state = random_init_state()

    if CBF_METHOD == CBFMethod.PACBF:
        p1_cur = P1_INIT * np.ones(NUM_CBF_CONS)  # p1 is part of the augmented state

    if CBF_METHOD == CBFMethod.RACBF:
        # r and r2 are the auxiliary relaxation states for every CBF constraint
        r_cur = R_INIT * np.ones(NUM_CBF_CONS)  # r_i
        r2_cur = np.zeros(NUM_CBF_CONS)  # ṙ_i

    d2h_list = [
        [] for _ in range(NUM_CBF_CONS)
    ]  # Store the expression of the second time derivative of h for each constraint
    d2h_previous = [0.0 for _ in range(NUM_CBF_CONS)]
    min_d_indices_previous = -np.arange(1, NUM_OBS_CONSIDER + 1, 1, dtype=int)

    N_STEPS: int = int(T_MAX / DT)
    for k in range(N_STEPS):
        print(f"\rt = {k * DT:.2f} s", end="", flush=True)
        t_now = k * DT
        x, y, theta, v = state
        pos_xy = np.array([x, y])

        # Determine the current tracking point
        # Agent will track a dynamic point. This dynamic point move along the centerline of corridor. The angle of this point is always the angle of the robot plus an offset (counterclockwise)
        angle_offset = np.pi / 36  # 5 degrees
        angle_robot = np.arctan2(y, x)
        angle_target = wrap_to_pi(angle_robot + angle_offset)
        target_xy = np.array(
            [
                CENTER[0] + R_CETERLINE * np.cos(angle_target),
                CENTER[1] + R_CETERLINE * np.sin(angle_target),
            ]
        )

        # Build QP
        u = cp.Variable(2, name="u")  # [u₁, u₂] -> [ω, a]
        # Two CLF ralaxation variables
        delta_theta = cp.Variable(1, name="delta_theta")
        delta_v = cp.Variable(1, name="delta_v")
        delta_cbf = (
            cp.Variable(NUM_CBF_CONS, name="delta_cbf") if IS_SLACK_CBF else None
        )

        # CLF constraint
        constraints = []
        constraints.append(delta_theta >= 0.0)
        constraints.append(delta_v >= 0.0)

        # CLF constraints for both angular velocity and linear acceleration
        desired_theta = np.arctan2(target_xy[1] - y, target_xy[0] - x)
        theta_error = desired_theta - theta
        theta_error_nom = np.arctan2(np.sin(theta_error), np.cos(theta_error))
        np.arctan2(np.sin(theta_error), np.cos(theta_error))
        v_error = V_REF - v

        constraints.append(
            compute_clf_condition(u[0], delta_theta, C_CLF, theta_error_nom)
        )
        constraints.append(compute_clf_condition(u[1], delta_v, C_CLF, v_error))

        u_nominal = nominal_robot_controller(theta_error_nom, v_error)

        # Objective (penalise both slacks)
        obj = (
            R_NOM[0] * cp.square(u[0] - u_nominal[0])
            + R_NOM[1] * cp.square(u[1] - u_nominal[1])
            + W_SLACK_CLF * cp.square(delta_theta)
            + W_SLACK_CLF * cp.square(delta_v)
        )
        if IS_SLACK_CBF:
            obj += W_SLACK_CBF * cp.sum_squares(delta_cbf)
            for i in range(NUM_CBF_CONS):
                constraints.append(
                    delta_cbf >= 0.0
                )  # Non-negativity of slack variables

        if CBF_METHOD == CBFMethod.aTTCBF:
            lambda_ttcbf_var = cp.Variable(NUM_CBF_CONS, name="lambda_ttcbf")
            obj += W_LAMBDA * cp.sum_squares(lambda_ttcbf_var)
            for i in range(NUM_CBF_CONS):
                constraints.append(lambda_ttcbf_var[i] >= 0.0)
                constraints.append(lambda_ttcbf_var[i] <= 1.0)

        if CBF_METHOD == CBFMethod.PACBF:
            nu_pacbf_var = cp.Variable(NUM_CBF_CONS, name="nu")  # For PACBF
            delta_p1 = cp.Variable(NUM_CBF_CONS, name="delta_p1")  # CLF
            p2_var = cp.Variable(NUM_CBF_CONS, name="p2")
            obj += (
                W_NU_PACBF * cp.sum_squares(nu_pacbf_var)
                + W_SLACK_CLF_P1 * cp.sum_squares(delta_p1)
                + W_SLACK_CLF_P2 * cp.sum_squares(p2_var)
            )
            V_p1 = (p1_cur - P1_DESIRED) ** 2
            for i in range(NUM_CBF_CONS):
                constraints.append(
                    nu_pacbf_var[i] + LAMBDA_P1 * p1_cur[i] >= 0
                )  # Construct a CBF to ensure p1 is non-negative
                constraints.append(
                    2 * (p1_cur[i] - P1_DESIRED) * nu_pacbf_var[i] + LAMBDA_P1 * V_p1[i]
                    <= delta_p1[i]
                )  # CLF
                constraints.append(
                    delta_p1[i] >= 0.0
                )  # Non-negativity of CLF relaxation variables
                constraints.append(
                    p2_var[i] >= 0.0
                )  # Non-negativity of p2 relaxation variables
        if CBF_METHOD == CBFMethod.RACBF:
            # ν_i –– auxiliary input driving r̈_i  (appears in ψ̈0)
            nu_racbf_var = cp.Variable(NUM_CBF_CONS, name="nu_racbf")
            # δ_r  –– CLF relaxation for driving r→R_TARGET
            delta_r = cp.Variable(NUM_CBF_CONS, name="delta_r")
            obj += W_NU_RACBF * cp.sum_squares(
                nu_racbf_var
            ) + W_SLACK_CLF_R * cp.sum_squares(delta_r)
            # enforce non-negativity slack

            for i in range(NUM_CBF_CONS):
                # --- HOCBF on r to guarantee r ≥ 0  ──────────
                constraints.append(delta_r[i] >= 0.0)
                constraints.append(
                    nu_racbf_var[i]
                    + (LAMBDA_R_1 + LAMBDA_R_2) * r2_cur[i]
                    + LAMBDA_R_1 * LAMBDA_R_2 * r_cur[i]
                    >= 0
                )  # CBF to ensure r is non-negative

                # --- CLF to drive r  ➜  R_TARGET  ────────────  (eq. 36 with V = (r2+k1(r-R*))² )
                V_r = (r2_cur[i] + K1_RACBF * (r_cur[i] - R_TARGET)) ** 2
                clf_r = (
                    2
                    * (r2_cur[i] + K1_RACBF * (r_cur[i] - R_TARGET))
                    * (nu_racbf_var[i] + K1_RACBF * r2_cur[i])
                    + C_CLF_PACBF * V_r
                )
                constraints.append(clf_r <= delta_r[i])

        prob_obj = cp.Minimize(obj)

        # Corridor-boundary CBFs and obstacle CBFs
        d_to_obs = np.linalg.norm(all_obs_xy_list - pos_xy, axis=1)
        min_d_indices = (
            argmink(d_to_obs, k=NUM_OBS_CONSIDER)
            if NUM_OBS_CONSIDER < len(all_obs_xy_list)
            else np.arange(len(all_obs_xy_list))
        )

        for i in range(NUM_CBF_CONS):
            if i == 0:
                # Corridor inner boundary
                h, dh, d2h, d2h_components = distance_cbf_terms(
                    state, u, CENTER, (R_INNER + R_ROBOT) ** 2
                )
            elif i == 1:
                # Corridor outer boundary
                h, dh, d2h, d2h_components = distance_cbf_terms(
                    state, u, CENTER, (R_OUTER - R_ROBOT) ** 2
                )
                h, dh, d2h = -h, -dh, -d2h
            else:
                # Obstacles
                obs_xy = all_obs_xy_list[min_d_indices[i - 2]]

                h, dh, d2h, d2h_components = distance_cbf_terms(
                    state, u, obs_xy, (R_ROBOT + R_OBS) ** 2
                )

            d2h_list[i] = d2h

            if CBF_METHOD == CBFMethod.HOCBF:
                cbf = d2h + (LAMBDA_1 + LAMBDA_2) * dh + LAMBDA_1 * LAMBDA_2 * h
            elif CBF_METHOD == CBFMethod.PACBF:
                cbf = (
                    d2h
                    + (p1_cur[i] + p2_var[i]) * dh
                    + (p1_cur[i] * p2_var[i] + nu_pacbf_var[i]) * h
                )
            elif CBF_METHOD == CBFMethod.RACBF:
                # CBF to ensure h-r is non-negative
                psi0 = h - r_cur[i]

                # ψ₁  =  ψ̇₀ + k₁ ψ₀
                psi1 = (dh - r2_cur[i]) + K1_RACBF * psi0

                # ψ̈₀  =  d2h − ν_i                (because r̈ = ν)
                psi0_ddot = d2h - nu_racbf_var[i]

                # ψ₂  =  ψ̈₀ + k₂ ψ₁ + 2k₁ ψ₀ ψ̇₀
                # cbf = psi0_ddot + K2_RACBF * psi1 + 2 * K1_RACBF * psi0 * (dh - r2_cur[i])
                cbf = psi0_ddot + K2_RACBF * psi1 + K1_RACBF * (dh - r2_cur[i])
            elif CBF_METHOD in [CBFMethod.TTCBF, CBFMethod.aTTCBF]:
                if CBF_METHOD == CBFMethod.TTCBF:
                    if CLASS_K == ClassKType.LINEAR:
                        psi_class_k_term = param * h
                    elif CLASS_K == ClassKType.EXP:
                        # \alpha(h) = a * h^b
                        a = param  # Deafult 0.2
                        b = 1.1  # [0.85, 1.3]
                        psi_class_k_term = a * h**b
                    elif CLASS_K == ClassKType.RATIONAL:
                        # \alpha(h) = a*h/(1+b*h)
                        # a/b <= 1 (to make sure \alpha(s) <= s)
                        a = param  # Default 0.2
                        b = 1
                        psi_class_k_term = a * h**2 / (1.0 + b * h)
                    else:
                        raise NotImplementedError(f"Unknown class K type: {CLASS_K}")
                elif CBF_METHOD == CBFMethod.aTTCBF:
                    if CLASS_K == ClassKType.LINEAR:
                        psi_class_k_term = lambda_ttcbf_var[i] * h
                    elif CLASS_K == ClassKType.EXP:
                        # \alpha(h) = a * h^b
                        a = lambda_ttcbf_var[i]
                        b = 1.1  # Default 0.9 [0.85, 1.35]
                        psi_class_k_term = a * h**b
                    elif CLASS_K == ClassKType.RATIONAL:
                        # \alpha(h) = a*h^2/(1+b*h)
                        # a/b <= 1 (to make sure \alpha(s) <= s)
                        a = lambda_ttcbf_var[i]
                        b = 1
                        psi_class_k_term = a * h**2 / (1.0 + b * h)
                    else:
                        raise NotImplementedError(f"Unknown class K type: {CLASS_K}")
                else:
                    raise NotImplementedError(f"Unknown CBF method: {CBF_METHOD}")

                    # psi_class_k_term = lambda_ttcbf_var[i] * h

                cbf = d2h * (DT_T**2) / 2.0 + dh * DT_T + psi_class_k_term

                # Cosider Taylor truncated error
                d2h_min = (
                    d2h_components[0]
                    + d2h_components[1]
                    * (-U1_MAX if d2h_components[1] >= 0 else U1_MAX)
                    + d2h_components[2]
                    * (-U2_MAX if d2h_components[2] >= 0 else U2_MAX)
                )
                d3h_min = (d2h_min - d2h_previous[i]) / DT_T
                R = d3h_min * DT_T**3 / 6
                cbf += R
            else:
                raise NotImplementedError(f"Unknown CBF method: {CBF_METHOD}")

            constraints += [cbf >= (-delta_cbf[i] if IS_SLACK_CBF else 0.0)]

            # Check for initial conditions for HOCBF
            if CBF_METHOD == CBFMethod.HOCBF and k == 0:
                psi_0 = h
                psi_1 = dh + p1_cur[i] * h
                assert psi_0 >= 0
                assert psi_1 >= 0

        # Hard control bounds
        constraints += [
            u[0] <= +U1_MAX,
            u[0] >= -U1_MAX,
            u[1] <= +U2_MAX,
            u[1] >= -U2_MAX,
        ]

        # Solve QP
        prob = cp.Problem(prob_obj, constraints)

        is_continue = False
        try:
            prob.solve(
                solver=cp.OSQP, warm_start=True, verbose=False, max_iter=20000
            )  # Default iterations of OSQP is 10000
        except cp.SolverError:
            print("[WARNING] QP error")
            is_continue = True

        hist.qp_solving_t.append(prob._solve_time)
        hist.qp_solving_iter.append(prob.solver_stats.num_iters)

        if prob.status not in {"optimal", "optimal_inaccurate"}:
            print("[WARNING] QP not optimal")
            is_continue = True

        if is_continue:
            # print(f"Objective: {prob_obj}")
            # for i_const in constraints:
            #     print(f"i_const: {i_const}")
            qp_infeas += 1
            hist.evt.append("QP-INF")
            # keep time-history clean
            hist.t.append(t_now)
            hist.xy.append(state[:2].copy())
            hist.state.append(state.copy())
            hist.v.append(state[3])
            state = random_init_state()
            hist.u1.append(0.0)
            hist.u2.append(0.0)
            hist.lambda_ttcbf.append(np.zeros(NUM_CBF_CONS, dtype=float))
            hist.nu_pacbf.append(np.zeros(NUM_CBF_CONS, dtype=float))
            hist.p1_pacbf.append(np.zeros(NUM_CBF_CONS, dtype=float))
            hist.p2_pacbf.append(np.zeros(NUM_CBF_CONS, dtype=float))
            hist.qp_solving_t.append(0.0)
            hist.qp_solving_iter.append(0)

            d2h_previous = [0.0 for _ in range(NUM_CBF_CONS)]
            continue

        # Logs
        if IS_SLACK_CBF:
            delta_cbf_values = np.array([i.value for i in delta_cbf])
            # if any(delta_cbf_values >= 1e-3):
            #     print(f" Slack variable too high (max delta_cbf = {np.max(delta_cbf.value)})!!!")
        # optimal control
        u_star = u.value
        u1_star, u2_star = float(u_star[0]), float(u_star[1])

        # Integrate one step (Euler)
        x_dot, y_dot, theta_dot, v_dot = unicycle_dynamics(state, u_star)
        state_next = state + DT * np.array([x_dot, y_dot, theta_dot, v_dot])
        state_next[2] = wrap_to_pi(state_next[2])
        # Euler step for the auxiliary dynamics  ṗ₁ = ν₁
        if CBF_METHOD == CBFMethod.PACBF:
            p1_cur += DT * nu_pacbf_var.value
            p1_cur = np.clip(p1_cur, 0, P1_MAX)  # Ensure p1 is non-negative
        if CBF_METHOD == CBFMethod.RACBF:
            r_cur += DT * r2_cur
            r2_cur += DT * nu_racbf_var.value
            # safety clip
            r_cur = np.maximum(r_cur, 0.0)

        # Collision check
        # road boundaries
        d_sq = (state_next[0] - CENTER[0]) ** 2 + (state_next[1] - CENTER[1]) ** 2
        h_inner = d_sq - (R_INNER + R_ROBOT) ** 2
        h_outer = (R_OUTER - R_ROBOT) ** 2 - d_sq

        collided = False
        if h_inner < -1e-3 or h_outer < -1e-3:
            coll_bound += 1
            collided = True
            hist.evt.append("COLL-B")
            print("[WARNING] Collision with boundary!")
        else:
            # obstacle collisions
            min_d_obs = np.min(d_to_obs)
            if min_d_obs <= R_ROBOT + R_OBS - 1e-3:
                coll_obs += 1
                collided = True
                hist.evt.append("COLL-O")
                print("[WARNING] Collision with obstacle!")

        # ⑩ record history (before potential reset)
        hist.t.append(t_now)
        hist.xy.append(state_next[:2].copy())
        hist.state.append(state_next.copy())
        hist.v.append(state_next[3])
        hist.u1.append(u1_star)
        hist.u2.append(u2_star)
        d2h_previous = [d2h_list[i].value for i in range(len(d2h_list))]
        if CBF_METHOD == CBFMethod.aTTCBF:
            hist.lambda_ttcbf.append(lambda_ttcbf_var.value)
        if CBF_METHOD == CBFMethod.PACBF:
            hist.nu_pacbf.append(nu_pacbf_var.value)
            hist.p1_pacbf.append(p1_cur)
            hist.p2_pacbf.append(p2_var.value)
        if CBF_METHOD == CBFMethod.RACBF:
            hist.r_racbf.append(r_cur)
            hist.nu_racbf.append(nu_racbf_var.value)
        if not collided:
            hist.evt.append("")

        # ⑪ reset if collision
        if collided:
            if is_terminate_on_collision:
                break
            else:
                state = random_init_state()
        else:
            state = state_next.copy()

    # ╔══════════════════════════════════════════════════════════════╗
    #  Summary statistics
    # ╚══════════════════════════════════════════════════════════════╝
    print("\n────────────────────── Simulation Summary ──────────────────────")
    print(f"{CBF_METHOD.name} with {CLASS_K.name} class K with {param}")
    # Print the absolute average u1 and u2
    print(
        f"Average u1 and u2: {np.mean(np.abs(hist.u1)):.2f} rad/s and {np.mean(np.abs(hist.u2)):.2f} m/s^2"
    )
    print(f"Road-boundary collisions : {coll_bound}")
    print(f"Obstacle collisions      : {coll_obs}")
    print(f"QP infeasibilities       : {qp_infeas}")
    print(f"Acerage speed            : {np.mean(hist.v):.2f} m/s")
    num_decision_variables = sum(var.size for var in prob.variables())
    num_constraints = len(prob.constraints)
    print(
        f"Number of decision variables and constraints: {num_decision_variables} and {num_constraints}."
    )
    print(f"QP solving time per step : {1000*np.mean(hist.qp_solving_t):.2f} ms")
    print(f"QP solving iterations per step: {np.mean(hist.qp_solving_iter):.1f} iters")

    print("────────────────────────────────────────────────────────────")

    return hist

def plot_multi_pos_trajectories(
    CLASS_K: ClassKType,
    hist_list: list[CorridorTrajectoryHistory],
    fig_name_prefix,
    parameter_list=None,
):
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))

    # cm = get_colormap(len(hist_list), style="tab10")
    cm = [
        "darkgoldenrod",
        "darkviolet",
        "darkred",
        "#008B8B",
    ]
    # ls = ["-", "--", "-.", ":"]
    ls = ["-" for _ in range(len(hist_list))]

    lw = 2.0

    safety_boundary_c = (1, 0, 0, 0.3)

    # Plot outer boundary, inner boundary, and centerline as circles
    circle_outer = plt.Circle(CENTER, R_OUTER, color="k", fill=False, lw=lw)
    circle_inner = plt.Circle(CENTER, R_INNER, color="k", fill=False, lw=lw)
    circle_centerline = plt.Circle(
        CENTER, R_CETERLINE, color="k", linestyle="-.", fill=False, lw=lw / 2, alpha=0.4
    )
    axs.add_patch(circle_outer)
    axs.add_patch(circle_inner)
    axs.add_patch(circle_centerline)
    # Safety boundary
    circle = plt.Circle(
        CENTER,
        R_OUTER - R_ROBOT,
        edgecolor=safety_boundary_c,
        fill=False,
        lw=lw / 2,
        linestyle="--",
    )
    axs.add_patch(circle)
    circle = plt.Circle(
        CENTER,
        R_INNER + R_ROBOT,
        edgecolor=safety_boundary_c,
        fill=False,
        lw=lw / 2,
        linestyle="--",
    )
    axs.add_patch(circle)

    # Show all obs_xy_list as circles with radius R_OBS
    for obs_idx, obs_i in enumerate(all_obs_xy_list):
        circle = plt.Circle(
            obs_i, R_OBS, facecolor="gray", edgecolor="black", fill=True, lw=lw
        )
        axs.add_patch(circle)
        # Safety boundary
        circle = plt.Circle(
            obs_i,
            R_OBS + R_ROBOT,
            edgecolor=safety_boundary_c,
            fill=False,
            lw=lw / 2,
            linestyle="--",
        )
        axs.add_patch(circle)
        # Add obstacle IDs
        axs.text(
            obs_i[0],
            obs_i[1] - 0.3,
            f"{obs_idx}",
            fontsize=13,
            ha="center",
            va="center",
            color="k",
            clip_on=True,
        )

    for i_hist, hist in enumerate(hist_list):
        hist_state_arr = np.vstack(hist.state)
        if parameter_list[i_hist] is not None:
            label_i = rf"$a = {parameter_list[i_hist]}$"
        else:
            label_i = "aTTCBF"
        center_tracking_error = np.abs(
            np.linalg.norm(hist_state_arr[:, 0:2] - CENTER, axis=1) - R_CETERLINE
        )
        # print(
        #     f"[INFO] Path-tracking error of {label_i}: {np.mean(center_tracking_error):.2f} m"
        # )

        axs.plot(
            hist_state_arr[:, 0],
            hist_state_arr[:, 1],
            color=cm[i_hist],
            lw=lw,
            ls=ls[i_hist],
            label=label_i,
        )

        # Initial position
        if i_hist == 0:
            init_circle = plt.Circle(
                (hist_state_arr[0, 0], hist_state_arr[0, 1]),
                R_ROBOT,
                color="k",
                linestyle="--",
                fill=False,
                lw=lw,
            )

            axs.arrow(
                hist_state_arr[0, 0],
                hist_state_arr[0, 1],
                4 * np.cos(hist_state_arr[0, 2]),
                4 * np.sin(hist_state_arr[0, 2]),
                head_width=0.8,
                head_length=1.2,
                fc="k",
                ec="k",
                lw=1.0,
                zorder=6,
            )

            axs.add_patch(init_circle)

        # highlight the actual collision points
        # Boolean masks for collisions (for red dots)
        coll_mask = np.array([c.startswith("COLL") for c in hist.evt])
        if np.any(coll_mask):
            coll_xy_arr = hist_state_arr[coll_mask, 0:2]

            for i in range(len(coll_xy_arr)):
                circle = plt.Circle(
                    (coll_xy_arr[i, 0], coll_xy_arr[i, 1]),
                    R_ROBOT,
                    color=cm[i_hist],
                    fill=False,
                    lw=lw,
                    linestyle="--",
                )
                axs.add_patch(circle)

    first_legend = axs.legend(loc="lower right", frameon=True, framealpha=1.0)
    # Create a proxy artist
    safety_line = mlines.Line2D(
        [],
        [],
        color=safety_boundary_c,
        linestyle="--",
        lw=lw / 2,
        label="Safety boundary",
    )
    # Create a second legend
    axs.legend(handles=[safety_line], loc="upper left", frameon=True, framealpha=1.0)
    # Re-add the first legend
    axs.add_artist(first_legend)

    axs.set_xlim(-5, 50)
    axs.set_ylim(-45, 0)
    axs.set_xticks(np.arange(0, 51, 10))
    axs.set_yticks(np.arange(-45, 1, 5))
    axs.grid(True, linestyle=":", alpha=0.4)

    axs.set_xlabel(r"$x$ [m]")
    axs.set_ylabel(r"$y$ [m]")
    axs.set_aspect("equal", adjustable="box")
    axs.tick_params(axis="both", direction="in")

    plt.tight_layout()

    fig_name = fig_name_prefix.parent / (
        fig_name_prefix.name + "_trajectories.pdf"
    )

    plt.savefig(fig_name, bbox_inches="tight", pad_inches=0)
    cprint(f"[INFO] Figure saved to {fig_name}")

    # ╔══════════════════════════════════════════════════════════════╗
    #  Plot u1, u2, and v separately
    # ╚══════════════════════════════════════════════════════════════╝
    lw = 2.0
    ls = [":", "--", "-.", "-"]
    labels = [rf"$a={p}$" for p in parameter_list[:-1]] + [r"aTTCBF"]
    cm = [
        "darkgoldenrod",
        "darkviolet",
        "darkred",
        "#008B8B",
    ]

    # ╔══════════════════════════════════════════════════════════════╗
    #  Plot u1
    # ╚══════════════════════════════════════════════════════════════╝
    fig, axs = plt.subplots(1, 1, figsize=(6, 2.5))
    plt.axhline(0, color="k", linestyle=":", linewidth=lw / 4)
    for i_hist, hist in enumerate(hist_list):
        hist_state_arr = np.vstack(hist.state)
        t_arr = np.array(hist.t)
        axs.plot(
            t_arr, hist.u1, color=cm[i_hist], lw=lw, ls=ls[i_hist], label=labels[i_hist]
        )

    t_max = t_arr[-1]
    axs.fill_between(
        x=[0, t_max],  # full x-axis range
        y1=U1_MAX,
        y2=U1_MAX + 1.0,
        color="tab:red",
        alpha=0.2,
        # label="Unsafe Region"
    )
    axs.fill_between(
        x=[0, t_max],  # full x-axis range
        y1=-U1_MAX - 1.0,
        y2=-U1_MAX,
        color="tab:red",
        alpha=0.2,
        # label="Unsafe Region"
    )
    plt.axhline(U1_MAX, color="tab:red", linestyle="--", linewidth=lw / 2)
    plt.axhline(-U1_MAX, color="tab:red", linestyle="--", linewidth=lw / 2)

    axs.set_xlabel(r"Time $t$ [s]")
    axs.set_ylabel(r"Steering rate $u_1$ [rad/s]")
    axs.set_xlim(0, t_max)
    axs.set_ylim(-U1_MAX - 0.5, U1_MAX + 0.5)
    
    if CLASS_K == ClassKType.LINEAR:
        xy_target_list = [(2.8, 2), (6.5, 2)]
        text_offset = (0.2, -0.8)
    elif CLASS_K == ClassKType.EXP:
        xy_target_list = [(2.9, 2), (6.7, 2)]
        text_offset = (0.2, -0.8)
    elif CLASS_K == ClassKType.RATIONAL:
        xy_target_list = [(2.8, 2), (6.55, 2)]
        text_offset = (0.2, -0.8)
    else:
        raise NotImplementedError(f"Unknown class K type: {CLASS_K}")

    for i, xy_target in enumerate(xy_target_list):
        axs.annotate(
            "Saturation",
            xy=(xy_target[0], xy_target[1]),
            xytext=(xy_target[0] + text_offset[0], xy_target[1] + text_offset[1]),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            fontsize=10,
            color="black",
            ha="left",
            va="top",
        )

    # First legend (methods)
    legend1 = axs.legend(loc="upper left", frameon=True, framealpha=1.0)

    # Create dummy handles for the second legend (safety bounds and unsafe region)

    bound_line = Line2D(
        [0],
        [0],
        color="tab:red",
        linestyle="--",
        linewidth=lw / 2,
        label=r"$u_{1,\mathrm{max}/\mathrm{min}}$",
    )

    # Add second legend
    legend2 = axs.legend(
        handles=[bound_line], loc="lower left", frameon=True, framealpha=1.0
    )
    axs.add_artist(legend1)  # Re-add the first legend so it does not get overwritten

    axs.grid(True, linestyle=":", alpha=0.4)
    axs.tick_params(axis="both", direction="in")

    plt.tight_layout()

    fig_name = fig_name_prefix.parent / (
        fig_name_prefix.name + "_u1.pdf"
    )

    plt.savefig(fig_name, bbox_inches="tight", pad_inches=0)
    cprint(f"[INFO] Figure saved to {fig_name}")

    # ╔══════════════════════════════════════════════════════════════╗
    #  Plot u2
    # ╚══════════════════════════════════════════════════════════════╝
    fig, axs = plt.subplots(1, 1, figsize=(6, 2.5))
    plt.axhline(0, color="k", linestyle=":", linewidth=lw / 4)
    for i_hist, hist in enumerate(hist_list):
        hist_state_arr = np.vstack(hist.state)
        t_arr = np.array(hist.t)
        axs.plot(
            t_arr, hist.u2, color=cm[i_hist], lw=lw, ls=ls[i_hist], label=labels[i_hist]
        )

    t_max = t_arr[-1]
    axs.fill_between(
        x=[0, t_max],  # full x-axis range
        y1=U2_MAX,
        y2=U2_MAX + 1.0,
        color="tab:red",
        alpha=0.2,
        # label="Unsafe Region"
    )
    axs.fill_between(
        x=[0, t_max],  # full x-axis range
        y1=-U2_MAX - 1.0,
        y2=-U2_MAX,
        color="tab:red",
        alpha=0.2,
        # label="Unsafe Region"
    )
    plt.axhline(U2_MAX, color="tab:red", linestyle="--", linewidth=lw / 2)
    plt.axhline(-U2_MAX, color="tab:red", linestyle="--", linewidth=lw / 2)

    axs.set_xlabel(r"Time $t$ [s]")
    axs.set_ylabel(r"Acceleration $u_2$ [m/s$^2$]")
    axs.set_xlim(0, t_max)
    axs.set_ylim(-U2_MAX - 0.5, U2_MAX + 0.5)
    
    if CLASS_K == ClassKType.LINEAR:
        xy_target = (6.5, -2)
        text_offset = (0.15, 1.1)
    elif CLASS_K == ClassKType.EXP:
        xy_target = (6.7, -2)
        text_offset = (0.15, 1.1)
    elif CLASS_K == ClassKType.RATIONAL:
        xy_target = (6.55, -2)
        text_offset = (0.15, 1.1)
    else:
        raise NotImplementedError(f"Unknown class K type: {CLASS_K}")

    axs.annotate(
        "Saturation",
        xy=(xy_target[0], xy_target[1]),
        xytext=(xy_target[0] + text_offset[0], xy_target[1] + text_offset[1]),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        fontsize=10,
        color="black",
        ha="left",
        va="top",
    )


    # First legend (methods)
    legend1 = axs.legend(loc="upper left", frameon=True, framealpha=1.0)

    # Create dummy handles for the second legend (safety bounds and unsafe region)

    bound_line = Line2D(
        [0],
        [0],
        color="tab:red",
        linestyle="--",
        linewidth=lw / 2,
        label=r"$u_{2,\mathrm{max}/\mathrm{min}}$",
    )

    # Add second legend
    legend2 = axs.legend(
        handles=[bound_line], loc="lower left", frameon=True, framealpha=1.0
    )
    axs.add_artist(legend1)  # Re-add the first legend so it does not get overwritten

    axs.grid(True, linestyle=":", alpha=0.4)
    axs.tick_params(axis="both", direction="in")

    plt.tight_layout()

    fig_name = fig_name_prefix.parent / (
        fig_name_prefix.name + "_u2.pdf"
    )

    plt.savefig(fig_name, bbox_inches="tight", pad_inches=0)
    cprint(f"[INFO] Figure saved to {fig_name}")

    # ╔══════════════════════════════════════════════════════════════╗
    #  Plot v
    # ╚══════════════════════════════════════════════════════════════╝
    fig, axs = plt.subplots(1, 1, figsize=(6, 2.5))
    plt.axhline(V_REF, color="tab:green", linestyle="--", linewidth=lw / 2)
    for i_hist, hist in enumerate(hist_list):
        hist_state_arr = np.vstack(hist.state)
        t_arr = np.array(hist.t)
        axs.plot(
            t_arr, hist.v, color=cm[i_hist], lw=lw, ls=ls[i_hist], label=labels[i_hist]
        )

    t_max = t_arr[-1]
    axs.fill_between(
        x=[0, t_max],  # full x-axis range
        y1=V_REF - 0.5,
        y2=V_REF + 0.5,
        color="tab:green",
        alpha=0.2,
    )

    axs.set_xlabel(r"Time $t$ [s]")
    axs.set_ylabel(r"Speed $v$ [m/s]")
    axs.set_xlim(0, t_max)
    axs.set_ylim(-1, V_REF + 2)

    # Use the first trajectory as reference for y-position
    t_ref = t_arr
    v_ref = hist_list[0].v

    # -------------------------
    # Annotations: abrupt deviation
    # -------------------------
    if CLASS_K == ClassKType.LINEAR:
        t_markers = [2.7, 6.5]
    elif CLASS_K == ClassKType.EXP:
        t_markers = [2.78, 6.7]
    elif CLASS_K == ClassKType.RATIONAL:
        t_markers = [2.68, 6.5]
    else:
        raise NotImplementedError(f"Unknown class K type: {CLASS_K}")
    
    for i, t_m in enumerate(t_markers):
        # Find closest index to the target time
        idx = int(np.argmin(np.abs(t_ref - t_m)))
        x_target = t_ref[idx]
        y_target = v_ref[idx]

        # Offset the text to avoid overlap with curves
        # Slightly different vertical offsets for readability
        if CLASS_K == ClassKType.LINEAR:
            xy_offset = (0, -0.2)
            text_offset = (0, -2.5)
        elif CLASS_K == ClassKType.EXP:
            xy_offset = (0, -0.35)
            text_offset = (0, -2.5)
        elif CLASS_K == ClassKType.RATIONAL:
            xy_offset = (0, -0.32) if i == 0 else (0, 0.1)
            text_offset = (0, -2.5)
        else:
            raise NotImplementedError(f"Unknown class K type: {CLASS_K}")
        
        axs.annotate(
            "Abrupt deviation",
            xy=(x_target + xy_offset[0], y_target + xy_offset[1]),
            xytext=(x_target + text_offset[0], y_target + text_offset[1]),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            fontsize=10,
            color="black",
            ha="center",
            va="center",
        )

    # First legend (methods)
    legend1 = axs.legend(loc="lower right", frameon=True, framealpha=1.0)

    # Create dummy handles for the second legend (safety bounds and unsafe region)

    bound_line = Line2D(
        [0],
        [0],
        color="tab:green",
        linestyle="--",
        linewidth=lw / 2,
        label=r"$v_{\mathrm{des}}$",
    )

    # Add second legend
    legend2 = axs.legend(
        handles=[bound_line], loc="upper left", frameon=True, framealpha=1.0
    )
    axs.add_artist(legend1)  # Re-add the first legend so it does not get overwritten

    axs.grid(True, linestyle=":", alpha=0.4)
    axs.tick_params(axis="both", direction="in")

    plt.tight_layout()

    fig_name = fig_name_prefix.parent / (
        fig_name_prefix.name + "_v.pdf"
    )

    plt.savefig(fig_name, bbox_inches="tight", pad_inches=0)
    cprint(f"[INFO] Figure saved to {fig_name}")

    # ╔══════════════════════════════════════════════════════════════╗
    #  Plot normalized control effort (stacked) and average speed
    # ╚══════════════════════════════════════════════════════════════╝
    fig2, ax_l = plt.subplots(figsize=(6, 3))  # primary (left) axis
    ax_r = ax_l.twinx()  # secondary (right) axis

    # x-axis: parameter value if provided, else case index
    if parameter_list is not None and all(p is not None for p in parameter_list):
        x_vals = np.asarray(parameter_list, dtype=float)
        ax_l.set_xlabel(r"$a$")
    else:
        x_vals = np.arange(len(hist_list))
        ax_l.set_xlabel("Case Index")

    # Compute averages
    avg_u1 = np.array([np.mean(np.abs(h.u1)) for h in hist_list])
    avg_u2 = np.array([np.mean(np.abs(h.u2)) for h in hist_list])
    avg_v = np.array([np.mean(h.v) for h in hist_list])

    # Normalize control effort
    norm_u1 = avg_u1 / U1_MAX * 100.0  # Convert to percentage
    norm_u2 = avg_u2 / U2_MAX * 100.0  # Convert to percentage

    # Plot stacked bars on primary axis
    bar1 = ax_l.bar(
        x_vals, norm_u1, color="tab:blue", label=r"Normalized $\overline{|\hat{u_1}|}$"
    )
    bar2 = ax_l.bar(
        x_vals,
        norm_u2,
        bottom=norm_u1,
        color="tab:green",
        label=r"Normalized $\overline{|\hat{u_2}|}$",
    )

    # Add text labels
    for i, (u1, u2) in enumerate(zip(norm_u1, norm_u2)):
        ax_l.text(
            i,
            u1 / 2,
            rf"{u1:.1f}$\%$",
            ha="center",
            va="center",
            fontsize=11,
            color="black",
        )
        ax_l.text(
            i,
            u1 + u2 / 2,
            rf"{u2:.1f}$\%$",
            ha="center",
            va="center",
            fontsize=11,
            color="black",
        )
        # ax_l.text(i, u1 + u2 + 0.5, rf"{u1 + u2:.1f}$\%$",
        #           ha='center', va='bottom', fontsize=11, color='black')
    # Also add text for the average speed

    # Add a horizontal line for u1+u2
    ax_l.axhline(
        y=norm_u1[-1] + norm_u2[-1],
        color="gray",
        linestyle="--",
        lw=1.5,
        label=r"Average $\overline{|\hat{u_1}|} + \overline{|\hat{u_2}|}$",
    )

    ax_l.set_ylabel(r"Normalized control effort [$\%$]")
    ax_l.grid(True, linestyle=":", alpha=0.4)
    ax_l.tick_params(axis="y", direction="in")
    ax_l.set_ylim(0, 200.0)

    # Plot average speed on secondary axis
    ln_v = ax_r.plot(
        x_vals,
        avg_v,
        marker="o",
        ls="-",
        lw=2,
        color="tab:red",
        label=r"Average speed $\overline{v}$",
    )

    for i, v in enumerate(avg_v):
        ax_r.text(
            i,
            v - 0.05,
            f"{v:.2f} m/s",
            ha="center",
            va="top",
            fontsize=11,
            color="tab:red",
        )

    ax_r.set_ylabel(r"Average speed $\overline{v}$ [m/s]")
    ax_r.tick_params(axis="y", direction="in", colors="tab:red")
    ax_r.spines["right"].set_color("tab:red")
    ax_r.set_ylim(5.0, 8.0)

    if parameter_list is not None and len(parameter_list) == len(hist_list):
        xtick_labels = [rf"$a={p}$" for p in parameter_list[:-1]] + [r"aTTCBF"]
        x_vals = np.arange(len(hist_list))  # place bars at 0,1,2,…
        ax_l.set_xticks(x_vals)
        ax_l.set_xticklabels(xtick_labels, rotation=0)
        ax_l.set_xlabel("")

    # Combine legend entries
    lines = [bar1, bar2] + ln_v
    labels = [l.get_label() for l in lines]
    ax_l.legend(lines, labels, frameon=True, framealpha=1.0, loc="upper right")

    plt.tight_layout()
    fig_name = fig_name_prefix.parent / (
        fig_name_prefix.name + "_u_v.pdf"
    )
    plt.savefig(fig_name, bbox_inches="tight", pad_inches=0)
    cprint(f"[INFO] Figure saved to {fig_name}")

    # plt.show()


def plot_single_pos_trajectories(
    hist_list: List[CorridorTrajectoryHistory], cbf_list: List[CBFMethod], fig_name_prefix
):

    fig, axs = plt.subplots(1, 1, figsize=(6, 6))

    cm = [
        "#008B8B",
        "tab:purple",
        "tab:orange",
    ]
    ls = ["-", "--", "-."]

    safety_boundary_c = (1, 0, 0, 0.3)

    axs.set_xlim(-50, 50)
    axs.set_ylim(-50, 50)
    axs.set_xticks(np.arange(-40, 51, 20))
    axs.set_yticks(np.arange(-40, 51, 20))
    axs.grid(True, linestyle=":", alpha=0.4)

    axs.set_xlabel(r"$x$ [m]")
    axs.set_ylabel(r"$y$ [m]")
    axs.set_aspect("equal", adjustable="box")
    axs.tick_params(axis="both", direction="in")

    lw = 2.0

    # Plot outer boundary, inner boundary, and centerline as circles
    circle_outer = plt.Circle(CENTER, R_OUTER, color="k", fill=False, lw=lw)
    circle_inner = plt.Circle(CENTER, R_INNER, color="k", fill=False, lw=lw)
    circle_centerline = plt.Circle(
        CENTER, R_CETERLINE, color="k", linestyle="-.", fill=False, lw=lw / 2, alpha=0.4
    )
    axs.add_patch(circle_outer)
    axs.add_patch(circle_inner)
    axs.add_patch(circle_centerline)
    # Safety boundary
    circle = plt.Circle(
        CENTER,
        R_OUTER - R_ROBOT,
        edgecolor=safety_boundary_c,
        fill=False,
        lw=lw / 3,
        linestyle="--",
    )
    axs.add_patch(circle)
    circle = plt.Circle(
        CENTER,
        R_INNER + R_ROBOT,
        edgecolor=safety_boundary_c,
        fill=False,
        lw=lw / 3,
        linestyle="--",
    )
    axs.add_patch(circle)

    # Show all obs_xy_list as circles with radius R_OBS
    for obs_idx, obs_i in enumerate(all_obs_xy_list):
        circle = plt.Circle(
            obs_i, R_OBS, facecolor="gray", edgecolor="black", fill=True, lw=lw
        )
        axs.add_patch(circle)
        # Safety boundary
        circle = plt.Circle(
            obs_i,
            R_OBS + R_ROBOT,
            edgecolor=safety_boundary_c,
            fill=False,
            lw=lw / 3,
            linestyle="--",
        )
        axs.add_patch(circle)
        # Add obstacle IDs
        axs.text(
            obs_i[0],
            obs_i[1] - 0.3,
            f"{obs_idx}",
            fontsize=13,
            ha="center",
            va="center",
            color="k",
            clip_on=True,
        )

    for i_hist, hist in enumerate(hist_list):
        hist_state_arr = np.vstack(hist.state)
        center_tracking_error = np.abs(
            np.linalg.norm(hist_state_arr[:, 0:2] - CENTER, axis=1) - R_CETERLINE
        )
        print(
            f"[INFO] Path-tracking error of {cbf_list[i_hist].name}: {np.mean(center_tracking_error):.2f} m"
        )

        axs.plot(
            hist_state_arr[:, 0],
            hist_state_arr[:, 1],
            color=cm[i_hist],
            lw=lw,
            ls=ls[i_hist],
            label=cbf_list[i_hist].name,
        )

        # --- Add inset zoom plot ---
        if i_hist == 0:
            ax_inset: plt.Axes = inset_axes(
                axs, width="38%", height="38%", loc="center"
            )
            # Alternatively: ax_inset = axs.inset_axes([0.5, 0.5, 0.48, 0.48]) for manual placement
            ax_inset.add_patch(
                plt.Circle(CENTER, R_OUTER, color="k", fill=False, lw=lw)
            )
            ax_inset.add_patch(
                plt.Circle(CENTER, R_INNER, color="k", fill=False, lw=lw)
            )
            ax_inset.add_patch(
                plt.Circle(
                    CENTER,
                    R_OUTER - R_ROBOT,
                    edgecolor=safety_boundary_c,
                    fill=False,
                    lw=lw / 2,
                    linestyle="--",
                )
            )
            ax_inset.add_patch(
                plt.Circle(
                    CENTER,
                    R_INNER + R_ROBOT,
                    edgecolor=safety_boundary_c,
                    fill=False,
                    lw=lw / 2,
                    linestyle="--",
                )
            )
            ax_inset.add_patch(
                plt.Circle(
                    CENTER,
                    R_CETERLINE,
                    color="k",
                    linestyle="-.",
                    fill=False,
                    lw=lw / 2,
                    alpha=0.4,
                )
            )
            obs_idx = 0
            obs_i = all_obs_xy_list[obs_idx]
            ax_inset.add_patch(
                plt.Circle(
                    obs_i, R_OBS, facecolor="gray", edgecolor="black", fill=True, lw=lw
                )
            )
            ax_inset.add_patch(
                plt.Circle(
                    obs_i,
                    R_OBS + R_ROBOT,
                    edgecolor=safety_boundary_c,
                    fill=False,
                    lw=lw / 2,
                    linestyle="--",
                )
            )
            ax_inset.text(
                obs_i[0],
                obs_i[1] - 0.3,
                f"{obs_idx}",
                fontsize=20,
                ha="center",
                va="center",
                color="k",
                clip_on=True,
            )

            x1, x2 = 30, 46
            y1, y2 = -8, 8
            ax_inset.set_xlim(x1, x2)
            ax_inset.set_ylim(y1, y2)
            # ax_inset.set_xlabel(r"$x$ [m]")
            # ax_inset.set_ylabel(r"$y$ [m]")
            ax_inset.set_xticks(np.arange(x1, x2 + 0.1, 5))
            ax_inset.set_yticks([-5, 0, 5])

            ax_inset.grid(True, linestyle=":", alpha=0.4)
            ax_inset.tick_params(axis="both", labelsize=11, direction="in")

            # Add a rectangle on the main plot to indicate the zoomed region
            zoom_rect = Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                # (1.8, -0.5), 2.6 - 1.8, 4.2 - (-0.5),
                linewidth=1.0,
                edgecolor="k",
                facecolor="none",
                zorder=6,
            )
            axs.add_patch(zoom_rect)

            # Connect the rectangle to the inset
            mark_inset(axs, ax_inset, loc1=1, loc2=4, fc="none", ec="k", lw=1)

        ax_inset.plot(
            hist_state_arr[:, 0],
            hist_state_arr[:, 1],
            color=cm[i_hist],
            lw=lw,
            ls=ls[i_hist],
            label=cbf_list[i_hist].name,
        )

        t_list = np.arange(7.8, 9.3, 0.2)

        for t in t_list:
            j = np.int32(np.ceil(t / DT))
            ax_inset.scatter(
                hist_state_arr[j, 0], hist_state_arr[j, 1], s=25, c=cm[i_hist], zorder=5
            )
            angle = np.arctan2(hist_state_arr[j, 1], hist_state_arr[j, 0])
            distance = np.linalg.norm(hist_state_arr[j, 0:2])
            if i_hist == 0:
                pos_text = distance * 0.95 * np.array([np.cos(angle), np.sin(angle)])
                ax_inset.text(
                    pos_text[0],
                    pos_text[1],
                    f"{t:.1f} s",
                    fontsize=12,
                    ha="center",
                    va="center",
                    color="k",
                    clip_on=True,
                    bbox=dict(
                        facecolor="white",
                        alpha=0.6,
                        edgecolor="none",
                        boxstyle="round,pad=0",
                    ),
                )

        # Initial position
        if i_hist == 0:
            init_circle = plt.Circle(
                (hist_state_arr[0, 0], hist_state_arr[0, 1]),
                R_ROBOT,
                color="k",
                linestyle="--",
                fill=False,
                lw=1.2,
            )
            axs.arrow(
                hist_state_arr[0, 0],
                hist_state_arr[0, 1],
                6 * np.cos(hist_state_arr[0, 2]),
                6 * np.sin(hist_state_arr[0, 2]),
                head_width=1.5,
                head_length=2.5,
                fc="k",
                ec="k",
                zorder=6,
            )
            axs.add_patch(init_circle)
        # End position
        axs.scatter(
            hist_state_arr[-1, 0], hist_state_arr[-1, 1], s=25, c=cm[i_hist], zorder=5
        )
        if i_hist == 0:
            axs.text(
                hist_state_arr[-1, 0] + 6,
                hist_state_arr[-1, 1],
                f"{hist.t[-1]:.1f} s",
                fontsize=12,
                ha="center",
                va="center",
                color="k",
                clip_on=True,
                bbox=dict(
                    facecolor="white",
                    alpha=0.6,
                    edgecolor="none",
                    boxstyle="round,pad=0",
                ),
            )

        # highlight the actual collision points
        # Boolean masks for collisions (for red dots)
        coll_mask = np.array([c.startswith("COLL") for c in hist.evt])
        if np.any(coll_mask):
            coll_xy_arr = hist_state_arr[coll_mask, 0:2]

            for i in range(len(coll_xy_arr)):
                circle = plt.Circle(
                    (coll_xy_arr[i, 0], coll_xy_arr[i, 1]),
                    R_ROBOT,
                    color=cm[i_hist],
                    fill=False,
                    lw=lw,
                    linestyle="--",
                )
                axs.add_patch(circle)

    first_legend = axs.legend(loc="lower right", frameon=True, framealpha=1.0)
    # Create a proxy artist
    safety_line = mlines.Line2D(
        [],
        [],
        color=safety_boundary_c,
        linestyle="--",
        lw=lw / 3,
        label="Safety boundary",
    )
    # Create a second legend
    second_legend = axs.legend(
        handles=[safety_line], loc="upper left", frameon=True, framealpha=1.0
    )
    # Re-add the first legend
    axs.add_artist(first_legend)

    plt.tight_layout()

    fig_name = fig_name_prefix.parent / (
        fig_name_prefix.name + "_trajectories.pdf"
    )

    plt.savefig(fig_name, bbox_inches="tight", pad_inches=0)
    cprint(f"[INFO] Figure saved to {fig_name}")

    # ╔══════════════════════════════════════════════════════════════╗
    #  Plot u1, u2, v, and path-tracking error
    # ╚══════════════════════════════════════════════════════════════╝
    lw = 2.0

    # Plot u1
    fig, axs = plt.subplots(1, 1, figsize=(6, 1.6))
    plt.axhline(0, color="k", linestyle=":", linewidth=lw / 4)
    for i_hist, hist in enumerate(hist_list):
        hist_state_arr = np.vstack(hist.state)
        t_arr = np.array(hist.t)
        axs.plot(
            t_arr,
            hist.u1,
            color=cm[i_hist],
            lw=lw,
            ls=ls[i_hist],
            label=cbf_list[i_hist].name,
        )

    t_max = t_arr[-1]
    axs.fill_between(
        x=[0, t_max],  # full x-axis range
        y1=U1_MAX,
        y2=U1_MAX + 1.0,
        color="tab:red",
        alpha=0.2,
        # label="Unsafe Region"
    )
    axs.fill_between(
        x=[0, t_max],  # full x-axis range
        y1=-U1_MAX - 1.0,
        y2=-U1_MAX,
        color="tab:red",
        alpha=0.2,
        # label="Unsafe Region"
    )
    plt.axhline(U1_MAX, color="tab:red", linestyle="--", linewidth=lw / 2)
    plt.axhline(-U1_MAX, color="tab:red", linestyle="--", linewidth=lw / 2)

    axs.set_xlabel(r"Time $t$ [s]")
    axs.set_ylabel(r"$u_1$ [rad/s]")
    axs.set_xlim(0, t_max)
    axs.set_ylim(-U1_MAX - 0.5, U1_MAX + 0.5)
    axs.set_yticks([-U1_MAX, 0, U1_MAX])

    # Add second legend
    legend1 = axs.legend(loc="lower right", frameon=True, framealpha=1.0)
    bound_line = Line2D(
        [0],
        [0],
        color="tab:red",
        linestyle="--",
        linewidth=lw / 2,
        label=r"$u_{1,\mathrm{max}/\mathrm{min}}$",
    )
    legend2 = axs.legend(
        handles=[bound_line], loc="lower left", frameon=True, framealpha=1.0
    )
    axs.add_artist(legend1)  # Re-add the first legend so it does not get overwritten

    axs.grid(True, linestyle=":", alpha=0.4)
    axs.tick_params(axis="both", direction="in")

    plt.tight_layout()

    fig_name = fig_name_prefix.parent / (
        fig_name_prefix.name + "_u1.pdf"
    )

    plt.savefig(fig_name, bbox_inches="tight", pad_inches=0)
    cprint(f"[INFO] Figure saved to {fig_name}")

    # Plot u2
    fig, axs = plt.subplots(1, 1, figsize=(6, 1.6))
    plt.axhline(0, color="k", linestyle=":", linewidth=lw / 4)
    for i_hist, hist in enumerate(hist_list):
        hist_state_arr = np.vstack(hist.state)
        t_arr = np.array(hist.t)
        axs.plot(
            t_arr,
            hist.u2,
            color=cm[i_hist],
            lw=lw,
            ls=ls[i_hist],
            label=cbf_list[i_hist].name,
        )

        # --- Add inset zoom plot ---
        if i_hist == 0:
            ax_inset = axs.inset_axes([0.45, 0.2, 0.2, 0.6])  # adjust numbers as needed
            ax_inset.axhline(0, color="k", linestyle=":", linewidth=lw / 4)
            x1, x2 = 7, 10
            y1, y2 = -0.4, 0.4
            ax_inset.set_xlim(x1, x2)
            ax_inset.set_ylim(y1, y2)
            ax_inset.set_xticks(np.arange(x1, x2 + 0.1, 1))
            ax_inset.set_yticks([-0.4, 0, 0.4])
            ax_inset.grid(True, linestyle=":", alpha=0.4)
            ax_inset.tick_params(axis="both", labelsize=11, direction="in")
            zoom_rect = Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=1.0,
                edgecolor="k",
                facecolor="none",
                zorder=6,
            )
            axs.add_patch(zoom_rect)
            mark_inset(axs, ax_inset, loc1=2, loc2=3, fc="none", ec="k", lw=1)
        ax_inset.plot(
            t_arr,
            hist.u2,
            color=cm[i_hist],
            lw=lw,
            ls=ls[i_hist],
            label=cbf_list[i_hist].name,
        )

    t_max = t_arr[-1]
    axs.fill_between(
        x=[0, t_max],  # full x-axis range
        y1=U2_MAX,
        y2=U2_MAX + 1.0,
        color="tab:red",
        alpha=0.2,
        # label="Unsafe Region"
    )
    axs.fill_between(
        x=[0, t_max],  # full x-axis range
        y1=-U2_MAX - 1.0,
        y2=-U2_MAX,
        color="tab:red",
        alpha=0.2,
        # label="Unsafe Region"
    )
    plt.axhline(U2_MAX, color="tab:red", linestyle="--", linewidth=lw / 2)
    plt.axhline(-U2_MAX, color="tab:red", linestyle="--", linewidth=lw / 2)

    axs.set_xlabel(r"Time $t$ [s]")
    axs.set_ylabel(r"$u_2$ [m/s$^2$]")
    axs.set_xlim(0, t_max)
    axs.set_ylim(-U2_MAX - 0.5, U2_MAX + 0.5)
    axs.set_yticks([-U2_MAX, 0, U2_MAX])

    # Add second legend
    legend1 = axs.legend(loc="lower right", frameon=True, framealpha=0.8)
    bound_line = Line2D(
        [0],
        [0],
        color="tab:red",
        linestyle="--",
        linewidth=lw / 2,
        label=r"$u_{2,\mathrm{max}/\mathrm{min}}$",
    )
    legend2 = axs.legend(
        handles=[bound_line], loc="lower left", frameon=True, framealpha=1.0
    )
    axs.add_artist(legend1)  # Re-add the first legend so it does not get overwritten
    axs.grid(True, linestyle=":", alpha=0.4)
    axs.tick_params(axis="both", direction="in")

    plt.tight_layout()
    fig_name = fig_name_prefix.parent / (
        fig_name_prefix.name + "_u2.pdf"
    )
    plt.savefig(fig_name, bbox_inches="tight", pad_inches=0)
    cprint(f"[INFO] Figure saved to {fig_name}")

    # Plot v
    fig, axs = plt.subplots(1, 1, figsize=(6, 1.6))
    plt.axhline(V_REF, color="tab:green", linestyle="--", linewidth=lw / 2)
    for i_hist, hist in enumerate(hist_list):
        hist_state_arr = np.vstack(hist.state)
        t_arr = np.array(hist.t)
        axs.plot(
            t_arr,
            hist.v,
            color=cm[i_hist],
            lw=lw,
            ls=ls[i_hist],
            label=cbf_list[i_hist].name,
        )

        # --- Add inset zoom plot ---
        if i_hist == 0:
            ax_inset = axs.inset_axes(
                [0.25, 0.21, 0.45, 0.45]
            )  # adjust numbers as needed
            ax_inset.axhline(V_REF, color="tab:green", linestyle="--", linewidth=lw / 2)
            x1, x2 = 7, 10
            y1, y2 = 9.6, 10.2
            ax_inset.set_xlim(x1, x2)
            ax_inset.set_ylim(y1, y2)
            ax_inset.set_xticks(np.arange(x1, x2 + 0.1, 1))
            ax_inset.set_yticks([9.6, 9.8, 10.0, 10.2])
            ax_inset.grid(True, linestyle=":", alpha=0.4)
            ax_inset.tick_params(axis="both", labelsize=11, direction="in")
            zoom_rect = Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=1.0,
                edgecolor="k",
                facecolor="none",
                zorder=6,
            )
            axs.add_patch(zoom_rect)
            mark_inset(axs, ax_inset, loc1=1, loc2=2, fc="none", ec="k", lw=1)
        ax_inset.plot(
            t_arr,
            hist.v,
            color=cm[i_hist],
            lw=lw,
            ls=ls[i_hist],
            label=cbf_list[i_hist].name,
        )

    t_max = t_arr[-1]
    axs.fill_between(
        x=[0, t_max],  # full x-axis range
        y1=V_REF - 0.5,
        y2=V_REF + 0.5,
        color="tab:green",
        alpha=0.2,
    )

    axs.set_xlabel(r"Time $t$ [s]")
    axs.set_ylabel(r"$v$ [m/s]")
    axs.set_xlim(0, t_max)
    axs.set_ylim(-1, V_REF + 2)

    legend1 = axs.legend(loc="lower right", frameon=True, framealpha=0.8)
    # Create dummy handles for the second legend (safety bounds and unsafe region)
    bound_line = Line2D(
        [0],
        [0],
        color="tab:green",
        linestyle="--",
        linewidth=lw / 2,
        label=r"$v_{\mathrm{des}}$",
    )
    # Add second legend
    legend2 = axs.legend(
        handles=[bound_line], loc="upper left", frameon=True, framealpha=0.8
    )
    axs.add_artist(legend1)  # Re-add the first legend so it does not get overwritten
    axs.grid(True, linestyle=":", alpha=0.4)
    axs.tick_params(axis="both", direction="in")

    plt.tight_layout()
    fig_name = fig_name_prefix.parent / (
        fig_name_prefix.name + "_v.pdf"
    )
    plt.savefig(fig_name, bbox_inches="tight", pad_inches=0)
    cprint(f"[INFO] Figure saved to {fig_name}")

    # Plot path-tracking error
    fig, axs = plt.subplots(1, 1, figsize=(6, 1.6))
    plt.axhline(0, color="k", linestyle=":", linewidth=lw / 4)
    for i_hist, hist in enumerate(hist_list):
        hist_state_arr = np.vstack(hist.state)
        t_arr = np.array(hist.t)
        center_tracking_error = np.abs(
            np.linalg.norm(hist_state_arr[:, 0:2] - CENTER, axis=1) - R_CETERLINE
        )
        axs.plot(
            t_arr,
            center_tracking_error,
            color=cm[i_hist],
            lw=lw,
            ls=ls[i_hist],
            label=cbf_list[i_hist].name,
        )

    axs.set_xlabel(r"Time $t$ [s]")
    axs.set_ylabel(r"$e_{\mathrm{path}}$ [m]")
    axs.set_xlim(0, t_arr[-1])
    axs.set_ylim(-0.2, 2)
    axs.set_yticks([0, 1, 2])
    axs.legend(loc="upper right", frameon=True, framealpha=0.8)
    axs.grid(True, linestyle=":", alpha=0.4)
    axs.tick_params(axis="both", direction="in")

    plt.tight_layout()

    fig_name = fig_name_prefix.parent / (
        fig_name_prefix.name + "_path_tracking_error.pdf"
    )

    plt.savefig(fig_name, bbox_inches="tight", pad_inches=0)
    cprint(f"[INFO] Figure saved to {fig_name}")

    # ╔════════════════════════════════════════════════════════════════════════╗
    #  Plot adaptive parameters (lambda_ttcbf, p1_pacbf, p2_pacbf, r_pacbf)
    # ╚════════════════════════════════════════════════════════════════════════╝
    #  Plot adaptive parameters lambda_ttcbf
    if CBFMethod.aTTCBF in cbf_list:
        fig, axs = plt.subplots(1, 1, figsize=(6, 1.8))
        # Index 0 for inner corridor, 1 for outer corridor, 2-17 for obstacles
        obs_idx_list = np.arange(0, 5, 1)

        labels_obs = [
            "Inner boundary",
            "Outer boundary",
            "Obstacle 0",
            "Obstacle 1",
            "Obstacle 2",
        ]
        ls = ["--", "--", "-.", "-.", "-."]
        zorder = [2, 2, 3, 2, 2]
        lw = 2.0

        idx_cbf = cbf_list.index(CBFMethod.aTTCBF)
        cm = get_colormap(len(obs_idx_list), style="tab10")
        arr_list = np.vstack(hist_list[idx_cbf].lambda_ttcbf)
        for obs_idx in obs_idx_list:
            axs.plot(
                hist_list[idx_cbf].t,
                arr_list[:, obs_idx],
                color=cm[obs_idx],
                lw=lw,
                ls=ls[obs_idx],
                label=labels_obs[obs_idx],
                zorder=zorder[obs_idx],
            )

        arr_obs_0 = arr_list[:, 2]
        max_idx = np.argmax(arr_obs_0)
        t_max = hist_list[idx_cbf].t[max_idx]
        y_max = arr_list[max_idx, 2]
        axs.annotate(
            rf"$t = {t_max:.2f}$ s",
            xy=(t_max, y_max),
            xytext=(t_max - 4.0, y_max + 0.05),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            fontsize=10,
            color="black",
            ha="left",
            va="center",
        )

        # Find the index of the first point after the maximum point on arr_obs_0 where the value is very close to zero
        zero_idx = np.where(arr_obs_0[max_idx:] < 1e-4)[0]
        if len(zero_idx) > 0:
            zero_idx = zero_idx[0] + max_idx
            t_zero = hist_list[idx_cbf].t[zero_idx]
            y_zero = arr_list[zero_idx, 2]
            axs.annotate(
                rf"$t = {t_zero:.2f}$ s",
                xy=(t_zero, y_zero),
                xytext=(t_zero - 4.5, y_zero + 0.25),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                fontsize=10,
                color="black",
                ha="left",
                va="center",
            )

        axs.grid(True, linestyle=":", alpha=0.4)
        axs.set_xlim(0, hist_list[idx_cbf].t[-1])
        axs.set_xticks(np.arange(0, hist_list[idx_cbf].t[-1] + 0.1, 2.0))
        axs.set_xlabel(r"Time $t$ [s]")
        axs.set_ylabel(r"$\eta$")
        axs.tick_params(axis="both", direction="in")

        axs.legend(loc="upper right", frameon=True, framealpha=0.5, fontsize=9)
        plt.tight_layout()

        fig_name = fig_name_prefix.parent / (
            fig_name_prefix.name + f"_adaption_{CBFMethod.aTTCBF.name.lower()}.pdf"
        )

        plt.savefig(fig_name, bbox_inches="tight", pad_inches=0)
        cprint(f"[INFO] Figure saved to {fig_name}")

    # Plot adaptive parameters p1_pacbf and p2_pacbf
    if CBFMethod.PACBF in cbf_list:
        fig, axs = plt.subplots(1, 1, figsize=(6, 1.8))
        # Index 0 for inner corridor, 1 for outer corridor, 2-17 for obstacles
        obs_idx_list = np.arange(0, 5, 1)

        labels_obs = [
            "Inner boundary",
            "Outer boundary",
            "Obstacle 0",
            "Obstacle 1",
            "Obstacle 2",
        ]
        ls = ["--", "--", "-.", "-.", "-."]
        zorder = [2, 2, 3, 2, 2]

        cm = get_colormap(len(obs_idx_list), style="tab10")

        idx_cbf = cbf_list.index(CBFMethod.PACBF)
        arr_list_1 = np.vstack(hist_list[idx_cbf].p1_pacbf)
        arr_list_2 = np.vstack(hist_list[idx_cbf].p1_pacbf)
        for obs_idx in obs_idx_list:
            axs.plot(
                hist_list[idx_cbf].t,
                arr_list_1[:, obs_idx],
                color=cm[obs_idx],
                lw=lw,
                ls=ls[obs_idx],
                label=labels_obs[obs_idx],
                zorder=zorder[obs_idx],
            )

        arr_obs_0 = arr_list_1[:, 2]
        max_idx = np.argmax(arr_obs_0)
        t_max = hist_list[idx_cbf].t[max_idx]
        y_max = arr_list_1[max_idx, 2]
        axs.annotate(
            rf"$t = {t_max:.2f}$ s",
            xy=(t_max, y_max),
            xytext=(t_max - 4.5, y_max + 0.3),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            fontsize=10,
            color="black",
            ha="left",
            va="center",
        )

        axs.grid(True, linestyle=":", alpha=0.4)
        axs.set_xlim(0, hist_list[idx_cbf].t[-1])
        axs.set_xticks(np.arange(0, hist_list[idx_cbf].t[-1] + 0.1, 2.0))
        axs.set_xlabel(r"Time $t$ [s]")
        axs.set_ylabel(r"$p_1$")
        axs.tick_params(axis="both", direction="in")

        axs.legend(loc="upper right", frameon=True, framealpha=0.5, fontsize=9)
        plt.tight_layout()

        fig_name = fig_name_prefix.parent / (
            fig_name_prefix.name + f"_adaption_{CBFMethod.PACBF.name.lower()}_p1.pdf"
        )

        plt.savefig(fig_name, bbox_inches="tight", pad_inches=0)
        cprint(f"[INFO] Figure saved to {fig_name}")

    if CBFMethod.PACBF in cbf_list:
        fig, axs = plt.subplots(1, 1, figsize=(6, 1.8))
        # Index 0 for inner corridor, 1 for outer corridor, 2-17 for obstacles
        obs_idx_list = np.arange(0, 5, 1)

        labels_obs = [
            "Inner boundary",
            "Outer boundary",
            "Obstacle 0",
            "Obstacle 1",
            "Obstacle 2",
        ]
        ls = ["--", "--", "-.", "-.", "-."]
        zorder = [2, 2, 3, 2, 2]

        cm = get_colormap(len(obs_idx_list), style="tab10")
        arr_list_2 = np.vstack(hist_list[idx_cbf].p2_pacbf)
        for obs_idx in obs_idx_list:
            axs.plot(
                hist_list[idx_cbf].t,
                arr_list_2[:, obs_idx],
                color=cm[obs_idx],
                lw=lw,
                ls=ls[obs_idx],
                label=labels_obs[obs_idx],
                zorder=zorder[obs_idx],
            )

        arr_obs_0 = arr_list_2[:, 2]
        max_idx = np.argmax(arr_obs_0)
        t_max = hist_list[idx_cbf].t[max_idx]
        y_max = arr_list_2[max_idx, 2]
        axs.annotate(
            rf"$t = {t_max:.2f}$ s",
            xy=(t_max, y_max),
            xytext=(t_max - 3, y_max + 4),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            fontsize=10,
            color="black",
            ha="left",
            va="center",
        )

        # Find the index of the first point after the maximum point on arr_obs_0 where the value is very close to zero
        zero_idx = np.where(arr_obs_0[max_idx:] < 1e-4)[0]
        if len(zero_idx) > 0:
            zero_idx = zero_idx[0] + max_idx
            t_zero = hist_list[idx_cbf].t[zero_idx]
            y_zero = arr_list_2[zero_idx, 2]
            axs.annotate(
                rf"$t = {t_zero:.2f}$ s",
                xy=(t_zero, y_zero),
                xytext=(t_zero - 5, y_zero + 10),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                fontsize=10,
                color="black",
                ha="left",
                va="center",
            )

        axs.grid(True, linestyle=":", alpha=0.4)
        axs.set_xlim(0, hist_list[idx_cbf].t[-1])
        axs.set_xticks(np.arange(0, hist_list[idx_cbf].t[-1] + 0.1, 2.0))
        axs.set_xlabel(r"Time $t$ [s]")
        axs.set_ylabel(r"$p_2$")
        axs.tick_params(axis="both", direction="in")

        axs.legend(loc="upper right", frameon=True, framealpha=0.5, fontsize=9)
        plt.tight_layout()

        fig_name = fig_name_prefix.parent / (
            fig_name_prefix.name + f"_adaption_{CBFMethod.PACBF.name.lower()}_p2.pdf"
        )

        plt.savefig(fig_name, bbox_inches="tight", pad_inches=0)
        cprint(f"[INFO] Figure saved to {fig_name}")

    # r_racbf
    if CBFMethod.RACBF in cbf_list:
        fig, axs = plt.subplots(1, 1, figsize=(6, 1.8))
        # Index 0 for inner corridor, 1 for outer corridor, 2-17 for obstacles

        labels_obs = [
            "Inner boundary",
            "Outer boundary",
            "Obstacle 0",
            "Obstacle 1",
            "Obstacle 2",
        ]
        obs_idx_list = [0, 1, 2, 3, 4]

        ls = ["--", "--", "-.", "-.", "-."]
        zorder = [2, 2, 3, 2, 2]

        cm = get_colormap(len(labels_obs), style="tab10")
        idx_cbf = cbf_list.index(CBFMethod.RACBF)
        arr_list = np.vstack(hist_list[idx_cbf].r_racbf)
        for obs_idx in obs_idx_list:
            axs.plot(
                hist_list[idx_cbf].t,
                arr_list[:, obs_idx],
                color=cm[obs_idx],
                lw=lw,
                ls=ls[obs_idx],
                label=labels_obs[obs_idx],
                zorder=zorder[obs_idx],
            )

        arr_obs_0 = arr_list[:, 2]

        # Annotate the first change point
        first_change_idx = np.where(arr_obs_0 != arr_obs_0[0])[0][0]
        t_first_change = hist_list[idx_cbf].t[first_change_idx]
        y_first_change = arr_obs_0[first_change_idx]
        axs.annotate(
            rf"$t = {t_first_change:.2f}$ s",
            xy=(t_first_change, y_first_change),
            xytext=(t_first_change - 2.0, y_first_change - 0.03),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            fontsize=10,
            color="black",
            ha="left",
            va="center",
        )
        # Annotate the first point after the first change where the value is starting to increase again (not larger than the first change)
        first_increase_idx = (
            np.where(np.diff(arr_obs_0[first_change_idx:]) > 0)[0][0]
            + first_change_idx
            + 1
        )
        t_first_increase = hist_list[idx_cbf].t[first_increase_idx]
        y_first_increase = arr_obs_0[first_increase_idx]
        axs.annotate(
            rf"$t = {t_first_increase:.2f}$ s",
            xy=(t_first_increase, y_first_increase),
            xytext=(t_first_increase - 2.0, y_first_increase - 0.03),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            fontsize=10,
            color="black",
            ha="left",
            va="center",
        )

        axs.grid(True, linestyle=":", alpha=0.4)
        axs.set_xlim(0, hist_list[idx_cbf].t[-1])
        axs.set_ylim(0, 2 * R_TARGET)
        axs.set_yticks(np.arange(0, 2 * R_TARGET + 0.001, 0.02))
        axs.set_xticks(np.arange(0, hist_list[idx_cbf].t[-1] + 0.1, 2.0))
        axs.set_xlabel(r"Time $t$ [s]")
        axs.set_ylabel(r"$h_a$")
        axs.tick_params(axis="both", direction="in")

        axs.legend(loc="upper right", frameon=True, framealpha=0.5, fontsize=9)
        plt.tight_layout()

        fig_name = fig_name_prefix.parent / (
            fig_name_prefix.name + f"_adaption_{CBFMethod.RACBF.name.lower()}.pdf"
        )

        plt.savefig(fig_name, bbox_inches="tight", pad_inches=0)
        cprint(f"[INFO] Figure saved to {fig_name}")

    # plt.show()


def export_video(
    hist: CorridorTrajectoryHistory | list[CorridorTrajectoryHistory],
    video_path: str | Path,
    fps: int | None = None,
    dpi: int = 150,
    traj_color: str | list[str] = "tab:blue",
    ls: str | list[str] = "-",
    is_full_map: bool = False,
    labels: str | list[str] | None = None,
    legend_title: str | None = None,
) -> None:
    """
    Export a video visualizing one or multiple robot trajectories.

    Parameters
    ----------
    hist : CorridorTrajectoryHistory or list[CorridorTrajectoryHistory]
        Simulation history (single) or histories (multiple) to visualize.
    video_path : str or Path
        Output video path, e.g., "video/run_linear.mp4".
    fps : int or None
        Frames per second. If None, use round(1 / DT).
    dpi : int
        Figure DPI for rendering quality.
    traj_color : str or list[str]
        Color(s) for trajectories and robot outlines. If list, must match #histories.
    ls : str or list[str]
        Line style(s) for trajectories. If list, must match #histories.
    is_full_map : bool
        Whether to use the full map view or zoomed-in view.
    """
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    if fps is None:
        fps = int(np.round(1.0 / DT))

    # -------------------------
    # Normalize inputs to lists
    # -------------------------
    hist_list: list[CorridorTrajectoryHistory]
    if isinstance(hist, list):
        hist_list = hist
    else:
        hist_list = [hist]

    n_traj = len(hist_list)

    if isinstance(traj_color, list):
        if len(traj_color) != n_traj:
            raise ValueError(
                f"traj_color must have length {n_traj}, got {len(traj_color)}."
            )
        color_list = traj_color
    else:
        color_list = [traj_color for _ in range(n_traj)]

    if isinstance(ls, list):
        if len(ls) != n_traj:
            raise ValueError(f"ls must have length {n_traj}, got {len(ls)}.")
        ls_list = ls
    else:
        ls_list = [ls for _ in range(n_traj)]

    # -------------------------
    # Prepare per-trajectory arrays
    # -------------------------
    xy_list: list[np.ndarray] = []
    theta_list: list[np.ndarray] = []
    t_list: list[np.ndarray] = []
    n_frames_list: list[int] = []

    for h in hist_list:
        state_arr = np.vstack(h.state)  # (N, 4)
        xy_arr = state_arr[:, 0:2]
        theta_arr = state_arr[:, 2]
        xy_list.append(xy_arr)
        theta_list.append(theta_arr)
        t_list.append(np.asarray(h.t, dtype=float))
        n_frames_list.append(xy_arr.shape[0])

    n_frames = int(np.max(n_frames_list))

    # -------------------------
    # Figure and axes
    # -------------------------
    lw = 2.0
    safety_boundary_c = (1, 0, 0, 0.3)

    if is_full_map:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi, constrained_layout=True)
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_xticks(np.arange(-40, 50, 20))
        ax.set_yticks(np.arange(-40, 50, 20))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4), dpi=dpi, constrained_layout=True)
        ax.set_xlim(-5, 50)
        ax.set_ylim(-45, 0)
        ax.set_xticks(np.arange(0, 51, 10))
        ax.set_yticks(np.arange(-45, 1, 5))

    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(axis="both", direction="in")

    # -------------------------
    # Static environment drawing
    # -------------------------
    circle_outer = plt.Circle(CENTER, R_OUTER, color="k", fill=False, lw=lw)
    circle_inner = plt.Circle(CENTER, R_INNER, color="k", fill=False, lw=lw)
    circle_centerline = plt.Circle(
        CENTER,
        R_CETERLINE,
        color="k",
        linestyle="-.",
        fill=False,
        lw=lw / 2,
        alpha=0.4,
    )
    ax.add_patch(circle_outer)
    ax.add_patch(circle_inner)
    ax.add_patch(circle_centerline)

    circle = plt.Circle(
        CENTER,
        R_OUTER - R_ROBOT,
        edgecolor=safety_boundary_c,
        fill=False,
        lw=lw / 2,
        linestyle="--",
    )
    ax.add_patch(circle)
    circle = plt.Circle(
        CENTER,
        R_INNER + R_ROBOT,
        edgecolor=safety_boundary_c,
        fill=False,
        lw=lw / 2,
        linestyle="--",
    )
    ax.add_patch(circle)

    for obs_idx, obs_i in enumerate(all_obs_xy_list):
        circle = plt.Circle(
            obs_i, R_OBS, facecolor="gray", edgecolor="black", fill=True, lw=lw
        )
        ax.add_patch(circle)

        circle = plt.Circle(
            obs_i,
            R_OBS + R_ROBOT,
            edgecolor=safety_boundary_c,
            fill=False,
            lw=lw / 2,
            linestyle="--",
        )
        ax.add_patch(circle)

        ax.text(
            obs_i[0],
            obs_i[1] - 0.3,
            f"{obs_idx}",
            fontsize=13,
            ha="center",
            va="center",
            color="k",
            clip_on=True,
        )

    # -------------------------
    # Dynamic artists (one set per trajectory)
    # -------------------------
    traj_lines: list[plt.Line2D] = []
    robot_circles: list[plt.Circle] = []
    heading_lines: list[plt.Line2D] = []

    arrow_len = 4.0

    for i in range(n_traj):
        # Trajectory line (past path)
        (traj_line,) = ax.plot(
            [],
            [],
            color=color_list[i],
            ls=ls_list[i],
            lw=2.0,
            zorder=3,
            label=labels[i] if labels is not None else None,
        )
        traj_lines.append(traj_line)

        # Robot body (use matching color)
        xy0 = xy_list[i][0]
        robot_circle = plt.Circle(
            (float(xy0[0]), float(xy0[1])),
            R_ROBOT,
            edgecolor=color_list[i],
            facecolor="none",
            lw=2.0,
            zorder=5,
        )
        ax.add_patch(robot_circle)
        robot_circles.append(robot_circle)

        # Heading arrow as a line (use matching color)
        (heading_line,) = ax.plot(
            [],
            [],
            color=color_list[i],
            lw=1.5,
            zorder=6,
        )
        heading_lines.append(heading_line)
    
    legend = ax.legend(
        handles=traj_lines,
        loc="lower right",
        frameon=True,
        framealpha=1.0,
    )

    if legend_title is not None:
        # Need a renderer to get the legend bbox; force one draw
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # Legend bbox in display coords -> convert to axes coords
        bbox_disp = legend.get_window_extent(renderer=renderer)
        bbox_axes = bbox_disp.transformed(ax.transAxes.inverted())

        # Place text centered horizontally, slightly above the legend
        x_center = 0.5 * (bbox_axes.x0 + bbox_axes.x1)
        y_top = bbox_axes.y1

        ax.text(
            x_center,
            y_top + 0.01,          # small vertical gap above legend
            legend_title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            bbox=dict(
                facecolor="white",
                alpha=0.6,
                edgecolor="none",
                boxstyle="round,pad=0.2",
            ),
            zorder=10,
        )

    # Time text (shared)
    time_text = ax.text(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(
            facecolor="white",
            alpha=0.6,
            edgecolor="none",
            boxstyle="round,pad=0.2",
        ),
    )

    # -------------------------
    # Animation functions
    # -------------------------
    def init():
        artists = []
        for i in range(n_traj):
            traj_lines[i].set_data([], [])
            xy0 = xy_list[i][0]
            robot_circles[i].center = (float(xy0[0]), float(xy0[1]))
            heading_lines[i].set_data([], [])
            artists.extend([traj_lines[i], robot_circles[i], heading_lines[i]])
        time_text.set_text("")
        artists.append(time_text)
        return tuple(artists)

    def update(frame_idx: int):
        artists = []

        # Use a global time label (DT-based) to avoid ambiguity across histories
        time_text.set_text(rf"$t = {frame_idx * DT:.2f}\,\mathrm{{s}}$")
        artists.append(time_text)

        for i in range(n_traj):
            n_i = n_frames_list[i]
            idx = min(frame_idx, n_i - 1)

            xy_arr = xy_list[i]
            theta_arr = theta_list[i]

            # Past trajectory up to idx
            traj_lines[i].set_data(
                xy_arr[: idx + 1, 0],
                xy_arr[: idx + 1, 1],
            )

            # Robot pose at idx
            x = float(xy_arr[idx, 0])
            y = float(xy_arr[idx, 1])
            theta = float(theta_arr[idx])
            robot_circles[i].center = (x, y)

            # Heading arrow
            x_head = x + arrow_len * float(np.cos(theta))
            y_head = y + arrow_len * float(np.sin(theta))
            heading_lines[i].set_data([x, x_head], [y, y_head])

            artists.extend([traj_lines[i], robot_circles[i], heading_lines[i]])

        return tuple(artists)

    # -------------------------
    # Create animation
    # -------------------------
    frames_iter = tqdm(range(n_frames), total=n_frames, desc="Rendering video", unit="frame")

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frames_iter,
        interval=1000.0 / fps,
        blit=True,
    )

    # -------------------------
    # Write video
    # -------------------------
    writer = animation.FFMpegWriter(
        fps=fps,
        metadata=dict(artist="export_video"),
        bitrate=3000,
    )

    anim.save(str(video_path), writer=writer)
    plt.close(fig)

    cprint(f"[INFO] Video saved to {video_path.resolve()}")
