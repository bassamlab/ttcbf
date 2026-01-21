# Copyright (c) 2026, Professorship for Adaptive Behavior of Autonomous Vehicles, University of the Bundeswehr Munich.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum, auto
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import cvxpy as cp
import math
from pathlib import Path
import pickle
from matplotlib import animation
from matplotlib.patches import Rectangle
from common import SpringMassTrajectory, cprint
from matplotlib.lines import Line2D
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
        "text.usetex": True,
    }
)


class CBFMethod(Enum):
    HOCBF = auto()
    TTCBF = auto()
    NONE = auto()  # Without CBF


CBF_METHOD = CBFMethod.TTCBF
IS_USE_SLACK_CBF = True

IS_ADAPTIVE_TTCBF = False
IS_ADAPTIVE_HOCBF = False

r = 6  # relative degree
# --------------------------------------------------------------------------- #
# Physical parameters (can be tuned)
# --------------------------------------------------------------------------- #
m1 = m2 = m3 = 1.0  # [kg]   masses
k1 = k2 = 5.0  # [N/m] spring constants
x3_ref = 1.0  # [m]   desired position of mass-3
x3_max = 1.5  # [m]   maximum position of mass-3
lambda_nom = 2.0  # [1/s] dominant pole location for (s+λ)^6
T_end, dt = 15.0, 0.01  # [s], [s] simulation duration & step size

# [m] Equilibrium positions
x_equilibrium = np.array([0.0, 1.0, 2.0])
# --------------------------------------------------------------------------- #
# State-space matrices (linearised about the force-free equilibrium)
# --------------------------------------------------------------------------- #
A = np.array(
    [
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [-k1 / m1, k1 / m1, 0, 0, 0, 0],
        [k1 / m2, -(k1 + k2) / m2, k2 / m2, 0, 0, 0],
        [0, k2 / m3, -k2 / m3, 0, 0, 0],
    ]
)

B = np.array([0, 0, 0, 1 / m1, 0, 0])
C_y = np.array([0, 0, 1, 0, 0, 0])  # output: x3
CA5B_y = float(C_y @ np.linalg.matrix_power(A, 5) @ B)  # decoupling constant
CA6_y = C_y @ np.linalg.matrix_power(A, 6)  # needed for feed-forward

C_h = np.array([0, 0, -1, 0, 0, 0])  # output: -x3
CA_h_list = [
    C_h @ np.linalg.matrix_power(A, k) for k in range(0, r + 1)
]  # [C, CA, CA^2, …, CA^r]
LgLf_h = float(CA_h_list[-2] @ B)  # C A^(r-1) B

lambdas_cbf = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # λ1, λ2, ..., λr
elem_cbf = np.poly(-lambdas_cbf)  # e0, e1, e2, ..., er

lam_ttc = 0.9
hp = 16 * r
dt_taylor = hp * dt

U_MIN = -5.0
U_MAX = +5.0

W_NOM = 1.0  # quadratic weight on ‖u − u_nom‖²
W_LAM_TTCBF = 5  # quadratic weight on ‖λ_TTCBF‖²
W_S_CBF = 1e8  # linear penalty on δ_CBF


# --------------------------------------------------------------------------- #
# Helper: desired error dynamics coefficients for (s + λ)^6
# --------------------------------------------------------------------------- #
coeff_desc = np.poly1d([1.0, lambda_nom]) ** 6  # s^6 + a5 s^5 + … + a0 with λ = 2
a5, a4, a3, a2, a1, a0 = coeff_desc.coeffs[1:]  # drop leading 1
# Pre-compute C A^i for i = 0 … 5 (needed each step)
CA_powers = [C_y @ np.linalg.matrix_power(A, i) for i in range(6)]

# --------------------------------------------------------------------------- #
# System dynamics (full nonlinear – includes rest lengths explicitly)
# --------------------------------------------------------------------------- #
def spring_forces(x: np.ndarray) -> Tuple[float, float]:
    """Return forces in spring-1 and spring-2."""
    f1 = k1 * (x[1] - x[0])  # k1 (x2 − x1)
    f2 = k2 * (x[2] - x[1])  # k2 (x3 − x2)
    return f1, f2


def dynamics(x: np.ndarray, u: float) -> np.ndarray:
    """Continuous-time dynamics  ẋ = f(x, u)."""
    f1, f2 = spring_forces(x)
    xdot = np.zeros_like(x)

    # Positions
    xdot[0:3] = x[3:6]

    # Accelerations (Newton’s 2nd law)
    xdot[3] = (u + f1) / m1
    xdot[4] = (-f1 + f2) / m2
    xdot[5] = (-f2) / m3
    return xdot


# --------------------------------------------------------------------------- #
# Nominal I/O-linearising control law (CLF)
# --------------------------------------------------------------------------- #
def nominal_controller(x: np.ndarray) -> float:
    """
    Exact I/O linearisation followed by pole placement:
        y^(6) + Σ ai y^(i) = 0   with ai from (s+λ)^6.
    """
    # Output and derivatives y, ẏ, …, y^(5)
    y_derivs = np.array([(CA_powers[i] @ x).item() for i in range(6)])

    # Error derivatives (reference and its derivatives are zero)
    e = np.empty_like(y_derivs)
    e[0] = y_derivs[0] - x3_ref  # position error
    e[1:] = y_derivs[1:]  # derivative errors

    # Virtual control v = desired 6th derivative
    v = -(a5 * e[5] + a4 * e[4] + a3 * e[3] + a2 * e[2] + a1 * e[1] + a0 * e[0])

    # True input: u = (v – C A⁶ x) / (C A⁵ B)
    u = (v - float(CA6_y @ x)) / CA5B_y

    # Bound u
    u = np.clip(u, U_MIN, U_MAX)

    return u


class CBF_QP:
    def __init__(self, cbf_method: CBFMethod):
        self.cbf_method = cbf_method
        self.hist: dict[str, list[float]] = {
            "t": [],
            "x": [],
            "lam_ttcbf": [],
            "delta": [],
            "u": [],  # control inputs
            "h": [],
            "h_k_plus_r_pred": [],
            "z": [],
            "v": [],
            "a": [],
            "pos": [],
            "cost": [],
            "cost_nom": [],
            "cost_cbf": [],
            "psi_r_cbf": [],
            "psi_r_u": [],
            "psi_r_drift": [],
            "psi_r_alpha": [],
            "psi_predicted": [],  # Predicted psi_r for the next 1, 2, ..., n steps
            "is_qp_feasible": [],
            "dr_h": [],
            "delta_var": [],
            "delta_cbf_var": [],
            "ct_qp_total": [],
            "ct_parameter_tuning": [],
            "n_param_tuning": [],  # Number of iterations for parameter tuning
        }

    def build(self, x, u_nom, k):
        self.hist["x"].append(x)
        self.hist["t"].append(k * dt)

        self.u = cp.Variable(name="u")
        self.s_cbf = cp.Variable(nonneg=True, name="s_cbf") if IS_USE_SLACK_CBF else 0.0

        self.u_nom = u_nom

        self.obj = 0

        self.obj_nom = W_NOM * cp.square(self.u - self.u_nom)
        self.obj += self.obj_nom

        self.cons = [
            self.u >= U_MIN,
            self.u <= U_MAX,
        ]

        if IS_USE_SLACK_CBF:
            self.obj_cbf = W_S_CBF * cp.square(self.s_cbf)
            self.obj += self.obj_cbf

        if IS_ADAPTIVE_TTCBF and self.cbf_method is CBFMethod.TTCBF:
            self.lam_ttcbf = cp.Variable(nonneg=True, name="lam_ttcbf")
            self.obj_lam = W_LAM_TTCBF * cp.square(self.lam_ttcbf)
            self.obj += self.obj_lam
            self.cons += [self.lam_ttcbf <= 1.0]
        else:
            self.lam_ttcbf = lam_ttc

        self.obj_fcn = cp.Minimize(self.obj)

        # h and its derivatives: [h, dh, ddh, …, h^{(r-1)}, h^{(r)}]
        self.h_list = [float(CA_h_list[0] @ x + x3_max)]
        for i in range(1, r):
            self.h_list.append(float(CA_h_list[i] @ x))

        # highest derivative with control
        h_r_drift = float(CA_h_list[r] @ x)
        self.h_list.append(h_r_drift + LgLf_h * self.u)

        if self.cbf_method is CBFMethod.HOCBF:
            psi_r_cbf = elem_cbf[::-1] @ self.h_list
            self.psi_r_cbf = psi_r_cbf
            self.cons.append(psi_r_cbf >= (-self.s_cbf if IS_USE_SLACK_CBF else 0.0))
        elif self.cbf_method is CBFMethod.TTCBF:  # TTCBF
            taylor_high_order = 0.0
            for k in range(1, r):
                taylor_high_order += (
                    self.h_list[k] * dt_taylor**k / math.factorial(k)
                )  # dh * dt + ddh * dt²/2 + ... + h^{(r-1)} * dt^{r-1}/(r-1)!
            taylor_high_order += (
                self.h_list[-1] * dt_taylor**r / math.factorial(r)
            )  # h^{(r)} * dt^r/r!
            psi_r_cbf = (
                taylor_high_order + self.lam_ttcbf * self.h_list[0]
            )  # class K term
            # Taylor remainder
            dr_h_previous = self.hist["dr_h"][-1] if len(self.hist["dr_h"]) > 0 else 0.0
            dr_h_min = LgLf_h * (U_MIN if LgLf_h >= 0 else U_MAX) + h_r_drift
            dr_plus_1_h_min = (dr_h_min - dr_h_previous) / dt_taylor
            R = (
                dr_plus_1_h_min * dt_taylor ** (r + 1) / math.factorial(r + 1)
            )  # Taylor remainder
            psi_r_cbf += R
            self.h_k_plus_r_pred = (
                taylor_high_order + self.h_list[0] + R
            )  # Predicted h_{k+r} = h_k + h_{k+1} * dt + h_{k+2} * dt²/2 + ... + h_{k+r} * dt^{r-1}/(r-1)!

            self.cons.append(psi_r_cbf >= (-self.s_cbf if IS_USE_SLACK_CBF else 0.0))

            self.psi_r_cbf = psi_r_cbf
        else:
            # Without CBF
            self.psi_r_cbf = None
            self.h_list = None
            self.h_k_plus_r_pred = None

    def solve(self):
        self.prob = cp.Problem(self.obj_fcn, self.cons)

        self.prob.solve(solver=cp.OSQP, warm_start=True, verbose=False, max_iter=200000)
        self.hist["ct_qp_total"].append(self.prob._solve_time)

        # self.prob.solve(solver=cp.ECOS, warm_start=True, verbose=False)
        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            print(f"[WARNING] QP failed: {self.prob.status}")
            self.hist["is_qp_feasible"].append(False)
            u = 10.0
            prob_value = np.nan
        else:
            self.hist["is_qp_feasible"].append(True)
            u = self.u.value
            prob_value = self.prob.value

        # Store data
        if self.cbf_method is not CBFMethod.NONE:
            self.hist["dr_h"].append(
                self.h_list[-1].value if self.h_list[-1].value else 0.0
            )
            if self.cbf_method is CBFMethod.HOCBF:
                self.hist["psi_r_cbf"].append(
                    self.psi_r_cbf.value if self.psi_r_cbf.value else 0.0
                )
            elif self.cbf_method is CBFMethod.TTCBF:  # TTCBF
                self.hist["psi_r_cbf"].append(
                    self.psi_r_cbf.value if self.psi_r_cbf.value else 0.0
                )
                self.hist["h_k_plus_r_pred"].append(
                    self.h_k_plus_r_pred.value if self.h_k_plus_r_pred.value else 0.0
                )

            self.hist["cost_cbf"].append(
                self.obj_cbf.value if (IS_USE_SLACK_CBF and self.obj_cbf.value) else 0.0
            )
            if IS_ADAPTIVE_TTCBF and self.cbf_method is CBFMethod.TTCBF:
                self.hist["lam_ttcbf"].append(self.lam_ttcbf.value)
            if IS_USE_SLACK_CBF and self.s_cbf.value >= 1e-2:
                print(f"[WARNING] Slack CBF is active: {self.s_cbf.value}")
            self.hist["h"].append(self.h_list[0])

        self.hist["u"].append(u)
        self.hist["cost"].append(self.obj.value if self.obj.value else 0.0)
        self.hist["cost_nom"].append(self.obj_nom.value if self.obj_nom.value else 0.0)

        return u, prob_value


# --------------------------------------------------------------------------- #
# Runge–Kutta-4 integrator
# --------------------------------------------------------------------------- #
def rk4_step(x: np.ndarray, u: float, h: float) -> np.ndarray:
    k1 = dynamics(x, u)
    k2 = dynamics(x + 0.5 * h * k1, u)
    k3 = dynamics(x + 0.5 * h * k2, u)
    k4 = dynamics(x + h * k3, u)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def save_trajectory(traj: SpringMassTrajectory, file_path: str | Path) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(traj, f, protocol=pickle.HIGHEST_PROTOCOL)
    cprint(f"\n[INFO] Trajectory saved to {file_path.resolve()}")


def load_trajectory(file_path: str | Path) -> SpringMassTrajectory:
    file_path = Path(file_path)
    with open(file_path, "rb") as f:
        traj: SpringMassTrajectory = pickle.load(f)
    print(f"[INFO] Trajectory loaded from {file_path.resolve()}")
    return traj


def hist_to_traj(hist: dict, cbf_method: CBFMethod) -> SpringMassTrajectory:
    # hist["t"] is a list of floats; hist["x"] is a list of np arrays (6,)
    t = np.asarray(hist["t"], dtype=float)
    x = np.vstack(hist["x"]).astype(float)
    u = np.asarray(hist["u"], dtype=float).reshape(-1)
    meta = {"cbf_method": cbf_method.name, "dt": dt}
    return SpringMassTrajectory(t=t, x=x, u=u, meta=meta)


def plot_results(fig_dir: Path, hist_list: list) -> None:
    """
    Plot the results: positions and control inputs
    """
    t_list = [np.vstack(hist.t) for hist in hist_list]
    pos_list = [np.vstack(hist.x) for hist in hist_list]
    u_list = [np.vstack(hist.u) for hist in hist_list]

    colors = [
        "#008B8B",
        "#848484",
        "#652884",
        "#E76253",
        "#375795",
        "#FED06E",
    ]
    line_styles = [
        "-.",
        "--",
        "-",
    ]
    labels_pos = [
        [r"$x_1$ (our)", r"$x_2$ (our)", r"$x_3$ (our)"],
        [r"$x_1$ (nominal)", r"$x_2$ (nominal)", r"$x_3$ (nominal)"],
    ]
    zorder = [10, 6]

    # --------------------------------------------------------------------- #
    # Visualisation: positions
    # --------------------------------------------------------------------- #
    plt.figure(figsize=(5, 3.8))
    lw = 2
    t_max = float(t_list[0][-1])

    for i in range(len(hist_list)):
        plt.plot(
            t_list[i],
            pos_list[i][:, 2] + x_equilibrium[2],
            color=colors[i],
            linestyle=line_styles[2],
            linewidth=lw,
            label=labels_pos[i][2],
            zorder=zorder[i],
        )
        plt.plot(
            t_list[i],
            pos_list[i][:, 1] + x_equilibrium[1],
            color=colors[i],
            linestyle=line_styles[1],
            linewidth=lw,
            label=labels_pos[i][1],
            zorder=zorder[i],
        )
        plt.plot(
            t_list[i],
            pos_list[i][:, 0] + x_equilibrium[0],
            color=colors[i],
            linestyle=line_styles[0],
            linewidth=lw,
            label=labels_pos[i][0],
            zorder=zorder[i],
        )
        if i == 1:
            x3_max_actual = pos_list[i][:, 2].max() + x_equilibrium[2]
            t3_max = t_list[i][np.argmax(pos_list[i][:, 2])]
            plt.annotate(
                rf"$x_3 = {x3_max_actual:.1f}$ m",
                xy=(t3_max, x3_max_actual),
                xytext=(t3_max + 1.2, x3_max_actual + 0.3),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.5),
                fontsize=10,
                color="k",
                ha="left",
                va="center",
                zorder=12,
            )

    plt.axhline(
        x3_ref + x_equilibrium[2],
        color="tab:blue",
        linestyle="--",
        linewidth=lw / 2,
        label=r"$x_{3,\mathrm{ref}}$",
        zorder=11,
    )
    plt.axhline(
        x3_max + x_equilibrium[2],
        color="r",
        linestyle="--",
        linewidth=lw / 2,
        label=r"$x_{3,\mathrm{safe}}$",
        zorder=11,
    )

    plt.fill_between(
        x=[0, t_max],
        y1=x3_ref + x_equilibrium[2] - 0.1,
        y2=x3_ref + x_equilibrium[2] + 0.1,
        color="tab:blue",
        alpha=0.2,
    )
    plt.fill_between(
        x=[0, t_max],
        y1=x3_max + x_equilibrium[2],
        y2=x3_max + x_equilibrium[2] + 2,
        color="tab:red",
        alpha=0.2,
    )

    plt.xlabel(r"Time $t$ [s]")
    plt.ylabel(r"Positions $x_1, x_2, x_3$ [m]")
    plt.xlim([0, t_max])
    plt.xticks(np.arange(0, t_max + 0.1, 2))
    plt.yticks(np.arange(-2, 5.1, 1))
    plt.ylim([-0.1, 4.5])
    plt.grid(True, linestyle="--", alpha=0.6, linewidth=0.5)
    plt.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.45),
        columnspacing=0.8,
        handletextpad=0.3,
        handlelength=1.8,
        frameon=True,
    )
    plt.tick_params(axis="both", direction="in")
    plt.tight_layout()

    fig_name = fig_dir / "fig_spring_mass_position.pdf"
    plt.savefig(fig_name, dpi=300, bbox_inches="tight", pad_inches=0)
    cprint(f"[INFO] Figure saved as {fig_name}")

    # --------------------------------------------------------------------- #
    # Visualisation: control input
    # --------------------------------------------------------------------- #
    plt.figure(figsize=(5, 2.5))
    labels_u = [
        r"$u$ (our)",
        r"$u$ (nominal)",
    ]
    line_styles_u = ["-", "--"]

    for i in range(len(hist_list)):
        plt.plot(
            t_list[i],
            u_list[i],
            color=colors[i],
            lw=lw,
            linestyle=line_styles_u[i],
            label=labels_u[i],
        )

    plt.axhline(
        5.0,
        color="r",
        lw=lw / 2,
        linestyle="--",
        label=r"$u_{\mathrm{min}/\mathrm{max}}$",
    )
    plt.axhline(-5.0, color="r", lw=lw / 2, linestyle="--")

    plt.fill_between(
        x=[0, t_max],
        y1=5.0,
        y2=10.0,
        color="tab:red",
        alpha=0.2,
    )
    plt.fill_between(
        x=[0, t_max],
        y1=-10.0,
        y2=-5.0,
        color="tab:red",
        alpha=0.2,
    )

    t_0 = 0.57
    y_0 = -0.66
    plt.annotate(
        rf"$t={t_0:.2f}$ s",
        xy=(t_0 - 0.1, y_0),
        xytext=(t_0 + 1.0, y_0 + 3),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        fontsize=10,
        color="black",
        ha="left",
        va="center",
    )

    t_1 = 0.6
    y_1 = -2.74
    plt.annotate(
        "TTCBF active",
        xy=(t_1 - 0.1, y_1),
        xytext=(t_1 + 2.5, y_1 - 1),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        fontsize=10,
        color="black",
        ha="left",
        va="center",
    )

    plt.xlabel(r"Time $t$ [s]")
    plt.ylabel(r"Control $u$ [N]")
    plt.xlim([0, t_max])
    plt.xticks(np.arange(0, t_max + 1, 2))
    plt.yticks(np.arange(-6, 7, 2))
    plt.ylim([-6, 6])
    plt.grid(True, linestyle="--", alpha=0.4, linewidth=0.5)
    plt.legend()
    plt.tick_params(axis="both", direction="in")
    plt.tight_layout()

    fig_name = fig_dir / "fig_spring_mass_force.pdf"
    plt.savefig(fig_name, dpi=300, bbox_inches="tight", pad_inches=0)
    cprint(f"[INFO] Figure saved as {fig_name}")

    # plt.show()
    
def export_video(
    traj: SpringMassTrajectory,
    video_path: str | Path,
    fps: int | None = None,
    dpi: int = 150,
    mass_size: float = 0.12,
    spring_lw: float = 2.0,
    traj_lw: float = 2.0,
    labels: list[str] | None = None,
    legend_title: str | None = None,
) -> None:
    """
    Animate the 1D spring-mass chain:
      - three masses as squares,
      - springs as lines connecting adjacent masses,
      - optional trailing position traces (with small y offsets).

    Parameters
    ----------
    traj : SpringMassTrajectory
        Loaded/simulated trajectory.
    video_path : str | Path
        Output .mp4 file.
    fps : int or None
        If None, use round(1/dt) where dt is inferred from traj.t.
    labels : list[str] or None
        Labels for masses, length 3. Default: ["$x_1$", "$x_2$", "$x_3$"].
    legend_title : str or None
        Optional text drawn immediately above the legend.
    """
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    t = traj.t
    x = traj.x
    u = traj.u
    n_frames = x.shape[0]

    # infer dt for video time label and fps fallback
    if len(t) >= 2:
        dt_sim = float(np.median(np.diff(t)))
    else:
        dt_sim = dt

    if fps is None:
        fps = int(np.round(1.0 / dt_sim))

    if labels is None:
        labels = [r"$x_1$", r"$x_2$", r"$x_3$"]
    if len(labels) != 3:
        raise ValueError(f"labels must have length 3, got {len(labels)}.")

    # positions (add equilibrium offsets for visualization consistency with your plots)
    x1 = x[:, 0] + x_equilibrium[0]
    x2 = x[:, 1] + x_equilibrium[1]
    x3 = x[:, 2] + x_equilibrium[2]
    
    # ---------------- force arrow (acts on mass 1) ----------------
    u = np.asarray(u, dtype=float).reshape(-1)
    u_min = float(np.min(u))
    u_max = float(np.max(u))
    u_abs_max = float(np.max(np.abs(u))) if u.size > 0 else 1.0
    if u_abs_max <= 1e-12:
        u_abs_max = 1.0  # avoid division by zero

    # arrow visual length (in x-axis units)
    arrow_len_max = 0.45  # tune: maximum displayed arrow length
    arrow_y = 0.16        # y location of arrow (above masses)


    # y offsets to show "trajectories" as separate rails
    # y_offsets = np.array([0.18, 0.0, -0.18], dtype=float)
    y_offsets = np.zeros(3, dtype=float)

    # axis limits
    x_min = float(np.min([x1.min(), x2.min(), x3.min()])) - 0.6
    x_max = float(np.max([x1.max(), x2.max(), x3.max()])) + 0.6

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.8), dpi=dpi, constrained_layout=True)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.25, 0.25)
    ax.set_yticks([])
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    ax.set_xlabel(r"Position [m]")

    colors = ["tab:blue", "tab:orange", "tab:green"]

    # reference / safety markers (optional but useful context)
    ref_line = ax.axvline(x3_ref + x_equilibrium[2], ls="--", lw=1.5, color=colors[2], label=r"$x_{3,\mathrm{ref}}$")
    safe_line = ax.axvline(x3_max + x_equilibrium[2], ls="--", lw=1.5, color="tab:red", label=r"$x_{3,\mathrm{safe}}$")

    # trajectory traces
    (trace1,) = ax.plot([], [], lw=traj_lw, color=colors[0], label=labels[0])
    (trace2,) = ax.plot([], [], lw=traj_lw, color=colors[1], label=labels[1])
    (trace3,) = ax.plot([], [], lw=traj_lw, color=colors[2], label=labels[2])

    # masses as squares (Rectangles)
    # Rectangle takes (x_left, y_bottom) + width/height
    mass_rects: list[Rectangle] = []
    for i, c in enumerate(colors):
        rect = Rectangle(
            (0.0, 0.0),
            width=mass_size,
            height=mass_size,
            facecolor="none",
            edgecolor=c,
            lw=2.0,
            zorder=5,
        )
        ax.add_patch(rect)
        mass_rects.append(rect)

    # springs as connecting lines (center-to-center)
    (spring12,) = ax.plot([], [], color="k", lw=spring_lw, zorder=4)
    (spring23,) = ax.plot([], [], color="k", lw=spring_lw, zorder=4)

    # time text (top-left)
    time_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        fontsize=11,
        va="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2"),
        zorder=10,
    )
    
    # force arrow (drawn with annotate so we can update start/end)
    force_annot = ax.annotate(
        "",
        xy=(0.0, arrow_y),
        xytext=(0.0, arrow_y),
        arrowprops=dict(arrowstyle="->", lw=2.0, color="k"),
        zorder=9,
    )
    force_text = ax.text(
        0.02,
        0.85,
        "",
        transform=ax.transAxes,
        fontsize=11,
        va="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2"),
        zorder=10,
    )


    # legend (bottom-right) for the mass traces
    # legend = ax.legend(loc="lower left", frameon=True, framealpha=1.0, ncol=2)
    dummy = Line2D([], [], linestyle="None", marker=None, alpha=0.0)

    handles = [trace1, trace2, trace3, ref_line, safe_line, dummy]
    labels_legend = [labels[0], labels[1], labels[2], ref_line.get_label(), safe_line.get_label(), ""]

    legend = ax.legend(
        handles,
        labels_legend,
        ncol=2,
        loc="lower left",
        frameon=True,
        framealpha=1.0,
        columnspacing=1.2,
        handletextpad=0.6,
        borderaxespad=0.3,
    )

    # optional text above legend (left-aligned with legend box)
    legend_title_text = None
    if legend_title is not None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox_disp = legend.get_window_extent(renderer=renderer)
        bbox_axes = bbox_disp.transformed(ax.transAxes.inverted())

        x_left = bbox_axes.x0
        y_top = bbox_axes.y1

        legend_title_text = ax.text(
            x_left,
            y_top + 0.02,
            legend_title,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2"),
            zorder=10,
        )


    def _set_mass_center(rect: Rectangle, cx: float, cy: float) -> None:
        rect.set_xy((cx - mass_size / 2.0, cy - mass_size / 2.0))

    def init():
        trace1.set_data([], [])
        trace2.set_data([], [])
        trace3.set_data([], [])

        # initialize masses at frame 0
        _set_mass_center(mass_rects[0], float(x1[0]), float(y_offsets[0]))
        _set_mass_center(mass_rects[1], float(x2[0]), float(y_offsets[1]))
        _set_mass_center(mass_rects[2], float(x3[0]), float(y_offsets[2]))

        # springs at frame 0 (connect actual mass centers, using their y positions)
        spring12.set_data([float(x1[0]), float(x2[0])], [float(y_offsets[0]), float(y_offsets[1])])
        spring23.set_data([float(x2[0]), float(x3[0])], [float(y_offsets[1]), float(y_offsets[2])])

        time_text.set_text("")
        
        # force arrow at frame 0
        u0 = float(u[0])
        du0 = arrow_len_max * (u0 / u_abs_max)  # signed
        x_start0 = float(x1[0])
        force_annot.set_position((x_start0, arrow_y))  # xytext
        force_annot.xy = (x_start0 + du0, arrow_y)     # xy (arrow head)
        force_text.set_text(rf"$u={u0:.2f}$")

        artists = [
            trace1, trace2, trace3,
            *mass_rects,
            spring12, spring23,
            time_text,
            force_annot,
            force_text,
        ]
        if legend_title_text is not None:
            artists.append(legend_title_text)
        return tuple(artists)

    def update(k: int):
        tail = 200  # number of samples to keep (tune)
        k0 = max(0, k - tail)

        trace1.set_data(x1[k0:k+1], np.full(k - k0 + 1, y_offsets[0]))
        trace2.set_data(x2[k0:k+1], np.full(k - k0 + 1, y_offsets[1]))
        trace3.set_data(x3[k0:k+1], np.full(k - k0 + 1, y_offsets[2]))

        # # traces (past positions on separate y rails)
        # trace1.set_data(x1[: k + 1], np.full(k + 1, y_offsets[0]))
        # trace2.set_data(x2[: k + 1], np.full(k + 1, y_offsets[1]))
        # trace3.set_data(x3[: k + 1], np.full(k + 1, y_offsets[2]))

        # masses at time k
        _set_mass_center(mass_rects[0], float(x1[k]), float(y_offsets[0]))
        _set_mass_center(mass_rects[1], float(x2[k]), float(y_offsets[1]))
        _set_mass_center(mass_rects[2], float(x3[k]), float(y_offsets[2]))

        # springs connect adjacent masses
        spring12.set_data([float(x1[k]), float(x2[k])], [float(y_offsets[0]), float(y_offsets[1])])
        spring23.set_data([float(x2[k]), float(x3[k])], [float(y_offsets[1]), float(y_offsets[2])])

        # time
        time_text.set_text(rf"$t = {float(t[k]):.2f}\,\mathrm{{s}}$")

        # force arrow at frame k (acts on mass 1)
        uk = float(u[k])
        duk = arrow_len_max * (uk / u_abs_max)  # signed, in [-arrow_len_max, +arrow_len_max]
        x_start = float(x1[k])

        force_annot.set_position((x_start, arrow_y))     # tail
        force_annot.xy = (x_start + duk, arrow_y)        # head
        force_text.set_text(rf"$u={uk:.2f}$")

        artists = [
            trace1, trace2, trace3,
            *mass_rects,
            spring12, spring23,
            time_text,
            force_annot,
            force_text,
        ]
        if legend_title_text is not None:
            artists.append(legend_title_text)
        return tuple(artists)

    frames_iter = tqdm(range(n_frames), total=n_frames, desc="Rendering video", unit="frame")

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frames_iter,
        interval=1000.0 / fps,
        blit=True,
    )

    writer = animation.FFMpegWriter(
        fps=fps,
        metadata=dict(artist="export_video"),
        bitrate=3000,
    )

    anim.save(str(video_path), writer=writer)
    plt.close(fig)
    cprint(f"[INFO] Video saved to {video_path.resolve()}")


# --------------------------------------------------------------------------- #
# Simulation loop
# --------------------------------------------------------------------------- #
def run_simulation(cbf_method: CBFMethod):
    n_steps = int(T_end / dt)

    # Initial state: springs at rest, all masses at zero, zero velocities
    x = np.array([0.0, 0.0, 0.0, 2.0, 1.0, 0.0])

    cbf_qp = CBF_QP(cbf_method)

    for k in range(n_steps):
        t = k * dt
        print(f"\r t={t:.2f}s / {T_end:.2f}s", end="", flush=True)
        u_nom = nominal_controller(x)

        cbf_qp.build(x, u_nom, k)
        u_opt, cost = cbf_qp.solve()

        x = rk4_step(x, u_opt, dt)

    return cbf_qp.hist
