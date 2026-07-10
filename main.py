#!/usr/bin/env python3
"""Reproduce the nonlinear obstacle-avoidance experiments for TTCBF.

The script compares TTCBF and adaptive TTCBF (aTTCBF) with ten baseline
controllers on the static disk-obstacle scenario from the accompanying
manuscript. It simulates the unicycle model, writes numerical logs as CSV/NPZ
files, and produces the manuscript-ready PDF figures.

Running ``python main.py`` evaluates the default scenario. Compatible
per-method results are reused from the output directory when available; stale
or missing results are simulated automatically. Run ``python main.py --help``
for method selection, scenario overrides, cache controls, and grid-sweep
options.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import itertools
import json
import math
import time
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.path import Path as MatplotlibPath
from matplotlib.transforms import Affine2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
import osqp
from scipy import optimize
from scipy import sparse

PLOT_DPI = 600
SINGLE_COLUMN_WIDTH = 3.4
PLOT_SIZE = (SINGLE_COLUMN_WIDTH, 2.4)
PLOT_TWO_PANEL_SIZE = (SINGLE_COLUMN_WIDTH, 1.6)
PLOT_TRAJECTORY_SIZE = (SINGLE_COLUMN_WIDTH, 3.0)
PLOT_TRAJECTORY_VIDEO_SIZE = (SINGLE_COLUMN_WIDTH, 2.1)
SAVEFIG_PAD_INCHES = 0.02
IS_ADD_ALL_METHOD_LEGEND = False
REFERENCE_LINEWIDTH = 1.0
SOLVER_FALLBACK_MARKER_SIZE = 20.0
COLLISION_OR_FINAL_MARKER_SIZE = 20.0
START_GOAL_MARKER_SIZE = 40.0
VIDEO_DPI = 180
VIDEO_FINAL_HOLD_SECONDS = 1.0
VIDEO_VEHICLE_MARKER_SIZE = 5.0
GOAL_REACHED_TOLERANCE = 0.1
COLLISION_H_TOLERANCE = 1.0e-6
DIRECT_QP_EPS_ABS = 1.0e-5
DIRECT_QP_EPS_REL = 1.0e-5
DIRECT_QP_MAX_ITER = 10000
DIRECT_QP_CONSTRAINT_TOL = 5.0e-4
NLP_FTOL = 1.0e-9
NLP_CONSTRAINT_TOL = 1.0e-7
NLP_DT_MAXITER = 100
NLP_ADT_MAXITER = 120

# Comment out entries here to remove metrics from fig_composite_results.pdf.
COMPOSITE_RESULT_METRICS = (
    "tuning",
    "runtime",
)

COMPOSITE_RESULT_METRIC_SPECS = {
    "tuning": {
        "label": "#Tun. para.",
        "color": "#008b8b",
    },
    "runtime": {
        "label": "Runtime [s]",
        "color": "#6a51a3",
    },
    "min_h": {
        "label": r"Minimum h [m$^2$]",
        "color": "tab:green",
    },
    "effort": {
        "label": "Control effort and smoothness",
        "color": "tab:red",
    },
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "legend.title_fontsize": 7,
        "figure.titlesize": 7,
        "text.usetex": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "savefig.pad_inches": SAVEFIG_PAD_INCHES,
        "savefig.dpi": PLOT_DPI,
    }
)


def new_paper_figure(
    figsize: tuple[float, float] = PLOT_SIZE,
    *subplots_args: object,
    constrained_layout: bool = True,
    **subplots_kwargs: object,
):
    return plt.subplots(
        *subplots_args,
        figsize=figsize,
        layout="constrained" if constrained_layout else None,
        **subplots_kwargs,
    )


def paper_figure_path(path: Path | str) -> Path:
    output = Path(path)
    if output.suffix.lower() != ".pdf":
        output = output.with_suffix(".pdf")
    if not output.name.startswith("fig_"):
        output = output.with_name(f"fig_{output.name}")
    return output


def save_paper_figure(fig: plt.Figure, path: Path | str) -> Path:
    output = paper_figure_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=PLOT_DPI,
        bbox_inches="tight",
        pad_inches=SAVEFIG_PAD_INCHES,
    )
    plt.close(fig)
    return output


def add_horizontal_reference(
    ax: plt.Axes,
    y: float,
    text: str,
    *,
    x: float = 1.0,
    va: str = "bottom",
    dy_points: float = 0.0,
) -> None:
    ax.axhline(
        y,
        color="black",
        linewidth=REFERENCE_LINEWIDTH,
        linestyle="--",
        label="_nolegend_",
    )
    ax.annotate(
        text,
        xy=(x, y),
        xycoords=ax.get_yaxis_transform(),
        xytext=(-2.0, dy_points),
        textcoords="offset points",
        ha="right",
        va=va,
        color="black",
        clip_on=False,
    )


def add_min_bound_touch_annotation(
    ax: plt.Axes,
    results: list[dict[str, np.ndarray | str | int | float]],
    *,
    control_index: int,
    bound: float,
    text: str,
    scenario: "Scenario",
    tolerance: float = 1.0e-6,
) -> None:
    touch: tuple[float, float] | None = None
    for result in results:
        controls = np.asarray(result["controls"], dtype=float)
        times = np.asarray(result["control_times"], dtype=float)
        n_values = min(controls.shape[0], times.size)
        if controls.ndim != 2 or controls.shape[1] <= control_index or n_values == 0:
            continue
        values = controls[:n_values, control_index]
        mask = np.isfinite(values) & np.isfinite(times[:n_values])
        mask &= values <= bound + tolerance
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            continue
        index = int(indices[0])
        candidate = (float(times[index]), float(values[index]))
        if touch is None or candidate[0] < touch[0]:
            touch = candidate

    if touch is None:
        return

    y_span = max(abs(ax.get_ylim()[1] - ax.get_ylim()[0]), 1.0e-12)
    ax.annotate(
        text,
        xy=touch,
        xytext=(
            min(touch[0] + 0.12 * scenario.horizon, scenario.horizon),
            touch[1] + 0.18 * y_span,
        ),
        textcoords="data",
        ha="left",
        va="bottom",
        color="black",
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.0,
            "color": "black",
            "relpos": (0.0, 0.5),
        },
    )


def add_series_legend(
    ax: plt.Axes,
    *,
    loc: str = "best",
    outside_threshold: int = 7,
    **legend_kwargs: object,
):
    handles, labels = ax.get_legend_handles_labels()
    filtered = [
        (handle, label)
        for handle, label in zip(handles, labels, strict=True)
        if label and not label.startswith("_")
    ]
    if not filtered:
        return None
    handles, labels = zip(*filtered, strict=True)
    return add_method_legend(
        ax,
        handles,
        labels,
        loc=loc,
        outside_threshold=outside_threshold,
        **legend_kwargs,
    )


def add_method_legend(
    ax: plt.Axes,
    handles: tuple[object, ...] | list[object],
    labels: tuple[str, ...] | list[str],
    *,
    loc: str = "best",
    outside_threshold: int = 7,
    **legend_kwargs: object,
):
    if not handles:
        return None
    if len(labels) >= outside_threshold:
        return ax.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=legend_column_count(labels),
            borderaxespad=0.0,
            columnspacing=0.8,
            handlelength=1.8,
            **legend_kwargs,
        )
    return ax.legend(handles, labels, loc=loc, **legend_kwargs)


def legend_column_count(labels: tuple[str, ...] | list[str]) -> int:
    if len(labels) < 7:
        return max(1, math.ceil(len(labels) / 2))
    return max(1, math.ceil(len(labels) / 4))

@dataclass(frozen=True)
class Scenario:
    dt: float = 0.05
    horizon: float = 14.0
    start: tuple[float, float, float, float] = (0.0, -1.5, 0.0, 1.0)
    goal: tuple[float, float] = (10.0, 0.0)
    obstacle: tuple[float, float] = (5.0, 0.0)
    obstacle_radius: float = 1.0
    vehicle_radius: float = 0.2
    omega_min: float = -2.0
    omega_max: float = 2.0
    accel_min: float = -1.0
    accel_max: float = 1.0
    v_min: float = 0.0
    v_max: float = 2.0
    cruise_speed: float = 1.0
    k_heading: float = 2.0
    k_speed: float = 1.5
    clf_lambda_heading: float = 3.0
    clf_lambda_speed: float = 2.5
    slack_weight: float = 250.0
    attcbf_eta_weight: float = 10000.0
    omega_rate_max: float = 20.0
    accel_rate_max: float = 6.0
    zoh_tlc_tau: float = 0.25
    event_triggered_tlc_tau: float = 0.25
    rtlc_tau: float = 0.25
    event_triggered_tlc_state_box_widths: tuple[float, float, float, float] = (0.02, 0.02, 0.02, 0.02)
    event_triggered_tlc_samples_per_dim: int = 4
    event_triggered_atlc_tau_candidates: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00)
    event_triggered_atlc_lookahead: float = 1.0
    ttcbf_alpha: float = 0.15
    pacbf_p1_0: float = 0.1
    pacbf_p1_des: float = 0.1
    pacbf_p2_des: float = 1.0
    pacbf_p1_clf_rate: float = 10.0
    pacbf_nu_weight: float = 0.02
    pacbf_nu_regularization: float = 1.0e-4
    pacbf_p1_clf_weight: float = 25.0
    pacbf_p2_weight: float = 10.0
    pacbf_epsilon: float = 1.0e-10
    racbf_k1: float = 1.0
    racbf_k2: float = 2.0
    racbf_l1: float = 1.0
    racbf_l2: float = 1.0
    racbf_r0: float = 0.2
    racbf_rdot0: float = 0.0
    racbf_r_des: float = 0.02
    racbf_r_min: float = 1.0e-6
    racbf_clf_rate: float = 10.0
    racbf_aux_weight: float = 1.0
    racbf_clf_weight: float = 25.0
    avcbf_k1: float = 1.0
    avcbf_k2: float = 1.0
    avcbf_l1: float = 0.1
    avcbf_l2: float = 0.1
    avcbf_a0: float = 1.0
    avcbf_adot0: float = 1.0
    avcbf_aux_input_des: float = 1.0
    avcbf_aux_weight: float = 1000.0
    avcbf_epsilon: float = 1.0e-10
    ct_hocbf_p1: float = 1.0
    ct_hocbf_p2: float = 6.0
    dt_hocbf_gamma1: float = 0.2
    dt_hocbf_gamma2: float = 0.1
    adt_hocbf_gamma1_0: float = 0.20
    adt_hocbf_gamma1_des: float = 0.20
    adt_hocbf_gamma2_des: float = 0.10
    adt_hocbf_gamma_min: float = 1.0e-4
    adt_hocbf_gamma_max: float = 0.25
    adt_hocbf_beta1: float = 0.2
    adt_hocbf_beta2: float = 0.5
    adt_hocbf_gamma_clf_rate: float = 1.0
    adt_hocbf_aux_weight: float = 1.0
    adt_hocbf_gamma_clf_weight: float = 1.0
    adt_hocbf_gamma2_weight: float = 1.0

    @property
    def safe_radius(self) -> float:
        return self.obstacle_radius + self.vehicle_radius

    @property
    def u_min(self) -> np.ndarray:
        return np.array([self.omega_min, self.accel_min], dtype=float)

    @property
    def u_max(self) -> np.ndarray:
        return np.array([self.omega_max, self.accel_max], dtype=float)

    @property
    def u_rate_max(self) -> np.ndarray:
        return np.array([self.omega_rate_max, self.accel_rate_max], dtype=float)


@dataclass
class QPResult:
    u: np.ndarray
    feasible: bool
    status: str
    eta: float = np.nan
    aux: float = np.nan
    solver_state: np.ndarray | None = None


@dataclass
class SafetyConstraint:
    A: np.ndarray
    b: float
    eta_coeff: float = 0.0


@dataclass
class TimingAccumulator:
    qp_calls: int = 0
    qp_wall_time: float = 0.0
    qp_statuses: list[str] = field(default_factory=list)


def canonical_qp_status(status: str) -> str:
    normalized = str(status).strip()
    lower = normalized.lower()
    if lower in {"optimal", "optimal_inaccurate", "infeasible", "infeasible_inaccurate", "unbounded"}:
        return lower
    if lower == "all_tau_infeasible":
        return "infeasible"
    if lower == "not_solved":
        return "not_solved"
    if lower == "held":
        return "held"
    if lower.startswith(("osqp_error:", "clarabel_error:", "ecos_error:", "scs_error:")):
        return "solver_error"
    if "optimization terminated successfully" in lower:
        return "optimal"
    if "infeasible" in lower:
        return "infeasible"
    return "other"


def is_nominal_qp_status(status: str) -> bool:
    return canonical_qp_status(status) == "optimal"


def record_qp_status(timing: TimingAccumulator | None, status: str) -> None:
    if timing is not None:
        timing.qp_statuses.append(str(status))


def status_counts(values: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for raw_status in np.asarray(values).astype(str):
        status = canonical_qp_status(raw_status)
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def status_counts_json(values: np.ndarray) -> str:
    return json.dumps(status_counts(values), sort_keys=True, separators=(",", ":"))

@dataclass(frozen=True)
class RelativeDegreeTwoHOCBF:
    """Fixed-gain relative-degree-two HOCBF with linear class-K functions.

    For a time-invariant barrier h(x) >= 0, use

        psi_1 = L_f h + p_1 h,
        psi_2 = L_f^2 h + L_g L_f h u + p_2 psi_1.

    The safety filter enforces psi_2 >= 0. This is Eq. (18) specialized to
    m = 2, alpha_1(s) = p_1 s, and alpha_2(s) = p_2 s.
    """

    p1: float
    p2: float

    def __post_init__(self) -> None:
        if self.p1 <= 0.0 or self.p2 <= 0.0:
            raise ValueError("HOCBF gains p1 and p2 must be positive")

    def affine_constraint(
        self,
        h: float,
        lf_h: float,
        lf2_h: float,
        lg_lf_h: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Return A, b for the affine control constraint A @ u + b >= 0."""
        A = np.asarray(lg_lf_h, dtype=float)
        psi1 = float(lf_h + self.p1 * h)
        b = float(lf2_h + self.p2 * psi1)
        return A, b


def relative_degree_two_hocbf_constraint(
    h: float,
    lf_h: float,
    lf2_h: float,
    lg_lf_h: np.ndarray,
    p1: float,
    p2: float,
) -> tuple[np.ndarray, float]:
    """Convenience wrapper returning A, b for A @ u + b >= 0."""
    return RelativeDegreeTwoHOCBF(p1=p1, p2=p2).affine_constraint(
        h=h,
        lf_h=lf_h,
        lf2_h=lf2_h,
        lg_lf_h=lg_lf_h,
    )


METHODS = (
    "ttcbf",
    "attcbf",
    "dt_hocbf",
    "adt_hocbf",
    "ct_hocbf",
    "pacbf",
    "racbf",
    "avcbf",
    "tlc",
    "rtlc",
    "event_triggered_tlc",
    "event_triggered_atlc",
)

ADAPTIVE_GRID_SWEEP_METHODS = (
    "attcbf",
    "adt_hocbf",
    "pacbf",
    "racbf",
    "avcbf",
    "event_triggered_atlc",
)

METHOD_LABELS = {
    "ttcbf": "TTCBF (our)",
    "attcbf": "aTTCBF (our)",
    "tlc": "ZOH-TLC",
    "event_triggered_tlc": "ET-TLC",
    "rtlc": "rTLC",
    "event_triggered_atlc": "ET-aTLC",
    "pacbf": "PACBF",
    "racbf": "RACBF",
    "avcbf": "AVCBF",
    "ct_hocbf": "CT-HOCBF",
    "dt_hocbf": "DT-HOCBF",
    "adt_hocbf": "aDT-HOCBF",
}

METHOD_COLORS = {
    "ttcbf": "#1f76b488",
    "attcbf": "#1f77b4",
    "ct_hocbf": "#6a51a388",
    "pacbf": "#6a51a3",
    "racbf": "#9573e4",
    "avcbf": "#d95fbc",
    "dt_hocbf": "#2ca02c88",
    "adt_hocbf": "#2ca02c",
    "tlc": "#c36c02ff",
    "rtlc": "#ff8c00",
    "event_triggered_tlc": "#ce001888",
    "event_triggered_atlc": "#ce0018",
}

METHOD_LINESTYLES = {
    "ttcbf": "-",
    "attcbf": "--",
    "ct_hocbf": "-",
    "pacbf": "--",
    "racbf": "-.",
    "avcbf": ":",
    "dt_hocbf": "-",
    "adt_hocbf": "--",
    "tlc": "-",
    "rtlc": "--",
    "event_triggered_tlc": "-.",
    "event_triggered_atlc": ":",
}

METHOD_LINEWIDTHS = {
    "ttcbf": 1.2,
    "attcbf": 1.2,
}

METHOD_ZORDERS = {
    "attcbf": 20,
    "ttcbf": 21,
}

PLOT_GROUPS = ((None, METHODS),)

CACHE_VERSION = 10
METHOD_CACHE_DIRNAME = "method_cache"
TTCBF_METHODS = ("ttcbf", "attcbf")
METHOD_PARAMETER_FIELDS = {
    "tlc": ("zoh_tlc_tau",),
    "event_triggered_tlc": (
        "event_triggered_tlc_tau",
        "event_triggered_tlc_state_box_widths",
        "event_triggered_tlc_samples_per_dim",
    ),
    "rtlc": ("rtlc_tau", "omega_rate_max", "accel_rate_max"),
    "event_triggered_atlc": (
        "event_triggered_atlc_tau_candidates",
        "event_triggered_atlc_lookahead",
        "event_triggered_tlc_state_box_widths",
        "event_triggered_tlc_samples_per_dim",
    ),
    "ttcbf": ("ttcbf_alpha", "omega_rate_max", "accel_rate_max"),
    "attcbf": ("attcbf_eta_weight", "omega_rate_max", "accel_rate_max"),
    "pacbf": (
        "pacbf_p1_0",
        "pacbf_p1_des",
        "pacbf_p2_des",
        "pacbf_p1_clf_rate",
        "pacbf_nu_weight",
        "pacbf_nu_regularization",
        "pacbf_p1_clf_weight",
        "pacbf_p2_weight",
        "pacbf_epsilon",
    ),
    "racbf": (
        "racbf_k1",
        "racbf_k2",
        "racbf_l1",
        "racbf_l2",
        "racbf_r0",
        "racbf_rdot0",
        "racbf_r_des",
        "racbf_r_min",
        "racbf_clf_rate",
        "racbf_aux_weight",
        "racbf_clf_weight",
    ),
    "avcbf": (
        "avcbf_k1",
        "avcbf_k2",
        "avcbf_l1",
        "avcbf_l2",
        "avcbf_a0",
        "avcbf_adot0",
        "avcbf_aux_input_des",
        "avcbf_aux_weight",
        "avcbf_epsilon",
    ),
    "ct_hocbf": ("ct_hocbf_p1", "ct_hocbf_p2"),
    "dt_hocbf": ("dt_hocbf_gamma1", "dt_hocbf_gamma2"),
    "adt_hocbf": (
        "adt_hocbf_gamma1_0",
        "adt_hocbf_gamma1_des",
        "adt_hocbf_gamma2_des",
        "adt_hocbf_gamma_min",
        "adt_hocbf_gamma_max",
        "adt_hocbf_beta1",
        "adt_hocbf_beta2",
        "adt_hocbf_gamma_clf_rate",
        "adt_hocbf_aux_weight",
        "adt_hocbf_gamma_clf_weight",
        "adt_hocbf_gamma2_weight",
    ),
}
METHOD_PARAMETER_FIELD_SET = {
    field_name
    for method_fields in METHOD_PARAMETER_FIELDS.values()
    for field_name in method_fields
}
TUNING_PARAMETER_FIELDS = {
    "tlc": ("zoh_tlc_tau",),
    "event_triggered_tlc": (
        "event_triggered_tlc_tau",
        "event_triggered_tlc_state_box_widths",
        "event_triggered_tlc_samples_per_dim",
    ),
    "rtlc": ("rtlc_tau",),
    "event_triggered_atlc": (
        "event_triggered_atlc_tau_candidates",
        "event_triggered_atlc_lookahead",
        "event_triggered_tlc_state_box_widths",
        "event_triggered_tlc_samples_per_dim",
    ),
    "ttcbf": ("ttcbf_alpha",),
    "attcbf": ("attcbf_eta_weight",),
    "pacbf": (
        "pacbf_p1_0",
        "pacbf_p1_des",
        "pacbf_p2_des",
        "pacbf_p1_clf_rate",
        "pacbf_nu_weight",
        "pacbf_nu_regularization",
        "pacbf_p1_clf_weight",
        "pacbf_p2_weight",
        "pacbf_epsilon",
    ),
    "racbf": (
        "racbf_k1",
        "racbf_k2",
        "racbf_l1",
        "racbf_l2",
        "racbf_r0",
        "racbf_rdot0",
        "racbf_r_des",
        "racbf_r_min",
        "racbf_clf_rate",
        "racbf_aux_weight",
        "racbf_clf_weight",
    ),
    "avcbf": (
        "avcbf_k1",
        "avcbf_k2",
        "avcbf_l1",
        "avcbf_l2",
        "avcbf_a0",
        "avcbf_adot0",
        "avcbf_aux_input_des",
        "avcbf_aux_weight",
        "avcbf_epsilon",
    ),
    "ct_hocbf": ("ct_hocbf_p1", "ct_hocbf_p2"),
    "dt_hocbf": ("dt_hocbf_gamma1", "dt_hocbf_gamma2"),
    "adt_hocbf": (
        "adt_hocbf_gamma1_0",
        "adt_hocbf_gamma1_des",
        "adt_hocbf_gamma2_des",
        "adt_hocbf_gamma_min",
        "adt_hocbf_gamma_max",
        "adt_hocbf_beta1",
        "adt_hocbf_beta2",
        "adt_hocbf_gamma_clf_rate",
        "adt_hocbf_aux_weight",
        "adt_hocbf_gamma_clf_weight",
        "adt_hocbf_gamma2_weight",
    ),
}


def normalize_json_value(value: object) -> object:
    if isinstance(value, tuple):
        return [normalize_json_value(item) for item in value]
    if isinstance(value, list):
        return [normalize_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def scenario_values(
    scenario: Scenario,
    names: set[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    selected = set(names) if names is not None else None
    values = {}
    for field in fields(Scenario):
        if selected is not None and field.name not in selected:
            continue
        values[field.name] = normalize_json_value(getattr(scenario, field.name))
    return values


def stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def shared_scenario_hash(scenario: Scenario) -> str:
    shared_names = {
        field.name
        for field in fields(Scenario)
        if field.name not in METHOD_PARAMETER_FIELD_SET
    }
    return stable_hash(scenario_values(scenario, shared_names))


def method_parameter_hash(method: str, scenario: Scenario) -> str:
    return stable_hash(scenario_values(scenario, METHOD_PARAMETER_FIELDS[method]))


def parse_methods_argument(raw_methods: str) -> tuple[str, ...]:
    methods = tuple(method.strip() for method in raw_methods.split(",") if method.strip())
    invalid = [method for method in methods if method not in METHODS]
    if invalid:
        raise ValueError(f"unknown method(s): {', '.join(invalid)}")
    if not methods:
        raise ValueError("at least one method must be provided")
    return methods


def parse_tuple_override(
    raw_value: str,
    current_value: tuple[object, ...],
    allow_variable_length: bool = False,
) -> tuple[object, ...]:
    text = raw_value.strip()
    if text.startswith(("(", "[")):
        parsed = ast.literal_eval(text)
        if not isinstance(parsed, (list, tuple)):
            raise ValueError("tuple fields require a tuple/list value")
        raw_items = list(parsed)
    else:
        raw_items = [item.strip() for item in text.split(",") if item.strip()]
    if not allow_variable_length and len(current_value) != 0 and len(raw_items) != len(current_value):
        raise ValueError(f"expected {len(current_value)} values")
    item_type = type(current_value[0]) if current_value else float
    return tuple(item_type(item) for item in raw_items)


def parse_scenario_override_value(name: str, raw_value: str, current_value: object) -> object:
    if isinstance(current_value, tuple):
        return parse_tuple_override(
            raw_value,
            current_value,
            allow_variable_length=(name == "event_triggered_atlc_tau_candidates"),
        )
    if isinstance(current_value, bool):
        normalized = raw_value.strip().lower()
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off"):
            return False
        raise ValueError("expected a boolean value")
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(raw_value)
    if isinstance(current_value, float):
        return float(raw_value)
    return raw_value


def apply_scenario_overrides(
    scenario: Scenario,
    overrides: list[str],
) -> Scenario:
    field_names = {field.name for field in fields(Scenario)}
    updates: dict[str, object] = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"invalid override {override!r}; expected name=value")
        name, raw_value = override.split("=", 1)
        name = name.strip()
        if name not in field_names:
            raise ValueError(f"unknown Scenario field: {name}")
        updates[name] = parse_scenario_override_value(name, raw_value, getattr(scenario, name))
    return replace(scenario, **updates)


def format_float_for_path(value: float) -> str:
    text = f"{value:.12g}"
    return "0" if text == "-0" else text


def format_accel_min_for_path(accel_min: float) -> str:
    return format_float_for_path(accel_min)


def default_output_dir(scenario: Scenario) -> Path:
    return Path(f"eval_results_accel_min{format_accel_min_for_path(scenario.accel_min)}")


def default_grid_sweep_output_dir(
    accel_min_end: float,
    omega_min_end: float,
    omega_sweep_fields: tuple[str, ...],
    *,
    is_auto_sweep: bool,
) -> Path:
    if len(omega_sweep_fields) == 1:
        omega_part = omega_sweep_fields[0]
    else:
        omega_part = "omega_both"
    mode_part = "auto" if is_auto_sweep else "manual"
    return Path(
        "eval_static_obstacle_grid_"
        f"accel_min{format_float_for_path(accel_min_end)}_"
        f"{omega_part}{format_float_for_path(abs(omega_min_end))}_"
        f"{mode_part}"
    )


def method_cache_path(cache_dir: Path, method: str) -> Path:
    return cache_dir / f"{method}.npz"


def method_cache_metadata(
    method: str,
    scenario: Scenario,
    result: dict[str, np.ndarray | str | int | float] | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "cache_version": CACHE_VERSION,
        "method": method,
        "label": METHOD_LABELS[method],
        "shared_scenario_hash": shared_scenario_hash(scenario),
        "method_parameter_hash": method_parameter_hash(method, scenario),
        "shared_scenario": scenario_values(
            scenario,
            {
                field.name
                for field in fields(Scenario)
                if field.name not in METHOD_PARAMETER_FIELD_SET
            },
        ),
        "method_parameters": scenario_values(scenario, METHOD_PARAMETER_FIELDS[method]),
    }
    if result is not None:
        metadata["result_scalars"] = {
            key: normalize_json_value(value)
            for key, value in result.items()
            if not isinstance(value, np.ndarray)
        }
    return metadata


def save_method_cache(
    result: dict[str, np.ndarray | str | int | float],
    scenario: Scenario,
    cache_dir: Path,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    method = str(result["method"])
    metadata = method_cache_metadata(method, scenario, result)
    arrays = {
        key: value
        for key, value in result.items()
        if isinstance(value, np.ndarray)
    }
    np.savez_compressed(
        method_cache_path(cache_dir, method),
        __metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        **arrays,
    )


def load_method_cache(
    method: str,
    scenario: Scenario,
    cache_dir: Path,
) -> tuple[dict[str, np.ndarray | str | int | float] | None, str]:
    cache_file = method_cache_path(cache_dir, method)
    if not cache_file.exists():
        return None, "missing cache"
    expected = method_cache_metadata(method, scenario)
    try:
        with np.load(cache_file, allow_pickle=False) as data:
            if "__metadata_json" not in data.files:
                return None, "cache metadata missing"
            metadata = json.loads(str(data["__metadata_json"].item()))
            if metadata.get("cache_version") != CACHE_VERSION:
                return None, "cache version changed"
            if metadata.get("method") != method:
                return None, "cache method mismatch"
            if metadata.get("shared_scenario_hash") != expected["shared_scenario_hash"]:
                return None, "shared scenario changed"
            if metadata.get("method_parameter_hash") != expected["method_parameter_hash"]:
                return None, "method parameters changed"
            scalars = metadata.get("result_scalars")
            if not isinstance(scalars, dict):
                return None, "cached result scalars missing"
            result: dict[str, np.ndarray | str | int | float] = dict(scalars)
            result["method"] = method
            result["label"] = METHOD_LABELS[method]
            for key in data.files:
                if key == "__metadata_json":
                    continue
                result[key] = np.asarray(data[key])
            return result, "loaded"
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, f"cache read failed: {exc}"


def method_color(result: dict[str, np.ndarray | str | int | float]) -> str:
    return METHOD_COLORS.get(str(result["method"]), "0.2")


def method_linestyle(result: dict[str, np.ndarray | str | int | float]) -> str:
    return METHOD_LINESTYLES.get(str(result["method"]), "-")


def method_linewidth(result: dict[str, np.ndarray | str | int | float]) -> float:
    return METHOD_LINEWIDTHS.get(str(result["method"]), 1.0)


def method_zorder(result: dict[str, np.ndarray | str | int | float]) -> int:
    return METHOD_ZORDERS.get(str(result["method"]), 2)


def method_plot_kwargs(
    result: dict[str, np.ndarray | str | int | float],
    **extra: object,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "color": method_color(result),
        "linestyle": method_linestyle(result),
        "linewidth": method_linewidth(result),
        "zorder": method_zorder(result),
    }
    kwargs.update(extra)
    return kwargs


def is_all_methods_result_set(
    results: list[dict[str, np.ndarray | str | int | float]],
) -> bool:
    return tuple(str(result["method"]) for result in results) == METHODS


def should_add_method_legend(
    results: list[dict[str, np.ndarray | str | int | float]],
) -> bool:
    return IS_ADD_ALL_METHOD_LEGEND or not is_all_methods_result_set(results)


def save_method_legend_figure(
    methods: tuple[str, ...],
    labels: list[str],
    out_dir: Path,
) -> None:
    handles = [
        Line2D([0.0], [0.0], label=label, **method_plot_kwargs({"method": method}))
        for method, label in zip(methods, labels, strict=True)
    ]
    ncol = legend_column_count(labels)
    nrows = max(1, math.ceil(len(labels) / ncol))
    fig = plt.figure(figsize=(SINGLE_COLUMN_WIDTH, 0.20 + 0.22 * nrows))
    fig.legend(
        handles=handles,
        labels=labels,
        loc="center",
        ncol=ncol,
        borderaxespad=0.0,
        columnspacing=0.8,
        handlelength=1.8,
    )
    save_paper_figure(fig, out_dir / "fig_all_methods_legend.pdf")


def save_all_methods_legend(
    results: list[dict[str, np.ndarray | str | int | float]],
    scenario: Scenario,
    out_dir: Path,
) -> None:
    if IS_ADD_ALL_METHOD_LEGEND or not is_all_methods_result_set(results):
        return

    labels = [result_legend_label(result, scenario) for result in results]
    save_method_legend_figure(
        tuple(str(result["method"]) for result in results),
        labels,
        out_dir,
    )


def result_stop_time(result: dict[str, np.ndarray | str | int | float]) -> float:
    raw_stop_time = result.get("stop_time")
    if raw_stop_time is not None:
        try:
            stop_time = float(raw_stop_time)
        except (TypeError, ValueError):
            stop_time = math.nan
        if math.isfinite(stop_time):
            return stop_time

    times = np.asarray(result.get("times", []), dtype=float)
    finite_times = times[np.isfinite(times)]
    if finite_times.size:
        return float(finite_times[-1])
    return 0.0


def result_duration(result: dict[str, np.ndarray | str | int | float]) -> float:
    return max(0.0, result_stop_time(result))


def result_stop_reason(result: dict[str, np.ndarray | str | int | float]) -> str:
    return str(result.get("stop_reason", "horizon"))


def final_goal_distance(result: dict[str, np.ndarray | str | int | float]) -> float:
    goal_distance = np.asarray(result.get("goal_distance", []), dtype=float)
    finite_goal_distance = goal_distance[np.isfinite(goal_distance)]
    if finite_goal_distance.size == 0:
        return math.nan
    return float(finite_goal_distance[-1])


def column_mean_or_nan(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return math.nan
    return float(np.mean(finite))


def column_max_abs_or_nan(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return math.nan
    return float(np.max(np.abs(finite)))


def reached_goal(
    result: dict[str, np.ndarray | str | int | float],
    tolerance: float = GOAL_REACHED_TOLERANCE,
) -> bool:
    goal_distance = np.asarray(result["goal_distance"], dtype=float)
    if goal_distance.size == 0:
        return False
    finite_goal_distance = goal_distance[np.isfinite(goal_distance)]
    if finite_goal_distance.size == 0:
        return False
    return bool(np.min(finite_goal_distance) <= tolerance)


def task_completion_time(
    result: dict[str, np.ndarray | str | int | float],
    tolerance: float = GOAL_REACHED_TOLERANCE,
) -> float:
    goal_distance = np.asarray(result["goal_distance"], dtype=float)
    times = np.asarray(result["times"], dtype=float)
    if goal_distance.size == 0 or times.size == 0:
        return math.nan

    n_values = min(goal_distance.size, times.size)
    reached_indices = np.flatnonzero(
        np.isfinite(goal_distance[:n_values])
        & (goal_distance[:n_values] <= tolerance)
    )
    if reached_indices.size == 0:
        return math.nan
    return float(times[int(reached_indices[0])])


def positions_outside_goal_tolerance(
    positions: np.ndarray,
    scenario: Scenario,
    tolerance: float = GOAL_REACHED_TOLERANCE,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    if positions.size == 0:
        return np.zeros(positions.shape[0], dtype=bool)
    goal = np.asarray(scenario.goal, dtype=float)
    return np.linalg.norm(positions - goal, axis=1) > tolerance


def fallback_marker_positions(
    result: dict[str, np.ndarray | str | int | float],
    scenario: Scenario,
) -> np.ndarray:
    fixed_interval = 2
    states = np.asarray(result["states"], dtype=float)
    feasible = np.asarray(result.get("feasible", []), dtype=bool)
    event_flags = np.asarray(result.get("event_flags", []), dtype=bool)
    if states.shape[0] < 2 or feasible.size == 0 or event_flags.size == 0:
        return np.empty((0, 2), dtype=float)

    event_count = min(states.shape[0] - 1, feasible.size, event_flags.size)
    positions = states[:event_count, :2]
    fallback_mask = event_flags[:event_count] & ~feasible[:event_count]
    fallback_mask &= positions_outside_goal_tolerance(positions, scenario)
    fallback_indices = np.flatnonzero(fallback_mask)
    if fallback_indices.size == 0:
        return np.empty((0, 2), dtype=float)

    selected_indices = []
    run_start = 0
    interval = max(1, int(fixed_interval))
    while run_start < fallback_indices.size:
        run_end = run_start + 1
        while (
            run_end < fallback_indices.size
            and fallback_indices[run_end] == fallback_indices[run_end - 1] + 1
        ):
            run_end += 1

        run_indices = fallback_indices[run_start:run_end]
        if run_indices.size > 6:
            selected_indices.extend(run_indices[::interval])
        else:
            selected_indices.extend(run_indices)
        run_start = run_end

    return positions[np.asarray(selected_indices, dtype=int)]


def segment_circle_intersection(
    p0: np.ndarray,
    p1: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> np.ndarray | None:
    direction = p1 - p0
    a = float(direction @ direction)
    if a <= 0.0:
        return None

    offset = p0 - center
    b = 2.0 * float(offset @ direction)
    c = float(offset @ offset - radius * radius)
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None

    root = math.sqrt(max(discriminant, 0.0))
    candidates = []
    for sign in (-1.0, 1.0):
        t = (-b + sign * root) / (2.0 * a)
        if -1.0e-9 <= t <= 1.0 + 1.0e-9:
            candidates.append(min(max(t, 0.0), 1.0))
    if not candidates:
        return None
    return p0 + min(candidates) * direction


def first_collision_point(
    result: dict[str, np.ndarray | str | int | float],
    scenario: Scenario,
) -> np.ndarray | None:
    states = np.asarray(result["states"], dtype=float)
    if states.shape[0] < 2:
        return None

    positions = states[:, :2]
    center = np.asarray(scenario.obstacle, dtype=float)
    h_values = np.sum((positions - center) ** 2, axis=1) - scenario.safe_radius**2
    inside_indices = np.flatnonzero(h_values < -COLLISION_H_TOLERANCE)
    if inside_indices.size == 0:
        return None

    first_inside = int(inside_indices[0])
    if first_inside == 0:
        return positions[0].copy()

    outside_indices = np.flatnonzero(h_values[:first_inside] > COLLISION_H_TOLERANCE)
    if outside_indices.size == 0:
        return positions[first_inside].copy()

    last_outside = int(outside_indices[-1])
    crossing = segment_circle_intersection(
        positions[last_outside],
        positions[first_inside],
        center,
        scenario.safe_radius,
    )
    if crossing is not None:
        return crossing

    h0 = h_values[last_outside]
    h1 = h_values[first_inside]
    if h0 > h1:
        t = min(max(h0 / (h0 - h1), 0.0), 1.0)
        return positions[last_outside] + t * (positions[first_inside] - positions[last_outside])
    return positions[first_inside].copy()


def result_failed(
    result: dict[str, np.ndarray | str | int | float],
    scenario: Scenario,
) -> bool:
    return not reached_goal(result) or first_collision_point(result, scenario) is not None


def result_legend_label(
    result: dict[str, np.ndarray | str | int | float],
    scenario: Scenario,
) -> str:
    label = str(result["label"])
    if result_failed(result, scenario):
        return f"{label} [failed]"
    return label


def draw_trajectory_status_markers(
    ax: plt.Axes,
    result: dict[str, np.ndarray | str | int | float],
    scenario: Scenario,
) -> None:
    color = method_color(result)
    states = np.asarray(result["states"], dtype=float)
    if states.size == 0:
        return

    fallback_positions = fallback_marker_positions(result, scenario)
    if fallback_positions.size:
        ax.scatter(
            fallback_positions[:, 0],
            fallback_positions[:, 1],
            marker="s",
            s=SOLVER_FALLBACK_MARKER_SIZE,
            facecolors="none",
            edgecolors=color,
            linewidths=1.0,
            alpha=0.8,
            zorder=29,
            label="_nolegend_",
        )

    if result_stop_reason(result) == "solver_infeasible":
        final_position = states[-1, :2]
        ax.scatter(
            [final_position[0]],
            [final_position[1]],
            marker="s",
            s=SOLVER_FALLBACK_MARKER_SIZE,
            facecolors="none",
            edgecolors=color,
            linewidths=1.0,
            alpha=0.9,
            zorder=33,
            label="_nolegend_",
        )

    crossing = first_collision_point(result, scenario)
    terminal_status_is_specific = (
        crossing is not None or result_stop_reason(result) == "solver_infeasible"
    )

    if not reached_goal(result) and not terminal_status_is_specific:
        final_position = states[-1, :2]
        ax.scatter(
            [final_position[0]],
            [final_position[1]],
            marker="o",
            s=COLLISION_OR_FINAL_MARKER_SIZE,
            color=color,
            linewidths=0.0,
            zorder=32,
            label="_nolegend_",
        )

    if crossing is not None:
        ax.scatter(
            [crossing[0]],
            [crossing[1]],
            marker="x",
            s=COLLISION_OR_FINAL_MARKER_SIZE,
            color=color,
            linewidths=1.4,
            zorder=31,
            label="_nolegend_",
        )


def trajectory_event_time(
    result: dict[str, np.ndarray | str | int | float],
    point: np.ndarray,
) -> float:
    states = np.asarray(result["states"], dtype=float)
    times = np.asarray(result["times"], dtype=float)
    n_values = min(states.shape[0], times.size)
    if n_values == 0:
        return result_stop_time(result)
    distances = np.linalg.norm(states[:n_values, :2] - point, axis=1)
    return float(times[int(np.argmin(distances))])


def representative_trajectory_events(
    results: list[dict[str, np.ndarray | str | int | float]],
    scenario: Scenario,
) -> dict[str, tuple[np.ndarray, float]]:
    events: dict[str, tuple[np.ndarray, float]] = {}

    for result in results:
        states = np.asarray(result["states"], dtype=float)
        crossing = first_collision_point(result, scenario)

        if "Collision" not in events:
            if crossing is not None:
                point = np.asarray(crossing, dtype=float)
                events["Collision"] = (point, trajectory_event_time(result, point))

        if "QP infeasible" not in events:
            if result_stop_reason(result) == "solver_infeasible" and states.size:
                point = states[-1, :2].copy()
                events["QP infeasible"] = (point, result_stop_time(result))
            else:
                fallback_positions = fallback_marker_positions(result, scenario)
                if fallback_positions.size:
                    point = fallback_positions[0].copy()
                    events["QP infeasible"] = (
                        point,
                        trajectory_event_time(result, point),
                    )

        if "Stopped" not in events:
            stop_reason = result_stop_reason(result)
            if (
                states.size
                and not reached_goal(result)
                and crossing is None
                and stop_reason != "solver_infeasible"
            ):
                point = states[-1, :2].copy()
                events["Stopped"] = (point, result_stop_time(result))

        if (
            "Collision" in events
            and "QP infeasible" in events
            and "Stopped" in events
        ):
            break

    return events


def representative_trajectory_event_points(
    results: list[dict[str, np.ndarray | str | int | float]],
    scenario: Scenario,
) -> dict[str, np.ndarray]:
    return {
        label: point
        for label, (point, _) in representative_trajectory_events(results, scenario).items()
    }


def annotate_trajectory_event_points(
    ax: plt.Axes,
    event_points: dict[str, np.ndarray],
) -> dict[str, plt.Annotation]:
    annotations: dict[str, plt.Annotation] = {}
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_span = max(abs(x_max - x_min), 1.0e-12)
    y_span = max(abs(y_max - y_min), 1.0e-12)
    for label in ("Collision", "QP infeasible", "Stopped"):
        point = event_points.get(label)
        if point is None:
            continue
        text_x = float(np.clip(point[0] + 0.13 * x_span, x_min + 0.08 * x_span, x_max - 0.24 * x_span))
        text_y = float(np.clip(point[1] + 0.22 * y_span, y_min + 0.12 * y_span, y_max - 0.18 * y_span))
        annotations[label] = ax.annotate(
            label,
            xy=(float(point[0]), float(point[1])),
            xycoords="data",
            xytext=(text_x, text_y),
            textcoords="data",
            ha="left",
            va="bottom",
            color="black",
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 0.8,
                "color": "black",
                "shrinkA": 0.0,
                "shrinkB": 3.0,
            },
        )
    return annotations


def print_progress(
    label: str,
    step: int,
    total: int,
    start_time: float,
    sim_time: float,
    force_newline: bool = False,
) -> None:
    width = 28
    fraction = 1.0 if total <= 0 else min(max(step / total, 0.0), 1.0)
    filled = int(round(width * fraction))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.perf_counter() - start_time
    message = (
        f"\r{label:<12} |{bar}| {100.0 * fraction:4.1f}% "
        f"step {step:4d}/{total:<4d} sim {sim_time:4.1f}s elapsed {elapsed:4.1f}s"
    )
    print(message, end="\n" if force_newline else "", flush=True)


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def dynamics(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    _, _, theta, v = x
    omega, accel = u
    return np.array(
        [v * math.cos(theta), v * math.sin(theta), omega, accel],
        dtype=float,
    )


def rk4_step(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    k1 = dynamics(x, u)
    k2 = dynamics(x + 0.5 * dt * k1, u)
    k3 = dynamics(x + 0.5 * dt * k2, u)
    k4 = dynamics(x + dt * k3, u)
    out = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    out[2] = wrap_angle(float(out[2]))
    return out


def barrier_terms(x: np.ndarray, scenario: Scenario) -> tuple[float, float, float, np.ndarray]:
    """Return h, L_f h, L_f^2 h, and L_g L_f h for the disk barrier."""
    px, py, theta, v = x
    ox, oy = scenario.obstacle
    dx = px - ox
    dy = py - oy
    c = math.cos(theta)
    s = math.sin(theta)
    q = dx * c + dy * s
    p = -dx * s + dy * c
    h = dx * dx + dy * dy - scenario.safe_radius**2
    lfh = 2.0 * v * q
    lf2h = 2.0 * v * v
    lg_lfh = np.array([2.0 * v * p, 2.0 * q], dtype=float)
    return h, lfh, lf2h, lg_lfh


def hddot_value(x: np.ndarray, u: np.ndarray, scenario: Scenario) -> float:
    _, _, lf2h, lg_lfh = barrier_terms(x, scenario)
    return float(lf2h + lg_lfh @ u)


def hthird_value(
    x: np.ndarray,
    u: np.ndarray,
    udot: np.ndarray,
    scenario: Scenario,
) -> float:
    px, py, theta, v = x
    ox, oy = scenario.obstacle
    dx = px - ox
    dy = py - oy
    q = dx * math.cos(theta) + dy * math.sin(theta)
    p = -dx * math.sin(theta) + dy * math.cos(theta)
    omega, accel = u
    omega_dot, accel_dot = udot
    return (
        6.0 * v * accel
        + 4.0 * p * omega * accel
        - 2.0 * v * q * omega * omega
        + 2.0 * v * p * omega_dot
        + 2.0 * q * accel_dot
    )


def nominal_control(x: np.ndarray, scenario: Scenario) -> np.ndarray:
    px, py, theta, v = x
    gx, gy = scenario.goal
    dx = gx - px
    dy = gy - py
    theta_des = math.atan2(dy, dx)
    heading_error = wrap_angle(theta_des - theta)
    v_des = scenario.cruise_speed
    omega = scenario.k_heading * heading_error
    accel = scenario.k_speed * (v_des - v)
    return np.clip(np.array([omega, accel], dtype=float), scenario.u_min, scenario.u_max)


def clf_row_data(x: np.ndarray, scenario: Scenario) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    px, py, theta, v = x
    gx, gy = scenario.goal
    dx = gx - px
    dy = gy - py
    theta_des = math.atan2(dy, dx)
    e_theta = wrap_angle(theta - theta_des)
    dist = math.hypot(dx, dy)
    v_des = scenario.cruise_speed
    e_v = v - v_des

    theta_des_dot = 0.0
    if dist > 1.0e-9:
        px_dot = v * math.cos(theta)
        py_dot = v * math.sin(theta)
        theta_des_dot = (dy * px_dot - dx * py_dot) / (dist * dist)

    return (
        np.array([2.0 * e_theta, 0.0], dtype=float),
        np.array([0.0, 2.0 * e_v], dtype=float),
        np.array(
            [
                -scenario.clf_lambda_heading * e_theta**2 + 2.0 * e_theta * theta_des_dot,
                -scenario.clf_lambda_speed * e_v**2,
            ],
            dtype=float,
        ),
    )


def valid_warm_start(warm_start: np.ndarray | None, dimension: int) -> np.ndarray | None:
    if warm_start is None:
        return None
    arr = np.asarray(warm_start, dtype=float).reshape(-1)
    if arr.size != dimension or not np.all(np.isfinite(arr)):
        return None
    return arr


def append_row(
    rows: list[np.ndarray],
    lower: list[float],
    upper: list[float],
    coefficients: np.ndarray,
    lo: float = -np.inf,
    hi: float = np.inf,
) -> None:
    rows.append(np.asarray(coefficients, dtype=float))
    lower.append(float(lo))
    upper.append(float(hi))


def append_control_speed_clf_constraints(
    rows: list[np.ndarray],
    lower: list[float],
    upper: list[float],
    dimension: int,
    x: np.ndarray,
    scenario: Scenario,
    slack_offset: int,
) -> None:
    row = np.zeros(dimension, dtype=float)
    row[0] = 1.0
    append_row(rows, lower, upper, row, scenario.omega_min, scenario.omega_max)

    row = np.zeros(dimension, dtype=float)
    row[1] = 1.0
    append_row(rows, lower, upper, row, scenario.accel_min, scenario.accel_max)

    row = np.zeros(dimension, dtype=float)
    row[1] = scenario.dt
    append_row(
        rows,
        lower,
        upper,
        row,
        scenario.v_min - float(x[3]),
        scenario.v_max - float(x[3]),
    )

    for idx in range(2):
        row = np.zeros(dimension, dtype=float)
        row[slack_offset + idx] = 1.0
        append_row(rows, lower, upper, row, 0.0, np.inf)

    heading_row, speed_row, clf_upper = clf_row_data(x, scenario)
    row = np.zeros(dimension, dtype=float)
    row[:2] = heading_row
    row[slack_offset] = -1.0
    append_row(rows, lower, upper, row, -np.inf, float(clf_upper[0]))

    row = np.zeros(dimension, dtype=float)
    row[:2] = speed_row
    row[slack_offset + 1] = -1.0
    append_row(rows, lower, upper, row, -np.inf, float(clf_upper[1]))


def direct_qp_solution_satisfies(
    A: sparse.csc_matrix,
    lower: np.ndarray,
    upper: np.ndarray,
    z: np.ndarray,
) -> bool:
    values = np.asarray(A @ z, dtype=float).reshape(-1)
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    if np.any(values[finite_lower] < lower[finite_lower] - DIRECT_QP_CONSTRAINT_TOL):
        return False
    if np.any(values[finite_upper] > upper[finite_upper] + DIRECT_QP_CONSTRAINT_TOL):
        return False
    return True


def solve_direct_osqp(
    P_diag: np.ndarray,
    q: np.ndarray,
    rows: list[np.ndarray],
    lower: list[float],
    upper: list[float],
    warm_start: np.ndarray | None = None,
) -> tuple[np.ndarray | None, str]:
    P_diag = np.asarray(P_diag, dtype=float)
    q = np.asarray(q, dtype=float)
    dimension = q.size
    if P_diag.size != dimension:
        raise ValueError("P_diag and q dimensions do not match")

    A = sparse.csc_matrix(np.vstack(rows) if rows else np.zeros((0, dimension), dtype=float))
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)
    P = sparse.diags(P_diag, format="csc")

    solver = osqp.OSQP()
    try:
        solver.setup(
            P=P,
            q=q,
            A=A,
            l=lower_arr,
            u=upper_arr,
            verbose=False,
            eps_abs=DIRECT_QP_EPS_ABS,
            eps_rel=DIRECT_QP_EPS_REL,
            max_iter=DIRECT_QP_MAX_ITER,
            polish=False,
            warm_starting=True,
        )
        z0 = valid_warm_start(warm_start, dimension)
        if z0 is not None:
            solver.warm_start(x=z0)
        result = solver.solve()
    except Exception as exc:  # OSQP raises several internal exception types.
        return None, f"osqp_error:{exc}"

    raw_status = str(result.info.status).lower()
    if raw_status == "solved":
        status = "optimal"
    elif raw_status == "solved inaccurate":
        status = "optimal_inaccurate"
    elif "primal infeasible" in raw_status:
        status = "infeasible"
    elif "dual infeasible" in raw_status:
        status = "unbounded"
    else:
        status = raw_status.replace(" ", "_")

    z = None if result.x is None else np.asarray(result.x, dtype=float)
    if z is None or not np.all(np.isfinite(z)):
        return None, status
    if canonical_qp_status(status) not in {"optimal", "optimal_inaccurate"}:
        return None, status
    if not direct_qp_solution_satisfies(A, lower_arr, upper_arr, z):
        return None, "infeasible_inaccurate"
    return z, status


def solve_qp(
    x: np.ndarray,
    scenario: Scenario,
    safety: SafetyConstraint,
    use_eta: bool = False,
    nominal: np.ndarray | None = None,
    warm_start: np.ndarray | None = None,
    timing: TimingAccumulator | None = None,
) -> QPResult:
    qp_started_at = time.perf_counter()
    status = "not_solved"
    try:
        if nominal is None:
            nominal = nominal_control(x, scenario)

        dimension = 5 if use_eta else 4
        slack_offset = 2
        eta_index = 4 if use_eta else None
        P_diag = np.zeros(dimension, dtype=float)
        q = np.zeros(dimension, dtype=float)
        P_diag[:2] = 2.0
        q[:2] = -2.0 * nominal
        P_diag[slack_offset : slack_offset + 2] = 2.0 * scenario.slack_weight
        if eta_index is not None:
            P_diag[eta_index] = 2.0 * scenario.attcbf_eta_weight

        rows: list[np.ndarray] = []
        lower: list[float] = []
        upper: list[float] = []
        append_control_speed_clf_constraints(rows, lower, upper, dimension, x, scenario, slack_offset)

        safety_row = np.zeros(dimension, dtype=float)
        safety_row[:2] = safety.A
        if use_eta:
            safety_row[eta_index] = safety.eta_coeff
            eta_bound = np.zeros(dimension, dtype=float)
            eta_bound[eta_index] = 1.0
            append_row(rows, lower, upper, eta_bound, 0.0, 1.0)
        append_row(rows, lower, upper, safety_row, -float(safety.b), np.inf)

        z, status = solve_direct_osqp(P_diag, q, rows, lower, upper, warm_start=warm_start)
        if z is not None:
            eta_value = float(z[eta_index]) if eta_index is not None else np.nan
            return QPResult(np.asarray(z[:2], dtype=float), True, status, eta_value, solver_state=z)

        accel_floor = max(scenario.accel_min, (scenario.v_min - x[3]) / scenario.dt)
        fallback = np.array([0.0, accel_floor], dtype=float)
        return QPResult(fallback, False, status, np.nan)
    finally:
        if timing is not None:
            record_qp_status(timing, status)
            timing.qp_calls += 1
            timing.qp_wall_time += time.perf_counter() - qp_started_at


def solve_avcbf_control(
    x: np.ndarray,
    aux_state: np.ndarray,
    scenario: Scenario,
    warm_start: np.ndarray | None = None,
    timing: TimingAccumulator | None = None,
) -> QPResult:
    """Solve the relative-degree-two AVCBF QP from the Liu et al. paper."""
    qp_started_at = time.perf_counter()
    status = "not_solved"
    try:
        nominal = nominal_control(x, scenario)
        h, lfh, lf2h, lg_lfh = barrier_terms(x, scenario)
        a_var, adot = float(aux_state[0]), float(aux_state[1])

        k1 = scenario.avcbf_k1
        k2 = scenario.avcbf_k2
        l1 = scenario.avcbf_l1
        l2 = scenario.avcbf_l2

        dimension = 5
        nu_index = 2
        slack_offset = 3
        P_diag = np.zeros(dimension, dtype=float)
        q = np.zeros(dimension, dtype=float)
        P_diag[:2] = 2.0
        q[:2] = -2.0 * nominal
        P_diag[nu_index] = 2.0 * scenario.avcbf_aux_weight
        q[nu_index] = -2.0 * scenario.avcbf_aux_weight * scenario.avcbf_aux_input_des
        P_diag[slack_offset : slack_offset + 2] = 2.0 * scenario.slack_weight

        rows: list[np.ndarray] = []
        lower: list[float] = []
        upper: list[float] = []
        append_control_speed_clf_constraints(rows, lower, upper, dimension, x, scenario, slack_offset)

        psi2_const = (
            a_var * lf2h
            + (2.0 * adot + a_var * (k1 + k2)) * lfh
            + (adot * (k1 + k2) + k1 * k2 * a_var) * h
        )
        row = np.zeros(dimension, dtype=float)
        row[:2] = a_var * lg_lfh
        row[nu_index] = h
        append_row(rows, lower, upper, row, -float(psi2_const), np.inf)

        aux_const = (l1 + l2) * adot + l1 * l2 * a_var - scenario.avcbf_epsilon
        row = np.zeros(dimension, dtype=float)
        row[nu_index] = 1.0
        append_row(rows, lower, upper, row, -float(aux_const), np.inf)

        z, status = solve_direct_osqp(P_diag, q, rows, lower, upper, warm_start=warm_start)
        if z is not None:
            return QPResult(
                np.asarray(z[:2], dtype=float),
                True,
                status,
                aux=float(z[nu_index]),
                solver_state=z,
            )

        accel_floor = max(scenario.accel_min, (scenario.v_min - x[3]) / scenario.dt)
        return QPResult(np.array([0.0, accel_floor], dtype=float), False, status, aux=0.0)
    finally:
        if timing is not None:
            record_qp_status(timing, status)
            timing.qp_calls += 1
            timing.qp_wall_time += time.perf_counter() - qp_started_at


def solve_pacbf_control(
    x: np.ndarray,
    p1: float,
    scenario: Scenario,
    warm_start: np.ndarray | None = None,
    timing: TimingAccumulator | None = None,
) -> QPResult:
    """Solve Xiao et al.'s relative-degree-two PACBF safety-filter QP."""
    qp_started_at = time.perf_counter()
    status = "not_solved"
    try:
        nominal = nominal_control(x, scenario)
        h, lfh, lf2h, lg_lfh = barrier_terms(x, scenario)
        p1 = max(float(p1), scenario.pacbf_epsilon)
        p1_error = p1 - scenario.pacbf_p1_des

        psi1 = lfh + p1 * h * h
        dimension = 7
        nu_index = 2
        p2_index = 3
        p1_delta_index = 4
        slack_offset = 5
        P_diag = np.zeros(dimension, dtype=float)
        q = np.zeros(dimension, dtype=float)
        P_diag[:2] = 2.0
        q[:2] = -2.0 * nominal
        P_diag[nu_index] = 2.0 * scenario.pacbf_nu_regularization
        q[nu_index] = scenario.pacbf_nu_weight
        P_diag[p1_delta_index] = 2.0 * scenario.pacbf_p1_clf_weight
        P_diag[p2_index] = 2.0 * scenario.pacbf_p2_weight
        q[p2_index] = -2.0 * scenario.pacbf_p2_weight * scenario.pacbf_p2_des
        P_diag[slack_offset : slack_offset + 2] = 2.0 * scenario.slack_weight

        rows: list[np.ndarray] = []
        lower: list[float] = []
        upper: list[float] = []
        append_control_speed_clf_constraints(rows, lower, upper, dimension, x, scenario, slack_offset)

        for idx in (p2_index, p1_delta_index):
            row = np.zeros(dimension, dtype=float)
            row[idx] = 1.0
            append_row(rows, lower, upper, row, 0.0, np.inf)

        row = np.zeros(dimension, dtype=float)
        row[:2] = lg_lfh
        row[nu_index] = h * h
        row[p2_index] = psi1
        psi2_const = lf2h + 2.0 * p1 * h * lfh
        append_row(rows, lower, upper, row, -float(psi2_const), np.inf)

        row = np.zeros(dimension, dtype=float)
        row[nu_index] = 1.0
        append_row(rows, lower, upper, row, -float(p1), np.inf)

        row = np.zeros(dimension, dtype=float)
        row[nu_index] = 2.0 * p1_error
        row[p1_delta_index] = -1.0
        append_row(
            rows,
            lower,
            upper,
            row,
            -np.inf,
            -scenario.pacbf_p1_clf_rate * p1_error**2,
        )

        z, status = solve_direct_osqp(P_diag, q, rows, lower, upper, warm_start=warm_start)
        if z is not None:
            return QPResult(
                np.asarray(z[:2], dtype=float),
                True,
                status,
                eta=float(z[p2_index]),
                aux=float(z[nu_index]),
                solver_state=z,
            )

        accel_floor = max(scenario.accel_min, (scenario.v_min - x[3]) / scenario.dt)
        return QPResult(
            np.array([0.0, accel_floor], dtype=float),
            False,
            status,
            eta=scenario.pacbf_p2_des,
            aux=0.0,
        )
    finally:
        if timing is not None:
            record_qp_status(timing, status)
            timing.qp_calls += 1
            timing.qp_wall_time += time.perf_counter() - qp_started_at


def solve_racbf_control(
    x: np.ndarray,
    relax_state: np.ndarray,
    scenario: Scenario,
    warm_start: np.ndarray | None = None,
    timing: TimingAccumulator | None = None,
) -> QPResult:
    """Solve Xiao et al.'s relative-degree-two RACBF safety-filter QP."""
    qp_started_at = time.perf_counter()
    status = "not_solved"
    try:
        nominal = nominal_control(x, scenario)
        h, lfh, lf2h, lg_lfh = barrier_terms(x, scenario)
        r_value, rdot = float(relax_state[0]), float(relax_state[1])

        psi0 = h - r_value
        psi1 = lfh - rdot + scenario.racbf_k1 * psi0 * psi0
        r_track = rdot + scenario.racbf_l1 * (r_value - scenario.racbf_r_des)

        dimension = 6
        nu_index = 2
        r_delta_index = 3
        slack_offset = 4
        P_diag = np.zeros(dimension, dtype=float)
        q = np.zeros(dimension, dtype=float)
        P_diag[:2] = 2.0
        q[:2] = -2.0 * nominal
        P_diag[nu_index] = 2.0 * scenario.racbf_aux_weight
        P_diag[r_delta_index] = 2.0 * scenario.racbf_clf_weight
        P_diag[slack_offset : slack_offset + 2] = 2.0 * scenario.slack_weight

        rows: list[np.ndarray] = []
        lower: list[float] = []
        upper: list[float] = []
        append_control_speed_clf_constraints(rows, lower, upper, dimension, x, scenario, slack_offset)

        row = np.zeros(dimension, dtype=float)
        row[:2] = lg_lfh
        row[nu_index] = -1.0
        psi2_const = (
            lf2h
            + 2.0 * scenario.racbf_k1 * psi0 * (lfh - rdot)
            + scenario.racbf_k2 * psi1
        )
        append_row(rows, lower, upper, row, -float(psi2_const), np.inf)

        row = np.zeros(dimension, dtype=float)
        row[nu_index] = 1.0
        r_hocbf_const = (
            (scenario.racbf_l1 + scenario.racbf_l2) * rdot
            + scenario.racbf_l1 * scenario.racbf_l2 * (r_value - scenario.racbf_r_min)
        )
        append_row(rows, lower, upper, row, -float(r_hocbf_const), np.inf)

        row = np.zeros(dimension, dtype=float)
        row[nu_index] = 2.0 * r_track
        row[r_delta_index] = -1.0
        r_clf_const = 2.0 * r_track * scenario.racbf_l1 * rdot + scenario.racbf_clf_rate * r_track**2
        append_row(rows, lower, upper, row, -np.inf, -float(r_clf_const))

        row = np.zeros(dimension, dtype=float)
        row[r_delta_index] = 1.0
        append_row(rows, lower, upper, row, 0.0, np.inf)

        z, status = solve_direct_osqp(P_diag, q, rows, lower, upper, warm_start=warm_start)
        if z is not None:
            return QPResult(
                np.asarray(z[:2], dtype=float),
                True,
                status,
                eta=r_value,
                aux=float(z[nu_index]),
                solver_state=z,
            )

        accel_floor = max(scenario.accel_min, (scenario.v_min - x[3]) / scenario.dt)
        return QPResult(np.array([0.0, accel_floor], dtype=float), False, status, eta=r_value, aux=0.0)
    finally:
        if timing is not None:
            record_qp_status(timing, status)
            timing.qp_calls += 1
            timing.qp_wall_time += time.perf_counter() - qp_started_at


def euler_sample_step(x: np.ndarray, u: np.ndarray, scenario: Scenario) -> np.ndarray:
    px, py, theta, v = x
    omega, accel = u
    dt = scenario.dt
    return np.array(
        [
            px + dt * v * math.cos(theta),
            py + dt * v * math.sin(theta),
            wrap_angle(theta + dt * omega),
            v + dt * accel,
        ],
        dtype=float,
    )


def barrier_value(x: np.ndarray, scenario: Scenario) -> float:
    px, py = float(x[0]), float(x[1])
    ox, oy = scenario.obstacle
    dx = px - ox
    dy = py - oy
    return dx * dx + dy * dy - scenario.safe_radius**2


def dt_hocbf_psi2(
    x: np.ndarray,
    u: np.ndarray,
    scenario: Scenario,
    gamma1: float,
    gamma2: float,
    gamma1_next: float | None = None,
) -> float:
    """Evaluate the relative-degree-two discrete-time HOCBF condition."""
    if gamma1_next is None:
        gamma1_next = gamma1

    x1 = euler_sample_step(x, u, scenario)
    # h(x_{k+1}) is independent of u_k for this sampled relative-degree-two map.
    h0 = barrier_value(x, scenario)
    h1 = barrier_value(x1, scenario)

    # The second position update depends on u_k through theta_{k+1} and v_{k+1};
    # the next control does not enter h because h depends only on position.
    x2_pos = np.array(
        [
            x1[0] + scenario.dt * x1[3] * math.cos(x1[2]),
            x1[1] + scenario.dt * x1[3] * math.sin(x1[2]),
            x1[2],
            x1[3],
        ],
        dtype=float,
    )
    h2 = barrier_value(x2_pos, scenario)

    psi1_k = h1 - h0 + gamma1 * h0
    psi1_next = h2 - h1 + gamma1_next * h1
    return float(psi1_next - psi1_k + gamma2 * psi1_k)


def solve_dt_hocbf_control(
    x: np.ndarray,
    scenario: Scenario,
    warm_start: np.ndarray | None = None,
    timing: TimingAccumulator | None = None,
) -> QPResult:
    solve_started_at = time.perf_counter()
    status = "not_solved"
    try:
        nominal = nominal_control(x, scenario)

        def objective(z: np.ndarray) -> float:
            return float(np.sum((z[:2] - nominal) ** 2))

        def safety_constraint(z: np.ndarray) -> float:
            return dt_hocbf_psi2(
                x,
                z[:2],
                scenario,
                scenario.dt_hocbf_gamma1,
                scenario.dt_hocbf_gamma2,
            )

        def speed_lower(z: np.ndarray) -> float:
            return float(x[3] + scenario.dt * z[1] - scenario.v_min)

        def speed_upper(z: np.ndarray) -> float:
            return float(scenario.v_max - (x[3] + scenario.dt * z[1]))

        def make_start(u_guess: np.ndarray) -> np.ndarray:
            z_guess = np.asarray(u_guess, dtype=float).copy()
            z_guess[0] = np.clip(z_guess[0], scenario.omega_min, scenario.omega_max)
            accel_min = max(scenario.accel_min, (scenario.v_min - x[3]) / scenario.dt)
            accel_max = min(scenario.accel_max, (scenario.v_max - x[3]) / scenario.dt)
            z_guess[1] = np.clip(z_guess[1], accel_min, accel_max)
            return z_guess

        def constraints_satisfied(z: np.ndarray) -> bool:
            return all(fun(z) >= -NLP_CONSTRAINT_TOL for fun in constraint_functions)

        bounds = optimize.Bounds(
            [scenario.omega_min, scenario.accel_min],
            [scenario.omega_max, scenario.accel_max],
        )
        constraint_functions = [
            safety_constraint,
            speed_lower,
            speed_upper,
        ]
        constraints = [{"type": "ineq", "fun": fun} for fun in constraint_functions]
        warm_u = valid_warm_start(warm_start, 2)
        start_controls = [
            nominal,
            np.array([scenario.omega_min, scenario.accel_min], dtype=float),
            np.array([scenario.omega_max, scenario.accel_min], dtype=float),
            np.array([scenario.omega_min, scenario.accel_max], dtype=float),
            np.array([scenario.omega_max, scenario.accel_max], dtype=float),
            np.array([0.0, scenario.accel_min], dtype=float),
            np.array([0.0, 0.0], dtype=float),
        ]
        last_message = "not_solved"
        best: optimize.OptimizeResult | None = None
        for u0 in start_controls:
            result = optimize.minimize(
                objective,
                make_start(u0),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": NLP_FTOL, "maxiter": NLP_DT_MAXITER, "disp": False},
            )
            last_message = str(result.message)
            if constraints_satisfied(result.x):
                if best is None or objective(result.x) < objective(best.x):
                    best = result
                if result.success:
                    break
        if best is None and warm_u is not None:
            result = optimize.minimize(
                objective,
                make_start(warm_u),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": NLP_FTOL, "maxiter": NLP_DT_MAXITER, "disp": False},
            )
            last_message = str(result.message)
            if constraints_satisfied(result.x):
                best = result
        if best is not None:
            status = "optimal" if bool(best.success) else "feasible_nonoptimal_slsqp"
            return QPResult(
                np.asarray(best.x[:2], dtype=float),
                True,
                status,
                solver_state=np.asarray(best.x[:2], dtype=float),
            )

        accel_floor = max(scenario.accel_min, (scenario.v_min - x[3]) / scenario.dt)
        status = last_message
        return QPResult(np.array([0.0, accel_floor], dtype=float), False, status)
    finally:
        if timing is not None:
            record_qp_status(timing, status)
            timing.qp_calls += 1
            timing.qp_wall_time += time.perf_counter() - solve_started_at


def solve_adt_hocbf_control(
    x: np.ndarray,
    gamma1: float,
    scenario: Scenario,
    warm_start: np.ndarray | None = None,
    timing: TimingAccumulator | None = None,
) -> QPResult:
    solve_started_at = time.perf_counter()
    status = "not_solved"
    try:
        nominal = nominal_control(x, scenario)
        g_min = scenario.adt_hocbf_gamma_min
        g_max = scenario.adt_hocbf_gamma_max

        def objective(z: np.ndarray) -> float:
            u = z[:2]
            nu_gamma = z[2]
            gamma2 = z[3]
            gamma_delta = z[4]
            return float(
                np.sum((u - nominal) ** 2)
                + scenario.adt_hocbf_aux_weight * nu_gamma**2
                + scenario.adt_hocbf_gamma_clf_weight * gamma_delta**2
                + scenario.adt_hocbf_gamma2_weight * (gamma2 - scenario.adt_hocbf_gamma2_des) ** 2
            )

        def safety_constraint(z: np.ndarray) -> float:
            gamma1_next = gamma1 + z[2]
            return dt_hocbf_psi2(x, z[:2], scenario, gamma1, z[3], gamma1_next=gamma1_next)

        def gamma_lower_cbf(z: np.ndarray) -> float:
            return float(z[2] + scenario.adt_hocbf_beta1 * (gamma1 - g_min))

        def gamma_upper_cbf(z: np.ndarray) -> float:
            return float(-z[2] + scenario.adt_hocbf_beta2 * (g_max - gamma1))

        def gamma_clf(z: np.ndarray) -> float:
            gamma1_next = gamma1 + z[2]
            v_now = (gamma1 - scenario.adt_hocbf_gamma1_des) ** 2
            v_next = (gamma1_next - scenario.adt_hocbf_gamma1_des) ** 2
            return float(z[4] - (v_next - v_now + scenario.adt_hocbf_gamma_clf_rate * v_now))

        def speed_lower(z: np.ndarray) -> float:
            return float(x[3] + scenario.dt * z[1] - scenario.v_min)

        def speed_upper(z: np.ndarray) -> float:
            return float(scenario.v_max - (x[3] + scenario.dt * z[1]))

        gamma_nu_min = -scenario.adt_hocbf_beta1 * (gamma1 - g_min)
        gamma_nu_max = scenario.adt_hocbf_beta2 * (g_max - gamma1)
        gamma_nu_min = min(gamma_nu_min, gamma_nu_max)

        def make_start(u_guess: np.ndarray, nu_guess: float, gamma2_guess: float) -> np.ndarray:
            z_guess = np.array(
                [
                    u_guess[0],
                    u_guess[1],
                    np.clip(nu_guess, gamma_nu_min, gamma_nu_max),
                    np.clip(gamma2_guess, g_min, g_max),
                    0.0,
                ],
                dtype=float,
            )
            z_guess[0] = np.clip(z_guess[0], scenario.omega_min, scenario.omega_max)
            accel_min = max(scenario.accel_min, (scenario.v_min - x[3]) / scenario.dt)
            accel_max = min(scenario.accel_max, (scenario.v_max - x[3]) / scenario.dt)
            z_guess[1] = np.clip(z_guess[1], accel_min, accel_max)
            z_guess[4] = max(0.0, -gamma_clf(z_guess))
            return z_guess

        def constraints_satisfied(z: np.ndarray) -> bool:
            return all(fun(z) >= -NLP_CONSTRAINT_TOL for fun in constraint_functions)

        bounds = optimize.Bounds(
            [
                scenario.omega_min,
                scenario.accel_min,
                gamma_nu_min,
                g_min,
                0.0,
            ],
            [
                scenario.omega_max,
                scenario.accel_max,
                gamma_nu_max,
                g_max,
                np.inf,
            ],
        )
        constraint_functions = [
            safety_constraint,
            gamma_lower_cbf,
            gamma_upper_cbf,
            gamma_clf,
            speed_lower,
            speed_upper,
        ]
        constraints = [{"type": "ineq", "fun": fun} for fun in constraint_functions]
        warm_z = valid_warm_start(warm_start, 5)
        start_controls = [
            nominal,
            np.array([scenario.omega_min, scenario.accel_min], dtype=float),
            np.array([scenario.omega_max, scenario.accel_min], dtype=float),
            np.array([scenario.omega_min, scenario.accel_max], dtype=float),
            np.array([scenario.omega_max, scenario.accel_max], dtype=float),
            np.array([0.0, scenario.accel_min], dtype=float),
            np.array([0.0, 0.0], dtype=float),
        ]
        gamma2_starts = [
            scenario.adt_hocbf_gamma2_des,
            g_min,
            0.5 * (g_min + g_max),
            g_max,
        ]
        nu_starts = [0.0, gamma_nu_min, gamma_nu_max]
        last_message = "not_solved"
        best: optimize.OptimizeResult | None = None
        for u0 in start_controls:
            for nu0 in nu_starts:
                for gamma20 in gamma2_starts:
                    result = optimize.minimize(
                        objective,
                        make_start(u0, nu0, gamma20),
                        method="SLSQP",
                        bounds=bounds,
                        constraints=constraints,
                        options={"ftol": NLP_FTOL, "maxiter": NLP_ADT_MAXITER, "disp": False},
                    )
                    last_message = str(result.message)
                    if constraints_satisfied(result.x):
                        if best is None or objective(result.x) < objective(best.x):
                            best = result
                        if result.success:
                            break
                if best is not None and best.success:
                    break
            if best is not None and best.success:
                break
        if best is None and warm_z is not None:
            result = optimize.minimize(
                objective,
                make_start(warm_z[:2], float(warm_z[2]), float(warm_z[3])),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": NLP_FTOL, "maxiter": NLP_ADT_MAXITER, "disp": False},
            )
            last_message = str(result.message)
            if constraints_satisfied(result.x):
                best = result
        if best is not None:
            status = "optimal" if bool(best.success) else "feasible_nonoptimal_slsqp"
            return QPResult(
                np.asarray(best.x[:2], dtype=float),
                True,
                status,
                eta=float(best.x[3]),
                aux=float(best.x[2]),
                solver_state=np.asarray(best.x, dtype=float),
            )

        accel_floor = max(scenario.accel_min, (scenario.v_min - x[3]) / scenario.dt)
        status = last_message
        return QPResult(np.array([0.0, accel_floor], dtype=float), False, status, aux=0.0)
    finally:
        if timing is not None:
            record_qp_status(timing, status)
            timing.qp_calls += 1
            timing.qp_wall_time += time.perf_counter() - solve_started_at


def ct_hocbf_safety(x: np.ndarray, scenario: Scenario) -> SafetyConstraint:
    h, lfh, lf2h, lg_lfh = barrier_terms(x, scenario)
    A, b = relative_degree_two_hocbf_constraint(
        h=h,
        lf_h=lfh,
        lf2_h=lf2h,
        lg_lf_h=lg_lfh,
        p1=scenario.ct_hocbf_p1,
        p2=scenario.ct_hocbf_p2,
    )
    return SafetyConstraint(A=A, b=b)


def pointwise_tlc_safety(x: np.ndarray, scenario: Scenario, tau: float) -> SafetyConstraint:
    h, lfh, lf2h, lg_lfh = barrier_terms(x, scenario)
    return SafetyConstraint(
        A=lg_lfh,
        b=lf2h + (2.0 / tau) * lfh + (2.0 / (tau * tau)) * h,
    )


def state_box_samples(
    x: np.ndarray,
    widths: np.ndarray,
    scenario: Scenario,
    samples_per_dim: int,
) -> list[np.ndarray]:
    grids = []
    for idx, width in enumerate(widths):
        if width <= 0.0 or samples_per_dim <= 1:
            values = np.array([x[idx]])
        else:
            values = np.linspace(x[idx] - width, x[idx] + width, samples_per_dim)
        if idx == 2:
            values = np.array([wrap_angle(float(v)) for v in values])
        if idx == 3:
            values = np.clip(values, scenario.v_min, scenario.v_max)
        grids.append(values)
    return [np.array(values, dtype=float) for values in itertools.product(*grids)]


def event_tlc_safety(
    x: np.ndarray,
    scenario: Scenario,
    tau: float,
    sign_reference: np.ndarray,
) -> SafetyConstraint:
    widths = np.array(scenario.event_triggered_tlc_state_box_widths, dtype=float)
    samples = state_box_samples(x, widths, scenario, scenario.event_triggered_tlc_samples_per_dim)
    base_values = []
    g_values = []
    for sample in samples:
        h, lfh, lf2h, lg_lfh = barrier_terms(sample, scenario)
        base_values.append(h + tau * lfh + 0.5 * tau * tau * lf2h)
        g_values.append(0.5 * tau * tau * lg_lfh)
    g_values_arr = np.asarray(g_values)
    G = np.empty(2, dtype=float)
    for j in range(2):
        if sign_reference[j] >= 0.0:
            G[j] = float(np.min(g_values_arr[:, j]))
        else:
            G[j] = float(np.max(g_values_arr[:, j]))
    return SafetyConstraint(A=G, b=float(np.min(base_values)))


def solve_event_tlc_control(
    x: np.ndarray,
    scenario: Scenario,
    tau: float,
    warm_start: np.ndarray | None = None,
    timing: TimingAccumulator | None = None,
) -> QPResult:
    pre_result = solve_qp(
        x,
        scenario,
        pointwise_tlc_safety(x, scenario, tau),
        warm_start=warm_start,
        timing=timing,
    )
    sign_reference = pre_result.u if pre_result.feasible else nominal_control(x, scenario)
    safety = event_tlc_safety(x, scenario, tau, sign_reference)
    return solve_qp(
        x,
        scenario,
        safety,
        warm_start=pre_result.solver_state if pre_result.feasible else warm_start,
        timing=timing,
    )


def rtlc_remainder_bound(x: np.ndarray, scenario: Scenario, tau: float) -> float:
    max_accel = max(abs(scenario.accel_min), abs(scenario.accel_max))
    v_candidates = (
        x[3],
        x[3] + scenario.accel_min * tau,
        x[3] + scenario.accel_max * tau,
    )
    v_reach = max(abs(float(v)) for v in v_candidates)
    pos_width = max(0.0, v_reach * tau + 0.5 * max_accel * tau * tau)
    local_widths = np.array(
        [
            pos_width,
            pos_width,
            scenario.omega_max * tau,
            max_accel * tau,
        ],
        dtype=float,
    )
    samples = state_box_samples(x, local_widths, scenario, 3)
    u_corners = [
        np.array([omega, accel], dtype=float)
        for omega in (scenario.omega_min, scenario.omega_max)
        for accel in (scenario.accel_min, scenario.accel_max)
    ]
    du_span = control_rate_bounds(scenario)
    udot_corners = [
        np.array([omega_dot, accel_dot], dtype=float)
        for omega_dot in (-du_span[0], du_span[0])
        for accel_dot in (-du_span[1], du_span[1])
    ]
    h3_min = math.inf
    for sample in samples:
        for u in u_corners:
            for udot in udot_corners:
                h3_min = min(h3_min, hthird_value(sample, u, udot, scenario))
    return (tau / 3.0) * h3_min


def rtlc_safety(x: np.ndarray, scenario: Scenario, tau: float) -> SafetyConstraint:
    base = pointwise_tlc_safety(x, scenario, tau)
    return SafetyConstraint(A=base.A, b=base.b + rtlc_remainder_bound(x, scenario, tau))


def min_affine_over_box(A: np.ndarray, scenario: Scenario) -> float:
    return float(
        sum(A[j] * (scenario.u_min[j] if A[j] >= 0.0 else scenario.u_max[j]) for j in range(2))
    )


Interval = tuple[float, float]


def interval(lo: float, hi: float) -> Interval:
    return (float(min(lo, hi)), float(max(lo, hi)))


def interval_add(a: Interval, b: Interval) -> Interval:
    return (a[0] + b[0], a[1] + b[1])


def interval_neg(a: Interval) -> Interval:
    return (-a[1], -a[0])


def interval_sub(a: Interval, b: Interval) -> Interval:
    return interval_add(a, interval_neg(b))


def interval_scale(a: Interval, scale: float) -> Interval:
    values = (scale * a[0], scale * a[1])
    return interval(values[0], values[1])


def interval_mul(a: Interval, b: Interval) -> Interval:
    products = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return (min(products), max(products))


def interval_square(a: Interval) -> Interval:
    if a[0] <= 0.0 <= a[1]:
        return (0.0, max(a[0] * a[0], a[1] * a[1]))
    return interval(a[0] * a[0], a[1] * a[1])


def interval_contains_periodic_point(a: Interval, point: float, period: float = 2.0 * math.pi) -> bool:
    k_min = math.ceil((a[0] - point) / period)
    k_max = math.floor((a[1] - point) / period)
    return k_min <= k_max


def interval_sin(a: Interval) -> Interval:
    if a[1] - a[0] >= 2.0 * math.pi:
        return (-1.0, 1.0)
    lo = min(math.sin(a[0]), math.sin(a[1]))
    hi = max(math.sin(a[0]), math.sin(a[1]))
    if interval_contains_periodic_point(a, math.pi / 2.0):
        hi = 1.0
    if interval_contains_periodic_point(a, -math.pi / 2.0):
        lo = -1.0
    return (lo, hi)


def interval_cos(a: Interval) -> Interval:
    return interval_sin((a[0] + math.pi / 2.0, a[1] + math.pi / 2.0))


def ttcbf_delta_taylor(scenario: Scenario) -> float:
    return 2.0 * scenario.dt


def control_rate_bounds(scenario: Scenario) -> np.ndarray:
    return scenario.u_rate_max


def ttcbf_reachable_state_box(x: np.ndarray, scenario: Scenario) -> tuple[np.ndarray, np.ndarray]:
    delta_taylor = ttcbf_delta_taylor(scenario)
    max_accel = max(abs(scenario.accel_min), abs(scenario.accel_max))
    v_candidates = (
        float(x[3]),
        float(x[3] + scenario.accel_min * delta_taylor),
        float(x[3] + scenario.accel_max * delta_taylor),
    )
    v_lo = max(scenario.v_min, min(v_candidates))
    v_hi = min(scenario.v_max, max(v_candidates))
    v_reach = max(abs(v_lo), abs(v_hi))
    pos_width = max(0.0, v_reach * delta_taylor + 0.5 * max_accel * delta_taylor * delta_taylor)
    theta_width = max(abs(scenario.omega_min), abs(scenario.omega_max)) * delta_taylor
    low = np.array(
        [x[0] - pos_width, x[1] - pos_width, x[2] - theta_width, v_lo],
        dtype=float,
    )
    high = np.array(
        [x[0] + pos_width, x[1] + pos_width, x[2] + theta_width, v_hi],
        dtype=float,
    )
    return low, high


def pq_intervals_from_state_box(
    state_low: np.ndarray,
    state_high: np.ndarray,
    scenario: Scenario,
) -> tuple[Interval, Interval]:
    ox, oy = scenario.obstacle
    dx = interval_sub((state_low[0], state_high[0]), (ox, ox))
    dy = interval_sub((state_low[1], state_high[1]), (oy, oy))
    theta = (state_low[2], state_high[2])
    c = interval_cos(theta)
    s = interval_sin(theta)
    q = interval_add(interval_mul(dx, c), interval_mul(dy, s))
    p = interval_add(interval_neg(interval_mul(dx, s)), interval_mul(dy, c))
    return p, q


def hthird_interval_lower(
    state_low: np.ndarray,
    state_high: np.ndarray,
    scenario: Scenario,
) -> float:
    p, q = pq_intervals_from_state_box(state_low, state_high, scenario)
    v = (state_low[3], state_high[3])
    omega = (scenario.omega_min, scenario.omega_max)
    accel = (scenario.accel_min, scenario.accel_max)
    udot_bounds = control_rate_bounds(scenario)
    omega_dot = (-udot_bounds[0], udot_bounds[0])
    accel_dot = (-udot_bounds[1], udot_bounds[1])
    omega_sq = interval_square(omega)

    terms = [
        interval_scale(interval_mul(v, accel), 6.0),
        interval_scale(interval_mul(interval_mul(p, omega), accel), 4.0),
        interval_scale(interval_mul(interval_mul(v, q), omega_sq), -2.0),
        interval_scale(interval_mul(interval_mul(v, p), omega_dot), 2.0),
        interval_scale(interval_mul(q, accel_dot), 2.0),
    ]
    total = (0.0, 0.0)
    for term in terms:
        total = interval_add(total, term)
    return float(total[0])


def rigorous_hthird_lower(x: np.ndarray, scenario: Scenario) -> float:
    state_low, state_high = ttcbf_reachable_state_box(x, scenario)
    return hthird_interval_lower(state_low, state_high, scenario)


def ttcbf_remainder_value(
    x: np.ndarray,
    scenario: Scenario,
) -> tuple[float, float]:
    delta_taylor = ttcbf_delta_taylor(scenario)
    h3_lower = rigorous_hthird_lower(x, scenario)
    return (delta_taylor**3 / 6.0) * h3_lower, h3_lower


def ttcbf_safety(
    x: np.ndarray,
    scenario: Scenario,
    adaptive: bool,
) -> SafetyConstraint:
    h, lfh, lf2h, lg_lfh = barrier_terms(x, scenario)
    delta_taylor = ttcbf_delta_taylor(scenario)
    remainder, _ = ttcbf_remainder_value(x, scenario)
    A = 0.5 * delta_taylor * delta_taylor * lg_lfh
    b = delta_taylor * lfh + 0.5 * delta_taylor * delta_taylor * lf2h + remainder
    if adaptive:
        return SafetyConstraint(A=A, b=b, eta_coeff=max(h, 0.0))
    return SafetyConstraint(A=A, b=b + scenario.ttcbf_alpha * max(h, 0.0))


def taylor_residual(x: np.ndarray, u: np.ndarray, x_forward: np.ndarray, scenario: Scenario) -> float:
    h, lfh, _, _ = barrier_terms(x, scenario)
    delta_taylor = 2.0 * scenario.dt
    h_forward = barrier_terms(x_forward, scenario)[0]
    hddot = hddot_value(x, u, scenario)
    return float(h_forward - h - delta_taylor * lfh - 0.5 * delta_taylor * delta_taylor * hddot)


def ttcbf_remainder_diagnostics(
    states: np.ndarray,
    controls: np.ndarray,
    scenario: Scenario,
) -> dict[str, np.ndarray]:
    n_steps = controls.shape[0]
    estimates = np.full(n_steps, np.nan, dtype=float)
    hthird_lower_bounds = np.full(n_steps, np.nan, dtype=float)
    closed_loop_residuals = np.full(n_steps, np.nan, dtype=float)
    hddot_min_values = np.full(n_steps, np.nan, dtype=float)
    previous_hddot_values = np.full(n_steps, np.nan, dtype=float)

    for k in range(n_steps):
        x = states[k]
        u = controls[k]
        _, _, lf2h, lg_lfh = barrier_terms(x, scenario)
        hddot_min = lf2h + min_affine_over_box(lg_lfh, scenario)
        previous_hddot = (
            hddot_value(states[0], np.zeros(2), scenario)
            if k == 0
            else hddot_value(states[k - 1], controls[k - 1], scenario)
        )
        estimates[k], hthird_lower_bounds[k] = ttcbf_remainder_value(
            x,
            scenario,
        )
        hddot_min_values[k] = hddot_min
        previous_hddot_values[k] = previous_hddot

        if k + 2 < states.shape[0]:
            closed_loop_residuals[k] = taylor_residual(x, u, states[k + 2], scenario)

    return {
        "ttcbf_remainder_estimate": estimates,
        "ttcbf_hthird_lower_bound": hthird_lower_bounds,
        "ttcbf_remainder_closed_loop_residual": closed_loop_residuals,
        "ttcbf_hddot_min": hddot_min_values,
        "ttcbf_previous_hddot": previous_hddot_values,
    }


def rollout_min_h(x: np.ndarray, u: np.ndarray, scenario: Scenario, duration: float) -> float:
    steps = max(1, int(round(duration / scenario.dt)))
    state = x.copy()
    min_h = barrier_terms(state, scenario)[0]
    for _ in range(steps):
        state = rk4_step(state, u, scenario.dt)
        min_h = min(min_h, barrier_terms(state, scenario)[0])
    return float(min_h)


def select_atlc_control(
    x: np.ndarray,
    scenario: Scenario,
    warm_start: np.ndarray | None = None,
    timing: TimingAccumulator | None = None,
) -> tuple[QPResult, float]:
    best_result: QPResult | None = None
    best_tau = float(scenario.event_triggered_atlc_tau_candidates[0])
    best_score = -math.inf
    for tau in scenario.event_triggered_atlc_tau_candidates:
        result = solve_event_tlc_control(x, scenario, tau, warm_start=warm_start, timing=timing)
        if not result.feasible:
            continue
        score = rollout_min_h(x, result.u, scenario, scenario.event_triggered_atlc_lookahead)
        if score > best_score:
            best_score = score
            best_result = result
            best_tau = float(tau)
    if best_result is None:
        fallback = QPResult(np.array([0.0, scenario.accel_min], dtype=float), False, "all_tau_infeasible")
        return fallback, best_tau
    return best_result, best_tau


def outside_event_box(x: np.ndarray, anchor: np.ndarray, scenario: Scenario) -> bool:
    widths = np.array(scenario.event_triggered_tlc_state_box_widths, dtype=float)
    diffs = np.abs(x - anchor)
    diffs[2] = abs(wrap_angle(float(x[2] - anchor[2])))
    return bool(np.any(diffs > widths))


def simulate_method(
    method: str,
    scenario: Scenario,
    show_progress: bool = False,
) -> dict[str, np.ndarray | str | int | float]:
    n_steps = int(round(scenario.horizon / scenario.dt))
    scheduled_times = np.linspace(0.0, scenario.horizon, n_steps + 1)

    states: list[np.ndarray] = [np.array(scenario.start, dtype=float)]
    times: list[float] = [0.0]
    controls: list[np.ndarray] = []
    control_times: list[float] = []
    step_times: list[float] = []
    step_compute_times: list[float] = []
    qp_compute_times: list[float] = []
    qp_calls_per_step: list[int] = []
    feasible: list[bool] = []
    step_statuses: list[str] = []
    qp_event_statuses: list[str] = []
    event_flags: list[bool] = []
    control_event_flags: list[bool] = []
    tau_values: list[float] = []
    eta_values: list[float] = []

    initial_h = barrier_terms(states[0], scenario)[0]
    h_values: list[float] = [initial_h]
    clearance_values: list[float] = [
        math.sqrt(max(initial_h + scenario.safe_radius**2, 0.0)) - scenario.safe_radius
    ]
    goal_distance: list[float] = [
        float(np.linalg.norm(states[0][:2] - np.array(scenario.goal)))
    ]

    held_u = nominal_control(states[0], scenario)
    event_anchor = states[0].copy()
    pacbf_p1 = scenario.pacbf_p1_0
    racbf_aux_state = np.array([scenario.racbf_r0, scenario.racbf_rdot0], dtype=float)
    avcbf_aux_state = np.array([scenario.avcbf_a0, scenario.avcbf_adot0], dtype=float)
    adt_hocbf_gamma1 = scenario.adt_hocbf_gamma1_0
    current_tau = np.nan
    current_eta = np.nan
    current_aux = np.nan
    solver_warm_start: np.ndarray | None = None
    timing = TimingAccumulator()
    stop_reason = "horizon"
    stop_time = float(scenario.horizon)
    last_status = "not_started"
    progress_started_at = time.perf_counter()
    progress_interval = max(1, n_steps // 100)

    def build_result(
        *,
        final_stop_reason: str,
        final_stop_time: float,
        final_status: str,
    ) -> dict[str, np.ndarray | str | int | float]:
        states_array = np.vstack(states).astype(float)
        controls_array = (
            np.vstack(controls).astype(float)
            if controls
            else np.empty((0, 2), dtype=float)
        )
        result: dict[str, np.ndarray | str | int | float] = {
            "method": method,
            "label": METHOD_LABELS[method],
            "times": np.asarray(times, dtype=float),
            "control_times": np.asarray(control_times, dtype=float),
            "step_times": np.asarray(step_times, dtype=float),
            "states": states_array,
            "controls": controls_array,
            "step_compute_times": np.asarray(step_compute_times, dtype=float),
            "qp_compute_times": np.asarray(qp_compute_times, dtype=float),
            "qp_calls_per_step": np.asarray(qp_calls_per_step, dtype=int),
            "feasible": np.asarray(feasible, dtype=bool),
            "step_statuses": np.asarray(step_statuses, dtype="U160"),
            "qp_event_statuses": np.asarray(qp_event_statuses, dtype="U160"),
            "qp_call_statuses": np.asarray(timing.qp_statuses, dtype="U160"),
            "event_flags": np.asarray(event_flags, dtype=bool),
            "control_event_flags": np.asarray(control_event_flags, dtype=bool),
            "tau_values": np.asarray(tau_values, dtype=float),
            "eta_values": np.asarray(eta_values, dtype=float),
            "h_values": np.asarray(h_values, dtype=float),
            "clearance_values": np.asarray(clearance_values, dtype=float),
            "goal_distance": np.asarray(goal_distance, dtype=float),
            "num_infeasible": int(np.count_nonzero(~np.asarray(feasible, dtype=bool))),
            "num_controller_updates": int(np.count_nonzero(np.asarray(event_flags, dtype=bool))),
            "num_qp_calls": int(np.sum(np.asarray(qp_calls_per_step, dtype=int))),
            "total_controller_compute_time": float(np.sum(np.asarray(step_compute_times, dtype=float))),
            "total_qp_compute_time": float(np.sum(np.asarray(qp_compute_times, dtype=float))),
            "last_status": final_status,
            "stop_reason": final_stop_reason,
            "stop_time": float(final_stop_time),
            "completed_horizon": bool(final_stop_reason == "horizon"),
            "num_applied_steps": int(controls_array.shape[0]),
            "num_attempted_steps": int(len(step_times)),
        }
        if method in TTCBF_METHODS:
            result.update(ttcbf_remainder_diagnostics(states_array, controls_array, scenario))
        return result

    if show_progress:
        print_progress(METHOD_LABELS[method], 0, n_steps, progress_started_at, 0.0)

    if h_values[0] < -COLLISION_H_TOLERANCE:
        if show_progress:
            print_progress(METHOD_LABELS[method], 0, n_steps, progress_started_at, 0.0, force_newline=True)
        return build_result(
            final_stop_reason="collision",
            final_stop_time=0.0,
            final_status="initial_collision",
        )
    if goal_distance[0] <= GOAL_REACHED_TOLERANCE:
        if show_progress:
            print_progress(METHOD_LABELS[method], 0, n_steps, progress_started_at, 0.0, force_newline=True)
        return build_result(
            final_stop_reason="goal_reached",
            final_stop_time=0.0,
            final_status="initial_goal_reached",
        )

    for k in range(n_steps):
        x = states[-1]
        step_time = float(scheduled_times[k])
        solve_now = True
        if method in ("event_triggered_tlc", "event_triggered_atlc"):
            solve_now = k == 0 or outside_event_box(x, event_anchor, scenario)

        qp_calls_before = timing.qp_calls
        qp_status_count_before = len(timing.qp_statuses)
        qp_wall_before = timing.qp_wall_time
        t_start = time.perf_counter()
        if solve_now:
            if method == "tlc":
                result = solve_qp(
                    x,
                    scenario,
                    pointwise_tlc_safety(x, scenario, scenario.zoh_tlc_tau),
                    warm_start=solver_warm_start,
                    timing=timing,
                )
                current_tau = scenario.zoh_tlc_tau
                current_eta = np.nan
            elif method == "event_triggered_tlc":
                result = solve_event_tlc_control(
                    x,
                    scenario,
                    scenario.event_triggered_tlc_tau,
                    warm_start=solver_warm_start,
                    timing=timing,
                )
                event_anchor = x.copy()
                current_tau = scenario.event_triggered_tlc_tau
                current_eta = np.nan
            elif method == "rtlc":
                result = solve_qp(
                    x,
                    scenario,
                    rtlc_safety(x, scenario, scenario.rtlc_tau),
                    warm_start=solver_warm_start,
                    timing=timing,
                )
                current_tau = scenario.rtlc_tau
                current_eta = np.nan
            elif method == "event_triggered_atlc":
                result, current_tau = select_atlc_control(
                    x,
                    scenario,
                    warm_start=solver_warm_start,
                    timing=timing,
                )
                event_anchor = x.copy()
                current_eta = np.nan
            elif method in TTCBF_METHODS:
                adaptive = method.startswith("attcbf")
                result = solve_qp(
                    x,
                    scenario,
                    ttcbf_safety(
                        x,
                        scenario,
                        adaptive=adaptive,
                    ),
                    use_eta=adaptive,
                    warm_start=solver_warm_start,
                    timing=timing,
                )
                current_tau = 2.0 * scenario.dt
                current_eta = result.eta if adaptive else np.nan
            elif method == "pacbf":
                result = solve_pacbf_control(
                    x,
                    pacbf_p1,
                    scenario,
                    warm_start=solver_warm_start,
                    timing=timing,
                )
                current_tau = np.nan
                current_eta = pacbf_p1
                current_aux = result.aux
            elif method == "racbf":
                result = solve_racbf_control(
                    x,
                    racbf_aux_state,
                    scenario,
                    warm_start=solver_warm_start,
                    timing=timing,
                )
                current_tau = np.nan
                current_eta = racbf_aux_state[0]
                current_aux = result.aux
            elif method == "avcbf":
                result = solve_avcbf_control(
                    x,
                    avcbf_aux_state,
                    scenario,
                    warm_start=solver_warm_start,
                    timing=timing,
                )
                current_tau = np.nan
                current_eta = np.nan
                current_aux = result.aux
            elif method == "ct_hocbf":
                result = solve_qp(
                    x,
                    scenario,
                    ct_hocbf_safety(x, scenario),
                    warm_start=solver_warm_start,
                    timing=timing,
                )
                current_tau = np.nan
                current_eta = np.nan
                current_aux = np.nan
            elif method == "dt_hocbf":
                result = solve_dt_hocbf_control(
                    x,
                    scenario,
                    warm_start=solver_warm_start,
                    timing=timing,
                )
                current_tau = np.nan
                current_eta = np.nan
                current_aux = np.nan
            elif method == "adt_hocbf":
                result = solve_adt_hocbf_control(
                    x,
                    adt_hocbf_gamma1,
                    scenario,
                    warm_start=solver_warm_start,
                    timing=timing,
                )
                current_tau = np.nan
                current_eta = result.eta
                current_aux = result.aux
            else:
                raise ValueError(f"unknown method: {method}")
            status = result.status
            if result.feasible:
                held_u = result.u
                solver_warm_start = result.solver_state
            else:
                solver_warm_start = None
        else:
            status = "held"

        new_qp_statuses = timing.qp_statuses[qp_status_count_before:]
        nonoptimal_statuses = [
            qp_status for qp_status in new_qp_statuses if not is_nominal_qp_status(qp_status)
        ]
        step_times.append(step_time)
        step_statuses.append(status)
        qp_event_statuses.append(nonoptimal_statuses[0] if nonoptimal_statuses else "")
        step_compute_times.append(time.perf_counter() - t_start)
        qp_compute_times.append(timing.qp_wall_time - qp_wall_before)
        qp_calls_per_step.append(timing.qp_calls - qp_calls_before)
        event_flags.append(bool(solve_now))
        last_status = status

        if solve_now and not result.feasible:
            feasible.append(False)
            stop_reason = "solver_infeasible"
            stop_time = step_time
            if show_progress:
                print_progress(
                    METHOD_LABELS[method],
                    k,
                    n_steps,
                    progress_started_at,
                    stop_time,
                    force_newline=True,
                )
            break

        feasible.append(True)
        controls.append(np.asarray(held_u, dtype=float).copy())
        control_times.append(step_time)
        control_event_flags.append(bool(solve_now))
        tau_values.append(float(current_tau))
        eta_values.append(float(current_eta))

        if method in ("dt_hocbf", "adt_hocbf"):
            next_state = euler_sample_step(x, held_u, scenario)
        else:
            next_state = rk4_step(x, held_u, scenario.dt)
        if method == "avcbf":
            nu = current_aux if math.isfinite(float(current_aux)) else 0.0
            avcbf_aux_state = np.array(
                [
                    avcbf_aux_state[0] + scenario.dt * avcbf_aux_state[1] + 0.5 * scenario.dt**2 * nu,
                    avcbf_aux_state[1] + scenario.dt * nu,
                ],
                dtype=float,
            )
        elif method == "pacbf":
            nu1 = current_aux if math.isfinite(float(current_aux)) else 0.0
            pacbf_p1 = max(scenario.pacbf_epsilon, float(pacbf_p1 + scenario.dt * nu1))
        elif method == "racbf":
            nu = current_aux if math.isfinite(float(current_aux)) else 0.0
            racbf_aux_state = np.array(
                [
                    racbf_aux_state[0] + scenario.dt * racbf_aux_state[1] + 0.5 * scenario.dt**2 * nu,
                    racbf_aux_state[1] + scenario.dt * nu,
                ],
                dtype=float,
            )
        elif method == "adt_hocbf":
            nu_gamma = current_aux if math.isfinite(float(current_aux)) else 0.0
            adt_hocbf_gamma1 = float(adt_hocbf_gamma1 + nu_gamma)

        states.append(next_state)
        next_time = float(scheduled_times[k + 1])
        times.append(next_time)
        h_next = barrier_terms(next_state, scenario)[0]
        h_values.append(h_next)
        clearance_values.append(
            math.sqrt(max(h_next + scenario.safe_radius**2, 0.0)) - scenario.safe_radius
        )
        goal_distance.append(float(np.linalg.norm(next_state[:2] - np.array(scenario.goal))))

        if h_next < -COLLISION_H_TOLERANCE:
            stop_reason = "collision"
            stop_time = next_time
        elif goal_distance[-1] <= GOAL_REACHED_TOLERANCE:
            stop_reason = "goal_reached"
            stop_time = next_time

        should_stop = stop_reason != "horizon"
        if show_progress and (
            (k + 1) % progress_interval == 0 or k + 1 == n_steps or should_stop
        ):
            print_progress(
                METHOD_LABELS[method],
                k + 1,
                n_steps,
                progress_started_at,
                times[-1],
                force_newline=(k + 1 == n_steps or should_stop),
            )
        if should_stop:
            break

    return build_result(
        final_stop_reason=stop_reason,
        final_stop_time=stop_time,
        final_status=last_status,
    )


def plot_taylor_residuals(
    results: list[dict[str, np.ndarray | str | int | float]],
    scenario: Scenario,
    output: Path,
) -> None:
    diagnostic_results = [
        result
        for result in results
        if "ttcbf_remainder_estimate" in result
        and "ttcbf_remainder_closed_loop_residual" in result
    ]
    if not diagnostic_results:
        return

    with plt.rc_context({"text.usetex": True}):
        fig, ax = new_paper_figure(figsize=(SINGLE_COLUMN_WIDTH, 1.5))
        has_data = False
        inset_series: list[tuple[np.ndarray, np.ndarray, dict[str, object]]] = []
        for result in diagnostic_results:
            control_times = np.asarray(result["control_times"], dtype=float)
            estimate = np.asarray(result["ttcbf_remainder_estimate"], dtype=float)
            closed_loop_residual = np.asarray(result["ttcbf_remainder_closed_loop_residual"], dtype=float)
            color = method_color(result)
            label = result_legend_label(result, scenario)
            label = label.replace(" (our)", "")

            estimate_mask = np.isfinite(estimate)
            if np.any(estimate_mask):
                has_data = True
                ax.plot(
                    control_times[estimate_mask],
                    estimate[estimate_mask],
                    color=color,
                    linestyle=method_linestyle(result),
                    linewidth=method_linewidth(result),
                    zorder=method_zorder(result),
                    label=f"{label} " + r"$\underline{R}_{\mathrm{T}}$",
                )

            closed_loop_residual_mask = np.isfinite(closed_loop_residual)
            if np.any(closed_loop_residual_mask):
                has_data = True
                is_attcbf_label = "aTTCBF" in label
                residual_kwargs = {
                    "color": "black",
                    "alpha": 0.6 if is_attcbf_label else 0.3,
                    "linestyle": method_linestyle(result),
                    "linewidth": method_linewidth(result),
                    "zorder": method_zorder(result),
                }
                ax.plot(
                    control_times[closed_loop_residual_mask],
                    closed_loop_residual[closed_loop_residual_mask],
                    **residual_kwargs,
                    label=f"{label} " + r"$R_{\mathrm{T}}$",
                )
                inset_series.append(
                    (
                        control_times[closed_loop_residual_mask],
                        closed_loop_residual[closed_loop_residual_mask],
                        residual_kwargs,
                    )
                )

        if not has_data:
            plt.close(fig)
            return

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(
            r"T. res. $R_{\mathrm{T}}$, $\underline{R}_{\mathrm{T}}$ [m$^2$]"
        )
        ax.set_xlim(0.0, scenario.horizon)
        ax.set_ylim(-0.05 , 0.05)
        ax.grid(True, alpha=0.20)
        if inset_series:
            axins = inset_axes(
                ax,
                width="30%",
                height="30%",
                loc="upper left",
                bbox_to_anchor=(0.2, -0.06, 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0.0,
            )
            for times, residuals, kwargs in inset_series:
                axins.plot(times, residuals, **kwargs)
            axins.set_xlim(3.2, 3.6)
            axins.set_ylim(-0.001, 0.004)
            axins.set_xticks([3.2, 3.4, 3.6])
            axins.set_yticks([-0.001, 0.0, 0.002, 0.004])
            axins.tick_params(axis="both", labelsize=5.5, length=1.8, pad=1.0)
            axins.grid(True, alpha=0.20)
            ax.annotate(
                "",
                xy=(3.5, 0.001),
                xycoords="data",
                xytext=(4.3, 0.013),
                textcoords="data",
                arrowprops={"arrowstyle": "->", "linewidth": 1.0, "color": "black"},
            )
        add_series_legend(ax, loc="upper right")
        save_paper_figure(fig, output)


def timing_metrics(
    result: dict[str, np.ndarray | str | int | float],
    scenario: Scenario,
) -> dict[str, float | int]:
    step_times = np.asarray(result["step_compute_times"], dtype=float)
    update_mask = np.asarray(result["event_flags"], dtype=bool)
    step_clock = np.asarray(result.get("step_times", []), dtype=float)
    update_times = step_times[update_mask]
    qp_times = np.asarray(result["qp_compute_times"], dtype=float)
    qp_calls = np.asarray(result["qp_calls_per_step"], dtype=int)

    update_indices = np.flatnonzero(update_mask)
    if step_clock.size == update_mask.size:
        update_clock = step_clock[update_indices]
        inter_update_intervals = np.diff(update_clock)
    else:
        inter_update_intervals = np.diff(update_indices) * scenario.dt
    num_steps = int(step_times.size)
    num_updates = int(update_mask.sum())
    num_qp_calls = int(qp_calls.sum())

    total_controller_time = float(np.sum(step_times))
    total_qp_time = float(np.sum(qp_times))
    simulated_duration = result_duration(result)
    real_time_factor = (
        float(simulated_duration / total_controller_time)
        if total_controller_time > 0.0
        else math.inf
    )
    mean_qp_call_time_ms = (
        1000.0 * total_qp_time / num_qp_calls
        if num_qp_calls > 0
        else 0.0
    )

    return {
        "num_steps": num_steps,
        "num_controller_updates": num_updates,
        "controller_update_ratio": float(num_updates / num_steps) if num_steps else 0.0,
        "num_qp_calls": num_qp_calls,
        "mean_qp_calls_per_update": float(num_qp_calls / num_updates) if num_updates else 0.0,
        "mean_step_compute_ms": float(1000.0 * np.mean(step_times)) if num_steps else 0.0,
        "mean_update_compute_ms": float(1000.0 * np.mean(update_times)) if update_times.size else 0.0,
        "median_update_compute_ms": float(1000.0 * np.median(update_times)) if update_times.size else 0.0,
        "p95_update_compute_ms": float(1000.0 * np.percentile(update_times, 95)) if update_times.size else 0.0,
        "max_update_compute_ms": float(1000.0 * np.max(update_times)) if update_times.size else 0.0,
        "mean_qp_call_compute_ms": mean_qp_call_time_ms,
        "total_controller_compute_s": total_controller_time,
        "total_qp_compute_s": total_qp_time,
        "controller_real_time_factor": real_time_factor,
        "controller_compute_load": (
            float(total_controller_time / simulated_duration)
            if simulated_duration > 0.0
            else (math.inf if total_controller_time > 0.0 else 0.0)
        ),
        "mean_inter_update_interval_s": (
            float(np.mean(inter_update_intervals)) if inter_update_intervals.size else 0.0
        ),
        "max_inter_update_interval_s": (
            float(np.max(inter_update_intervals)) if inter_update_intervals.size else 0.0
        ),
    }


def control_effort_components(
    result: dict[str, np.ndarray | str | int | float],
    scenario: Scenario,
) -> dict[str, float]:
    controls = np.asarray(result["controls"], dtype=float)
    if controls.size == 0:
        return {
            "control_magnitude_raw": math.nan,
            "control_rate_raw": math.nan,
            "control_effort_raw": math.nan,
        }

    omega = controls[:, 0]
    accel = controls[:, 1]
    omega_scale = max(abs(scenario.omega_min), abs(scenario.omega_max), 1.0e-12)
    accel_positive_scale = max(abs(scenario.accel_max), 1.0e-12)
    accel_negative_scale = max(abs(scenario.accel_min), 1.0e-12)
    accel_scale = np.where(accel >= 0.0, accel_positive_scale, accel_negative_scale)
    magnitude_terms = 0.5 * ((omega / omega_scale) ** 2 + (accel / accel_scale) ** 2)
    finite_magnitude_terms = magnitude_terms[np.isfinite(magnitude_terms)]
    magnitude_raw = (
        float(math.sqrt(float(np.mean(finite_magnitude_terms))))
        if finite_magnitude_terms.size
        else math.nan
    )

    if controls.shape[0] > 1 and scenario.dt > 0.0:
        control_rates = np.diff(controls, axis=0) / scenario.dt
        omega_rate_scale = max(abs(scenario.omega_rate_max), 1.0e-12)
        accel_rate_scale = max(abs(scenario.accel_rate_max), 1.0e-12)
        rate_terms = 0.5 * (
            (control_rates[:, 0] / omega_rate_scale) ** 2
            + (control_rates[:, 1] / accel_rate_scale) ** 2
        )
        finite_rate_terms = rate_terms[np.isfinite(rate_terms)]
        rate_raw = (
            float(math.sqrt(float(np.mean(finite_rate_terms))))
            if finite_rate_terms.size
            else math.nan
        )
    else:
        rate_raw = 0.0

    combined_terms = [
        0.5 * value**2
        for value in (magnitude_raw, rate_raw)
        if math.isfinite(value)
    ]
    combined_raw = (
        float(math.sqrt(float(np.sum(combined_terms))))
        if combined_terms
        else math.nan
    )
    return {
        "control_magnitude_raw": magnitude_raw,
        "control_rate_raw": rate_raw,
        "control_effort_raw": combined_raw,
    }


def controller_runtime_full_horizon(
    result: dict[str, np.ndarray | str | int | float],
) -> float:
    step_times = np.asarray(result["step_compute_times"], dtype=float)
    finite_step_times = np.where(np.isfinite(step_times), step_times, 0.0)
    if finite_step_times.size == 0:
        return 0.0
    return float(np.sum(finite_step_times))


def tuning_parameter_count(method: str) -> int:
    return len(TUNING_PARAMETER_FIELDS.get(method, ()))


def composite_metric_value(
    metric: str,
    result: dict[str, np.ndarray | str | int | float],
    scenario: Scenario,
) -> float:
    method = str(result["method"])
    if metric == "tuning":
        return float(tuning_parameter_count(method))
    if metric == "runtime":
        return controller_runtime_full_horizon(result)
    if metric == "min_h":
        return float(np.min(np.asarray(result["h_values"], dtype=float)))
    if metric == "effort":
        return float(control_effort_components(result, scenario)["control_effort_raw"])
    raise ValueError(f"unknown composite result metric: {metric}")


def composite_scale_label(value: float) -> str:
    if not math.isfinite(value):
        return ""
    if abs(value) >= 100.0:
        return f"{value:.0f}"
    if abs(value) >= 10.0:
        return f"{value:.1f}"
    if abs(value) >= 1.0:
        return f"{value:.2f}"
    if value == 0.0:
        return "0"
    decimals = max(0, 1 - math.floor(math.log10(abs(value))))
    return f"{value:.{decimals}f}"


def composite_bar_label(metric: str, value: float) -> str:
    if not math.isfinite(value):
        return ""
    if metric == "tuning":
        return f"{value:.0f}"
    return composite_scale_label(value)


def save_composite_results_plot(
    results: list[dict[str, np.ndarray | str | int | float]],
    scenario: Scenario,
    out_dir: Path,
) -> None:
    if not results:
        return

    metrics = tuple(
        metric
        for metric in COMPOSITE_RESULT_METRICS
        if metric in COMPOSITE_RESULT_METRIC_SPECS
    )
    if not metrics:
        return

    positions = np.arange(len(results))
    labels = [str(result["label"]) for result in results]
    bar_colors = [method_color(result) for result in results]
    fig, ax = new_paper_figure(
        figsize=(SINGLE_COLUMN_WIDTH, 1.2),
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.18, right=0.90, bottom=0.24, top=0.98)

    x_min = -0.5
    x_max = len(results) - 0.5
    bar_width = 0.70
    frame_style = {
        "color": "0.82",
        "linewidth": 0.45,
        "linestyle": (0.0, (1.0, 1.6)),
        "zorder": 0,
    }

    ytick_positions = []
    ytick_labels = []
    for layer_index, metric in enumerate(metrics):
        spec = COMPOSITE_RESULT_METRIC_SPECS[metric]
        values = np.asarray(
            [composite_metric_value(metric, result, scenario) for result in results],
            dtype=float,
        )
        plot_values = np.where(np.isfinite(values), values, 0.0)
        if metric == "min_h":
            plot_values = np.maximum(plot_values, 0.0)
        max_value = float(np.max(plot_values)) if plot_values.size else 0.0
        if not math.isfinite(max_value) or max_value < 0.0:
            max_value = 0.0
        denominator = max(max_value, 1.0e-12)

        layer_bottom = float(layer_index)
        layer_top = float(layer_index + 1)
        normalized = np.clip(plot_values / denominator, 0.0, 1.0)
        ax.bar(
            positions,
            normalized,
            width=bar_width,
            bottom=layer_bottom,
            color=bar_colors,
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )
        ax.hlines(
            [layer_bottom, layer_top],
            x_min,
            x_max,
            **frame_style,
        )
        ax.vlines(
            [x_min, x_max],
            layer_bottom,
            layer_top,
            **frame_style,
        )
        for x_pos, value, bar_height in zip(positions, values, normalized):
            label = composite_bar_label(metric, float(value))
            if not label:
                continue
            bar_top = layer_bottom + float(bar_height)
            if bar_height > 0.8:
                text_y = bar_top
                text_offset = -1.5
                text_va = "top"
                text_color = "white"
            else:
                text_y = bar_top
                text_offset = 1.5
                text_va = "bottom"
                text_color = "0.20"
            ax.annotate(
                label,
                xy=(float(x_pos), text_y),
                xytext=(0.0, text_offset),
                textcoords="offset points",
                ha="center",
                va=text_va,
                fontsize=5.8,
                color=text_color,
                clip_on=False,
                zorder=3,
            )
        ytick_positions.append(layer_bottom + 0.5)
        ytick_labels.append(str(spec["label"]))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.0, float(len(metrics)))
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35.0, ha="right", fontsize=5.5)  # Custom font size
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, rotation=90.0, va="center")
    ax.set_ylabel("")
    ax.tick_params(axis="y", length=0.0)
    ax.tick_params(axis="x", pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save_paper_figure(fig, out_dir / "fig_composite_results.pdf")


def save_plots(results: list[dict[str, np.ndarray | str | int | float]], scenario: Scenario, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = new_paper_figure(figsize=(SINGLE_COLUMN_WIDTH, 1.0), constrained_layout=False)
    h_lines = []
    for result in results:
        (line,) = ax.plot(
            result["times"],
            result["h_values"],
            label=result_legend_label(result, scenario),
            **method_plot_kwargs(result),
        )
        h_lines.append(line)
    add_horizontal_reference(ax, 0.0, r"$h=0$")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"CBF $h$ [m$^2$]")
    ax.set_xlim(0.0, scenario.horizon)
    ax.grid(True, alpha=0.20)
    if should_add_method_legend(results):
        add_series_legend(ax)
    if scenario.horizon >= 8.0:
        axins = inset_axes(
            ax,
            width="35%",
            height="55%",
            loc="lower left",
            bbox_to_anchor=(0.24, 0.4, 1, 1),   # [x, y, width, height] in axes coords
            bbox_transform=ax.transAxes,
            borderpad=0.0,
        )
        for line in h_lines:
            axins.plot(
                line.get_xdata(),
                line.get_ydata(),
                color=line.get_color(),
                linestyle=line.get_linestyle(),
                linewidth=line.get_linewidth(),
            )
        axins.axhline(
            0.0,
            color="black",
            linewidth=REFERENCE_LINEWIDTH,
            linestyle="--",
            label="_nolegend_",
        )
        axins.set_xlim(4.0, 7.0)
        axins.set_ylim(-0.2, 1.0)
        axins.tick_params(labelleft=True, labelbottom=True)
        axins.set_xticks([4.0, 5.0, 6.0, 7.0])
        axins.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axins.grid(True, alpha=0.20)
        mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")
    save_paper_figure(fig, out_dir / "fig_cbf_h.pdf")

    fig, ax = new_paper_figure(figsize=(SINGLE_COLUMN_WIDTH, 1.6))
    for result in results:
        states = result["states"]
        ax.plot(
            result["times"],
            states[:, 3],
            label=result_legend_label(result, scenario),
            **method_plot_kwargs(result),
        )
    add_horizontal_reference(ax, scenario.v_min, r"$v_{\mathrm{min}}$")
    add_horizontal_reference(ax, scenario.v_max, r"$v_{\mathrm{max}}$", va="top", dy_points=-2.0)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Speed $v$ [m/s]")
    ax.set_xlim(0.0, scenario.horizon)
    ax.grid(True, alpha=0.20)
    if should_add_method_legend(results):
        add_series_legend(ax)
    save_paper_figure(fig, out_dir / "fig_speed.pdf")

    fig, (ax_accel, ax_steer) = new_paper_figure(
        PLOT_TWO_PANEL_SIZE,
        2,
        1,
        sharex=True,
    )
    for result in results:
        controls = result["controls"]
        ax_accel.plot(
            result["control_times"],
            controls[:, 1],
            label=result_legend_label(result, scenario),
            **method_plot_kwargs(result),
        )
        ax_steer.plot(
            result["control_times"],
            controls[:, 0],
            label=result_legend_label(result, scenario),
            **method_plot_kwargs(result),
        )
    add_horizontal_reference(ax_accel, scenario.accel_min, r"$u_{2,\mathrm{min}}$")
    add_horizontal_reference(ax_accel, scenario.accel_max, r"$u_{2,\mathrm{max}}$", va="top", dy_points=-2.0)
    add_horizontal_reference(ax_steer, scenario.omega_min, r"$u_{1,\mathrm{min}}$")
    add_horizontal_reference(ax_steer, scenario.omega_max, r"$u_{1,\mathrm{max}}$", va="top", dy_points=-2.0)
    add_min_bound_touch_annotation(
        ax_accel,
        results,
        control_index=1,
        bound=scenario.accel_min,
        text="Input bound active",
        scenario=scenario,
    )
    add_min_bound_touch_annotation(
        ax_steer,
        results,
        control_index=0,
        bound=scenario.omega_min,
        text="Input bound active",
        scenario=scenario,
    )
    ax_accel.set_ylabel(r"$u_2$ [m/s$^2$]")
    ax_steer.set_ylabel(r"$u_1$ [rad/s]")
    ax_accel.set_xlim(0.0, scenario.horizon)
    ax_accel.set_ylim(scenario.accel_min - 0.2, scenario.accel_max + 0.2)
    ax_steer.set_ylim(scenario.omega_min - 0.4, scenario.omega_max + 0.4)
    ax_steer.set_xlim(0.0, scenario.horizon)
    ax_accel.grid(True, alpha=0.20)
    ax_steer.grid(True, alpha=0.20)
    if should_add_method_legend(results):
        add_series_legend(ax_accel)
    ax_steer.set_xlabel("Time [s]")
    save_paper_figure(fig, out_dir / "fig_acceleration_steering_rate.pdf")

    plot_taylor_residuals(results, scenario, out_dir / "fig_taylor_residuals.pdf")

    save_trajectory_plot(results, scenario, out_dir)


def save_summary(
    results: list[dict[str, np.ndarray | str | int | float]],
    scenario: Scenario,
    out_dir: Path,
) -> None:
    rows = []
    for result in results:
        controls = np.asarray(result["controls"], dtype=float)
        states = np.asarray(result["states"], dtype=float)
        h_values = np.asarray(result["h_values"], dtype=float)
        metrics = timing_metrics(result, scenario)
        rows.append(
            {
                "method": result["method"],
                "label": result["label"],
                "stop_reason": result_stop_reason(result),
                "stop_time_s": result_stop_time(result),
                "completed_horizon": bool(result.get("completed_horizon", False)),
                "num_applied_steps": int(result.get("num_applied_steps", controls.shape[0])),
                "num_attempted_steps": int(result.get("num_attempted_steps", metrics["num_steps"])),
                "min_h": float(np.min(h_values)),
                "min_clearance_m": float(np.min(result["clearance_values"])),
                "final_goal_distance_m": final_goal_distance(result),
                "mean_speed_mps": column_mean_or_nan(states[:, 3]),
                "mean_abs_steering_rate": (
                    column_mean_or_nan(np.abs(controls[:, 0]))
                    if controls.shape[0]
                    else math.nan
                ),
                "mean_abs_acceleration": (
                    column_mean_or_nan(np.abs(controls[:, 1]))
                    if controls.shape[0]
                    else math.nan
                ),
                "max_abs_steering_rate": (
                    column_max_abs_or_nan(controls[:, 0])
                    if controls.shape[0]
                    else math.nan
                ),
                "max_abs_acceleration": (
                    column_max_abs_or_nan(controls[:, 1])
                    if controls.shape[0]
                    else math.nan
                ),
                **metrics,
                "num_infeasible": int(result["num_infeasible"]),
                "step_status_counts": status_counts_json(np.asarray(result["step_statuses"])),
                "qp_status_counts": status_counts_json(np.asarray(result["qp_call_statuses"])),
            }
        )

    with (out_dir / "summary.csv").open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        out_dir / "simulation_logs.npz",
        **{
            f"{result['method']}_{key}": value
            for result in results
            for key, value in result.items()
            if isinstance(value, np.ndarray)
        },
    )


def inclusive_negative_grid(end: float, step: float, name: str) -> tuple[float, ...]:
    if end >= 0.0:
        raise ValueError(f"{name}_end must be negative")
    if step <= 0.0:
        raise ValueError(f"{name}_step must be positive")

    count_float = abs(end) / step
    count = int(round(count_float))
    if count < 1 or not math.isclose(count_float, count, rel_tol=1.0e-9, abs_tol=1.0e-9):
        raise ValueError(
            f"{name}_end={end:g} is not reachable from -{step:g} "
            f"with step size {step:g}"
        )
    decimals = max(0, -int(math.floor(math.log10(step))) + 2) if step < 1.0 else 12
    return tuple(round(-step * index, decimals) for index in range(1, count + 1))


def inclusive_positive_grid(end: float, step: float, name: str) -> tuple[float, ...]:
    """Create a grid of positive values from step to end (inclusive).

    This is the positive counterpart of ``inclusive_negative_grid`` and is used
    for positive upper-bound sweeps.
    """
    if end <= 0.0:
        raise ValueError(f"{name}_end must be positive")
    if step <= 0.0:
        raise ValueError(f"{name}_step must be positive")

    count_float = end / step
    count = int(round(count_float))
    if count < 1 or not math.isclose(count_float, count, rel_tol=1.0e-9, abs_tol=1.0e-9):
        raise ValueError(
            f"{name}_end={end:g} is not reachable from {step:g} "
            f"with step size {step:g}"
        )
    decimals = max(0, -int(math.floor(math.log10(step))) + 2) if step < 1.0 else 12
    return tuple(round(step * index, decimals) for index in range(1, count + 1))


def auto_sweep_omega_fields(
    scenario: Scenario,
    *,
    tolerance: float = 1.0e-12,
) -> tuple[str, ...]:
    start = np.asarray(scenario.start[:2], dtype=float)
    goal = np.asarray(scenario.goal, dtype=float)
    obstacle = np.asarray(scenario.obstacle, dtype=float)
    goal_vector = goal - start
    obstacle_vector = obstacle - start
    cross = float(goal_vector[0] * obstacle_vector[1] - goal_vector[1] * obstacle_vector[0])
    if cross > tolerance:
        return ("omega_min",)
    if cross < -tolerance:
        return ("omega_max",)
    return ("omega_min", "omega_max")


def omega_sweep_values(
    omega_field: str,
    omega_end: float,
    omega_step: float,
) -> tuple[float, ...]:
    omega_mag = abs(omega_end)
    if omega_field == "omega_min":
        return tuple(
            -value
            for value in inclusive_positive_grid(omega_mag, omega_step, "omega_mag")
        )
    if omega_field == "omega_max":
        return inclusive_positive_grid(omega_mag, omega_step, "omega_mag")
    raise ValueError(f"unknown omega sweep field: {omega_field}")


def rollout_statistics_row(
    result: dict[str, np.ndarray | str | int | float],
    scenario: Scenario,
    combo_index: int,
    method: str,
    omega_sweep_field: str,
) -> dict[str, object]:
    controls = np.asarray(result["controls"], dtype=float)
    h_values = np.asarray(result["h_values"], dtype=float)
    metrics = timing_metrics(result, scenario)
    effort_components = control_effort_components(result, scenario)
    is_collision = first_collision_point(result, scenario) is not None
    num_infeasible = int(result["num_infeasible"])
    stop_reason = result_stop_reason(result)

    row: dict[str, object] = {
        "combo_index": int(combo_index),
        "accel_min": float(scenario.accel_min),
        "accel_max": float(scenario.accel_max),
        "omega_min": float(scenario.omega_min),
        "omega_max": float(scenario.omega_max),
        "omega_sweep_field": omega_sweep_field,
        "method": method,
        "label": result["label"],
        "is_valid": reached_goal(result),
        "is_collision": bool(is_collision),
        "is_feasible": bool(num_infeasible == 0 and stop_reason != "solver_infeasible"),
        "stop_reason": stop_reason,
        "stop_time_s": result_stop_time(result),
        "task_completion_time_s": task_completion_time(result),
        "final_goal_distance_m": final_goal_distance(result),
        "min_h": float(np.min(h_values)),
        "min_clearance_m": float(np.min(result["clearance_values"])),
        "tuning_parameter_count": tuning_parameter_count(method),
        "tuning_parameter_fields": ";".join(TUNING_PARAMETER_FIELDS.get(method, ())),
        "total_controller_compute_s": float(metrics["total_controller_compute_s"]),
        "mean_step_compute_ms": float(metrics["mean_step_compute_ms"]),
        "mean_update_compute_ms": float(metrics["mean_update_compute_ms"]),
        "mean_qp_call_compute_ms": float(metrics["mean_qp_call_compute_ms"]),
        "controller_real_time_factor": float(metrics["controller_real_time_factor"]),
        "num_qp_calls": int(metrics["num_qp_calls"]),
        "num_controller_updates": int(metrics["num_controller_updates"]),
        "control_effort_raw": float(effort_components["control_effort_raw"]),
        "control_magnitude_raw": float(effort_components["control_magnitude_raw"]),
        "control_rate_raw": float(effort_components["control_rate_raw"]),
        "mean_abs_steering_rate": (
            column_mean_or_nan(np.abs(controls[:, 0])) if controls.shape[0] else math.nan
        ),
        "mean_abs_acceleration": (
            column_mean_or_nan(np.abs(controls[:, 1])) if controls.shape[0] else math.nan
        ),
        "max_abs_steering_rate": (
            column_max_abs_or_nan(controls[:, 0]) if controls.shape[0] else math.nan
        ),
        "max_abs_acceleration": (
            column_max_abs_or_nan(controls[:, 1]) if controls.shape[0] else math.nan
        ),
        "num_infeasible": num_infeasible,
        "step_status_counts": status_counts_json(np.asarray(result["step_statuses"])),
        "qp_status_counts": status_counts_json(np.asarray(result["qp_call_statuses"])),
    }
    return row


def finite_mean_min_max(values: list[object]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": float(np.mean(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def grid_sweep_method_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary_rows = []
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        num_rollouts = len(method_rows)
        num_valid = sum(1 for row in method_rows if bool(row["is_valid"]))
        num_collision = sum(1 for row in method_rows if bool(row["is_collision"]))
        num_infeasible_rollouts = sum(1 for row in method_rows if not bool(row["is_feasible"]))
        num_timeout = sum(1 for row in method_rows if row["stop_reason"] == "horizon")
        summary: dict[str, object] = {
            "method": method,
            "label": METHOD_LABELS[method],
            "num_rollouts": num_rollouts,
            "num_valid": num_valid,
            "valid_rate": float(num_valid / num_rollouts) if num_rollouts else math.nan,
            "num_collision": num_collision,
            "collision_rate": float(num_collision / num_rollouts) if num_rollouts else math.nan,
            "num_infeasible_rollouts": num_infeasible_rollouts,
            "infeasible_rate": (
                float(num_infeasible_rollouts / num_rollouts) if num_rollouts else math.nan
            ),
            "Timeout": num_timeout,
            "timeout_rate": float(num_timeout / num_rollouts) if num_rollouts else math.nan,
        }
        for key in (
            "min_h",
            "task_completion_time_s",
            "total_controller_compute_s",
            "control_effort_raw",
        ):
            stats = finite_mean_min_max([row[key] for row in method_rows])
            summary[f"{key}_mean"] = stats["mean"]
            summary[f"{key}_min"] = stats["min"]
            summary[f"{key}_max"] = stats["max"]
        summary_rows.append(summary)
    return summary_rows


def new_trajectory_figure(scenario: Scenario) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = new_paper_figure(PLOT_TRAJECTORY_SIZE, constrained_layout=False)
    ax.add_patch(
        plt.Circle(
            scenario.obstacle,
            scenario.safe_radius,
            edgecolor="tab:red",
            facecolor=to_rgba("tab:red", 0.4),
            linestyle="-",
            linewidth=0.5,
            zorder=0,
        )
    )
    ax.scatter(
        [scenario.start[0]],
        [scenario.start[1]],
        marker="^",
        s=START_GOAL_MARKER_SIZE * 0.5,
        color="gold",
        edgecolor="black",
        zorder=40,
        label="_nolegend_",
    )
    ax.scatter(
        [scenario.goal[0]],
        [scenario.goal[1]],
        marker="*",
        s=START_GOAL_MARKER_SIZE,
        color="gold",
        edgecolor="black",
        zorder=40,
        label="_nolegend_",
    )
    ax.annotate(
        "Start",
        xy=(scenario.start[0], scenario.start[1]),
        xytext=(0, -3),
        color="black",
        textcoords="offset points",
        ha="center",
        va="top",
    )
    ax.annotate(
        "Goal",
        xy=scenario.goal,
        xytext=(0, -3),
        color="black",
        textcoords="offset points",
        ha="center",
        va="top",
    )
    ax.text(
        scenario.obstacle[0],
        scenario.obstacle[1],
        "Obstacle",
        ha="center",
        va="center",
        color="black",
        zorder=30,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylim(-5.0, 1.25)
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    ax.grid(True, alpha=0.20)
    return fig, ax


def add_trajectory_legend(
    fig: plt.Figure,
    ax: plt.Axes,
    results: list[dict[str, np.ndarray | str | int | float]],
) -> None:
    method_handles, method_labels = ax.get_legend_handles_labels()
    if method_handles and should_add_method_legend(results):
        fig.legend(
            method_handles,
            method_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.74),
            ncol=legend_column_count(list(method_labels)),
            borderaxespad=0.0,
            columnspacing=0.8,
            handlelength=1.8,
        )


def add_trajectory_inset(ax: plt.Axes, scenario: Scenario) -> plt.Axes:
    axins = inset_axes(
        ax,
        width="48%",
        height="95%",
        loc="lower left",
        bbox_to_anchor=(0.50, -0.18, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    axins.add_patch(
        plt.Circle(
            scenario.obstacle,
            scenario.safe_radius,
            edgecolor="tab:red",
            facecolor=to_rgba("tab:red", 0.4),
            linestyle="-",
            linewidth=0.5,
            zorder=0,
        )
    )
    axins.set_xlim(3.7, 5.5)
    axins.set_ylim(-1.5, -0.6)
    axins.set_aspect("equal")
    axins.set_xticks([4.0, 5.0])
    axins.set_yticks([-1.4, -1.2, -1.0, -0.8, -0.6])
    axins.tick_params(axis="both", labelsize=5.5, length=1.8, pad=1.0)
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
    return axins


def save_trajectory_plot(
    results: list[dict[str, np.ndarray | str | int | float]],
    scenario: Scenario,
    out_dir: Path,
) -> None:
    """Save the XY trajectory figure used by standard runs and grid sweeps."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = new_trajectory_figure(scenario)
    trajectory_lines = []
    for result in results:
        states = np.asarray(result["states"], dtype=float)
        (line,) = ax.plot(
            states[:, 0],
            states[:, 1],
            label=result_legend_label(result, scenario),
            **method_plot_kwargs(result),
        )
        trajectory_lines.append(line)
        draw_trajectory_status_markers(ax, result, scenario)
    add_trajectory_legend(fig, ax, results)

    axins = add_trajectory_inset(ax, scenario)
    for line in trajectory_lines:
        axins.plot(
            line.get_xdata(),
            line.get_ydata(),
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
            zorder=line.get_zorder(),
        )
    for result in results:
        draw_trajectory_status_markers(axins, result, scenario)
    annotate_trajectory_event_points(
        axins,
        representative_trajectory_event_points(results, scenario),
    )
    save_paper_figure(fig, out_dir / "fig_xy_trajectories.pdf")


def hidden_trajectory_status_artists(
    ax: plt.Axes,
    result: dict[str, np.ndarray | str | int | float],
    scenario: Scenario,
) -> list[plt.Artist]:
    existing_artists = set(ax.get_children())
    draw_trajectory_status_markers(ax, result, scenario)
    artists = [artist for artist in ax.get_children() if artist not in existing_artists]
    for artist in artists:
        artist.set_visible(False)
    return artists


def heading_marker_path(theta: float) -> MatplotlibPath:
    marker = MatplotlibPath(
        np.array(
            [
                [1.0, 0.0],
                [-0.65, 0.55],
                [-0.65, -0.55],
                [1.0, 0.0],
            ],
            dtype=float,
        ),
        closed=True,
    )
    return marker.transformed(Affine2D().rotate(theta))


def save_trajectory_video(
    results: list[dict[str, np.ndarray | str | int | float]],
    scenario: Scenario,
    out_dir: Path,
) -> Path:
    """Animate fresh or cache-loaded trajectories and save an H.264 MP4."""
    if not results:
        raise ValueError("cannot export a trajectory video without simulation results")
    if scenario.dt <= 0.0:
        raise ValueError("Scenario.dt must be positive to export a trajectory video")
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError(
            "FFmpeg is required for --save-video. Install it with "
            "'brew install ffmpeg' on macOS or 'sudo apt-get install ffmpeg' on Ubuntu."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "video_xy_trajectories.mp4"
    fig, ax = new_trajectory_figure(scenario)
    if not should_add_method_legend(results):
        fig.set_size_inches(*PLOT_TRAJECTORY_VIDEO_SIZE)

    main_lines: list[Line2D] = []
    inset_lines: list[Line2D] = []
    main_vehicles: list[Line2D] = []
    inset_vehicles: list[Line2D] = []
    status_artists: list[list[plt.Artist]] = []
    result_states: list[np.ndarray] = []
    result_times: list[np.ndarray] = []

    for result in results:
        states = np.asarray(result["states"], dtype=float)
        times = np.asarray(result["times"], dtype=float)
        n_values = min(states.shape[0], times.size)
        if n_values == 0:
            raise ValueError(f"{result['method']} has no trajectory samples")
        states = states[:n_values]
        times = times[:n_values]
        result_states.append(states)
        result_times.append(times)

        (line,) = ax.plot(
            [],
            [],
            label=result_legend_label(result, scenario),
            **method_plot_kwargs(result),
        )
        main_lines.append(line)
        color = method_color(result)
        (vehicle,) = ax.plot(
            [],
            [],
            linestyle="None",
            marker=heading_marker_path(float(states[0, 2])),
            markersize=VIDEO_VEHICLE_MARKER_SIZE,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.4,
            zorder=35,
            label="_nolegend_",
        )
        main_vehicles.append(vehicle)
        status_artists.append(hidden_trajectory_status_artists(ax, result, scenario))

    add_trajectory_legend(fig, ax, results)
    axins = add_trajectory_inset(ax, scenario)
    for result, states in zip(results, result_states, strict=True):
        (line,) = axins.plot([], [], **method_plot_kwargs(result))
        inset_lines.append(line)
        color = method_color(result)
        (vehicle,) = axins.plot(
            [],
            [],
            linestyle="None",
            marker=heading_marker_path(float(states[0, 2])),
            markersize=VIDEO_VEHICLE_MARKER_SIZE,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.4,
            zorder=35,
            label="_nolegend_",
        )
        inset_vehicles.append(vehicle)

    inset_status_artists = [
        hidden_trajectory_status_artists(axins, result, scenario)
        for result in results
    ]
    event_data = representative_trajectory_events(results, scenario)
    annotations = annotate_trajectory_event_points(
        axins,
        {label: point for label, (point, _) in event_data.items()},
    )
    for annotation in annotations.values():
        annotation.set_visible(False)

    max_time = max(float(times[-1]) for times in result_times)
    frame_times = np.arange(0.0, max_time + 0.5 * scenario.dt, scenario.dt)
    final_hold_frames = max(1, int(round(VIDEO_FINAL_HOLD_SECONDS / scenario.dt)))
    frame_times = np.concatenate(
        (frame_times, np.full(final_hold_frames, max_time, dtype=float))
    )

    all_status_artists = [
        *[artist for group in status_artists for artist in group],
        *[artist for group in inset_status_artists for artist in group],
    ]
    animated_artists: list[plt.Artist] = [
        *main_lines,
        *inset_lines,
        *main_vehicles,
        *inset_vehicles,
        *all_status_artists,
        *annotations.values(),
    ]

    def update(frame_time: float) -> list[plt.Artist]:
        for index, (result, states, times) in enumerate(
            zip(results, result_states, result_times, strict=True)
        ):
            state_index = int(np.searchsorted(times, frame_time, side="right") - 1)
            state_index = max(0, min(state_index, states.shape[0] - 1))
            visible_states = states[: state_index + 1]
            for line in (main_lines[index], inset_lines[index]):
                line.set_data(visible_states[:, 0], visible_states[:, 1])

            current_state = states[state_index]
            marker = heading_marker_path(float(current_state[2]))
            for vehicle in (main_vehicles[index], inset_vehicles[index]):
                vehicle.set_data([current_state[0]], [current_state[1]])
                vehicle.set_marker(marker)

            show_status = frame_time >= result_stop_time(result) - 0.5 * scenario.dt
            for artist in (*status_artists[index], *inset_status_artists[index]):
                artist.set_visible(show_status)

        for label, annotation in annotations.items():
            annotation.set_visible(
                frame_time >= event_data[label][1] - 0.5 * scenario.dt
            )
        return animated_artists

    trajectory_animation = animation.FuncAnimation(
        fig,
        update,
        frames=frame_times,
        interval=1000.0 * scenario.dt,
        blit=False,
        repeat=False,
    )
    writer = animation.FFMpegWriter(
        fps=1.0 / scenario.dt,
        codec="libx264",
        bitrate=2400,
        metadata={"title": "TTCBF trajectory comparison"},
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    try:
        trajectory_animation.save(output, writer=writer, dpi=VIDEO_DPI)
    finally:
        plt.close(fig)
    return output


def run_grid_sweep(args: argparse.Namespace, base_scenario: Scenario, out_dir: Path) -> None:
    accel_min_values = inclusive_negative_grid(
        args.accel_min_end,
        args.accel_min_step,
        "accel_min",
    )
    omega_sweep_fields = (
        auto_sweep_omega_fields(base_scenario)
        if args.is_auto_sweep
        else ("omega_min", "omega_max")
    )
    omega_field_values = {
        omega_field: omega_sweep_values(omega_field, args.omega_min_end, args.omega_min_step)
        for omega_field in omega_sweep_fields
    }
    grid_iterable = tuple(
        (accel_min, omega_field, omega_value)
        for accel_min in accel_min_values
        for omega_field in omega_sweep_fields
        for omega_value in omega_field_values[omega_field]
    )

    result_methods = (
        tuple(args.methods)
        if args.methods is not None
        else ADAPTIVE_GRID_SWEEP_METHODS
    )
    forced_fresh_methods = set(args.methods or ())
    total_combos = len(grid_iterable)
    total_rollouts = total_combos * len(result_methods)
    rows: list[dict[str, object]] = []
    started_at = time.perf_counter()
    completed = 0

    sweep_desc = "auto" if args.is_auto_sweep else "manual"
    omega_desc = ", ".join(
        f"{omega_field} ({len(omega_field_values[omega_field])} values)"
        for omega_field in omega_sweep_fields
    )
    print(
        f"Running static obstacle grid sweep ({sweep_desc} omega sweep): "
        f"{len(accel_min_values)} accel_min values x "
        f"{omega_desc} x "
        f"{len(result_methods)} methods = {total_rollouts} rollouts.",
        flush=True,
    )
    if not IS_ADD_ALL_METHOD_LEGEND:
        save_method_legend_figure(
            result_methods,
            [METHOD_LABELS[method] for method in result_methods],
            out_dir,
        )
    for combo_index, (accel_min, omega_field, omega_value) in enumerate(grid_iterable, start=1):
        scenario = replace(
            base_scenario,
            accel_min=accel_min,
            **{omega_field: omega_value},
        )
        combo_cache_dir = out_dir / METHOD_CACHE_DIRNAME / (
            f"accel_min{format_float_for_path(accel_min)}"
            f"_{omega_field}{format_float_for_path(omega_value)}"
        )
        combo_label = (
            f"accel_min{format_float_for_path(accel_min)}"
            f"_{omega_field}{format_float_for_path(omega_value)}"
        )
        param_desc = (
            f"accel_min={accel_min:g} "
            f"accel_max={scenario.accel_max:g} "
            f"{omega_field}={omega_value:g}"
        )
        combo_results: list[dict[str, np.ndarray | str | int | float]] = []
        for method in result_methods:
            completed += 1
            result = None
            if args.reuse_cache and method not in forced_fresh_methods:
                result, cache_status = load_method_cache(method, scenario, combo_cache_dir)
                if result is not None:
                    print(
                        f"[{completed}/{total_rollouts}] "
                        f"{param_desc} "
                        f"method={method} (cached)",
                        flush=True,
                    )
            if result is None:
                print(
                    f"[{completed}/{total_rollouts}] "
                    f"{param_desc} "
                    f"method={method}",
                    flush=True,
                )
                result = simulate_method(method, scenario, show_progress=False)
                if args.reuse_cache:
                    save_method_cache(result, scenario, combo_cache_dir)
            combo_results.append(result)
            rows.append(
                rollout_statistics_row(
                    result, scenario, combo_index, method, omega_field,
                )
            )  # fmt: skip
        if args.plot_figure:
            save_trajectory_plot(combo_results, scenario, out_dir / combo_label)

    rollout_path = out_dir / "grid_sweep_rollouts.csv"
    with rollout_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = grid_sweep_method_summary_rows(rows)
    summary_path = out_dir / "grid_sweep_method_summary.csv"
    with summary_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    elapsed_s = time.perf_counter() - started_at
    print(f"\nGrid sweep complete in {elapsed_s:.1f}s.")
    print(f"Wrote rollout statistics to: {rollout_path.resolve()}")
    print(f"Wrote method summary to: {summary_path.resolve()}")


def print_summary(
    results: list[dict[str, np.ndarray | str | int | float]],
    scenario: Scenario,
) -> None:
    print("\nStatic obstacle comparison summary")
    print("-" * 158)
    header = (
        f"{'method':<24} {'stop':>15} {'t stop':>7} {'min h':>10} {'updates':>9} {'QP calls':>9} "
        f"{'tick ms':>9} {'update ms':>10} {'QP ms/call':>11} "
        f"{'total s':>9} {'RTF':>9} {'max upd ms':>11} {'infeas':>7}"
    )
    print(header)
    print("-" * 158)
    for result in results:
        metrics = timing_metrics(result, scenario)
        print(
            f"{result['label']:<24} "
            f"{result_stop_reason(result):>15} "
            f"{result_stop_time(result):>7.2f} "
            f"{float(np.min(result['h_values'])):>10.4f} "
            f"{int(metrics['num_controller_updates']):>9d} "
            f"{int(metrics['num_qp_calls']):>9d} "
            f"{float(metrics['mean_step_compute_ms']):>9.3f} "
            f"{float(metrics['mean_update_compute_ms']):>10.3f} "
            f"{float(metrics['mean_qp_call_compute_ms']):>11.3f} "
            f"{float(metrics['total_controller_compute_s']):>9.3f} "
            f"{float(metrics['controller_real_time_factor']):>9.3g} "
            f"{float(metrics['max_update_compute_ms']):>11.3f} "
            f"{int(result['num_infeasible']):>7d}"
        )
    print("\nQP status counts")
    for result in results:
        print(f"  {result['label']}: {status_counts_json(np.asarray(result['qp_call_statuses']))}")


def print_timing_interpretation(
    results: list[dict[str, np.ndarray | str | int | float]],
    scenario: Scenario,
) -> None:
    metrics_by_label = {
        str(result["label"]): timing_metrics(result, scenario)
        for result in results
    }
    min_h_by_label = {
        str(result["label"]): float(np.min(result["h_values"]))
        for result in results
    }

    total_rank = sorted(
        metrics_by_label,
        key=lambda label: float(metrics_by_label[label]["total_controller_compute_s"]),
    )
    update_rank = sorted(
        metrics_by_label,
        key=lambda label: float(metrics_by_label[label]["mean_update_compute_ms"]),
    )
    qp_rank = sorted(
        metrics_by_label,
        key=lambda label: float(metrics_by_label[label]["num_qp_calls"]),
    )
    unsafe_labels = [label for label, min_h in min_h_by_label.items() if min_h < 0.0]

    print("\nTiming interpretation")
    print("-" * 132)
    print(
        "The per fixed tick mean is the average controller cost seen by a fixed-rate "
        "monitor running at the simulation step size."
    )
    print(
        "The per controller update mean omits intervals without a new QP solve, so it compares "
        "the cost of recomputing a new control input."
    )
    print(
        "The QP call count is reported separately because Event-triggered TLC uses a "
        "preliminary QP plus an event QP at each update, and Event-triggered aTLC repeats "
        "that process for every candidate time scale."
    )
    print(
        "By total controller wall time over each method's actual simulated duration, "
        f"{total_rank[0]} is the lightest method and {total_rank[-1]} is the heaviest method."
    )
    print(
        f"By computation cost conditional on an actual controller update, "
        f"{update_rank[0]} has the smallest mean update time and {update_rank[-1]} has the largest mean update time."
    )
    print(
        f"By the number of QP calls, {qp_rank[0]} uses the fewest QP calls and "
        f"{qp_rank[-1]} uses the most QP calls."
    )
    if unsafe_labels:
        print(
            "The methods with negative minimum barrier values in this run are "
            f"{', '.join(unsafe_labels)}, so their timing should not be interpreted as a "
            "successful safe-control comparison without also reporting the safety violation."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help=(
            "directory for plots and data; default is eval_results_accel_min{accel_min} "
            "for single runs and eval_static_obstacle_grid_accel_min{end}_"
            "omega_{field_or_both}{magnitude}_{auto_or_manual} for --grid-sweep"
        ),
    )
    parser.add_argument(
        "--accel-min",
        type=float,
        help=(
            "minimum acceleration bound for the Scenario; equivalent to "
            "--set accel_min=VALUE and takes precedence if both are provided"
        ),
    )
    parser.add_argument(
        "--grid-sweep",
        action="store_true",
        default=False,
        help=(
            "run a deterministic accel_min/omega-bound grid sweep and export CSV "
            "statistics for adaptive methods instead of the default single-scenario plots"
        ),
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        default=False,
        help=(
            "export video_xy_trajectories.mp4 after a single-scenario run; "
            "supports fresh, cached, and mixed results"
        ),
    )
    parser.add_argument(
        "--is-auto-sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "when --grid-sweep is set, automatically choose whether to sweep "
            "omega_min or omega_max from obstacle side relative to the start-goal "
            "line; pass --no-is-auto-sweep to sweep both bounds"
        ),
    )
    parser.add_argument(
        "--plot-figure",
        action="store_true",
        dest="plot_figure",
        default=True,
        help=(
            "save fig_xy_trajectories.pdf for each grid-sweep combo "
            "(enabled by default for --grid-sweep)"
        ),
    )
    parser.add_argument(
        "--no-plot-figure",
        action="store_false",
        dest="plot_figure",
        help=(
            "skip fig_xy_trajectories.pdf exports for grid-sweep combos "
            "to keep grid-sweep runs lightweight"
        ),
    )
    parser.add_argument(
        "--accel-min-end",
        type=float,
        default=-1.0,
        help="inclusive final accel_min value for --grid-sweep (default: -1.0)",
    )
    parser.add_argument(
        "--omega-min-end",
        type=float,
        default=-2.0,
        help=(
            "inclusive final omega-bound magnitude for --grid-sweep (default: -2.0); "
            "the absolute value is used, with negative values applied to omega_min "
            "and positive values applied to omega_max"
        ),
    )
    parser.add_argument(
        "--accel-min-step",
        type=float,
        default=0.05,
        help="positive accel_min grid step for --grid-sweep (default: 0.05)",
    )
    parser.add_argument(
        "--omega-min-step",
        type=float,
        default=0.1,
        help="positive omega-bound magnitude grid step for --grid-sweep (default: 0.1)",
    )
    parser.add_argument(
        "--methods",
        help=(
            "comma-separated methods to simulate fresh, e.g. avcbf or racbf,avcbf; "
            "with cache reuse enabled, compatible cached baselines are still plotted"
        ),
    )
    parser.add_argument(
        "--reuse-cache",
        dest="reuse_cache",
        action="store_true",
        default=True,
        help=(
            "reuse compatible per-method caches from OUT_DIR/method_cache and "
            "simulate only requested, missing, or stale methods (default)"
        ),
    )
    parser.add_argument(
        "--no-reuse-cache",
        dest="reuse_cache",
        action="store_false",
        help=(
            "disable cache reuse and simulate only requested methods, or all "
            "methods if --methods is omitted"
        ),
    )
    parser.add_argument(
        "--set",
        dest="scenario_overrides",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "override a Scenario field; repeat for multiple overrides, e.g. "
            "--set avcbf_k1=0.5 --set avcbf_k2=0.5"
        ),
    )
    args = parser.parse_args()
    if args.methods is not None:
        try:
            args.methods = parse_methods_argument(args.methods)
        except ValueError as exc:
            parser.error(str(exc))
    if args.grid_sweep and args.save_video:
        parser.error("--save-video is only supported for single-scenario runs")
    if args.grid_sweep:
        try:
            inclusive_negative_grid(args.accel_min_end, args.accel_min_step, "accel_min")
            inclusive_positive_grid(abs(args.omega_min_end), args.omega_min_step, "omega_mag")
        except ValueError as exc:
            parser.error(str(exc))
    return args


def main() -> None:
    args = parse_args()
    scenario_overrides = list(args.scenario_overrides)
    if args.accel_min is not None:
        scenario_overrides.append(f"accel_min={args.accel_min}")
    try:
        scenario = apply_scenario_overrides(Scenario(), scenario_overrides)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    if args.out_dir is not None:
        out_dir = args.out_dir
    elif args.grid_sweep:
        omega_sweep_fields = (
            auto_sweep_omega_fields(scenario)
            if args.is_auto_sweep
            else ("omega_min", "omega_max")
        )
        out_dir = default_grid_sweep_output_dir(
            args.accel_min_end,
            args.omega_min_end,
            omega_sweep_fields,
            is_auto_sweep=args.is_auto_sweep,
        )
    else:
        out_dir = default_output_dir(scenario)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.grid_sweep:
        run_grid_sweep(args, scenario, out_dir)
        return

    cache_dir = out_dir / METHOD_CACHE_DIRNAME
    forced_fresh_methods = set(args.methods or ())
    if args.reuse_cache:
        result_methods = METHODS
    else:
        result_methods = tuple(args.methods) if args.methods is not None else METHODS

    results = []
    for method in result_methods:
        result = None
        if args.reuse_cache and method not in forced_fresh_methods:
            result, cache_status = load_method_cache(method, scenario, cache_dir)
            if result is not None:
                print(
                    f"Loaded cached {METHOD_LABELS[method]} from {method_cache_path(cache_dir, method)}",
                    flush=True,
                )
            else:
                print(
                    f"Cache miss for {METHOD_LABELS[method]} ({cache_status}); simulating.",
                    flush=True,
                )
        else:
            if args.reuse_cache and method in forced_fresh_methods:
                print(
                    f"Simulating {METHOD_LABELS[method]} because it was requested with --methods.",
                    flush=True,
                )

        if result is None:
            result = simulate_method(method, scenario, show_progress=True)
            save_method_cache(result, scenario, cache_dir)
        results.append(result)

    results_by_method = {str(result["method"]): result for result in results}

    save_summary(results, scenario, out_dir)
    save_all_methods_legend(results, scenario, out_dir)
    save_composite_results_plot(results, scenario, out_dir)
    for group_subdir, group_methods in PLOT_GROUPS:
        group_out_dir = out_dir if group_subdir is None else out_dir / group_subdir
        group_results = [
            results_by_method[method]
            for method in group_methods
            if method in results_by_method
        ]
        if not group_results:
            continue
        save_plots(group_results, scenario, group_out_dir)
    if args.save_video:
        try:
            video_path = save_trajectory_video(results, scenario, out_dir)
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(f"error: {exc}") from exc
        print(f"Wrote trajectory video to: {video_path.resolve()}")
    print_summary(results, scenario)
    print_timing_interpretation(results, scenario)
    print(f"\nWrote plots and logs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
