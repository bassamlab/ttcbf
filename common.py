# Copyright (c) 2026, Professorship for Adaptive Behavior of Autonomous Vehicles, University of the Bundeswehr Munich.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class CorridorTrajectoryHistory:
    t: List[float] = field(default_factory=list)
    xy: List[np.ndarray] = field(default_factory=list)
    state: List[np.ndarray] = field(default_factory=list)
    v: List[float] = field(default_factory=list)
    u1: List[float] = field(default_factory=list)
    u2: List[float] = field(default_factory=list)
    evt: List[str] = field(default_factory=list)  # "", "COLL", "QP-INF"
    lambda_ttcbf: List[float] = field(default_factory=list)
    nu_pacbf: List[np.ndarray] = field(default_factory=list)
    p1_pacbf: List[np.ndarray] = field(default_factory=list)
    p2_pacbf: List[np.ndarray] = field(default_factory=list)
    r_racbf: List[np.ndarray] = field(default_factory=list)
    nu_racbf: List[np.ndarray] = field(default_factory=list)
    qp_solving_t: List[float] = field(
        default_factory=list
    )  # QP solving time in seconds
    qp_solving_iter: List[int] = field(default_factory=list)  # QP solving iterations
    
    
    
@dataclass
class SpringMassTrajectory:
    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    meta: dict = field(default_factory=dict)

def cprint(text: str, color_code: int = 34) -> None:
    # Print text in color in the terminal
    print(f"\033[{color_code}m{text}\033[0m")