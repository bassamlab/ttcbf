# Copyright (c) 2026, Professorship for Adaptive Behavior of Autonomous Vehicles, University of the Bundeswehr Munich.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from scenarios.corridor_navigation import (
    simulate,
    save_history,
    plot_single_pos_trajectories,
    CBFMethod,
    ClassKType,
    DT,
)

from plot_corridor_2 import main as plot_main

# ---------------------------------------------------------------------
# Compare PACBF, aTTCBF, and RACBF
# ---------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 80)
    print("Comparing aTTCBF with PACBF and RACBF")
    print("=" * 80)
    # -------------------------
    # Global experiment setup
    # -------------------------
    T_MAX = 25.0 + DT
    IS_SLACK_CBF = False
    CLASS_K = ClassKType.LINEAR

    CBF_METHOD_list = [
        CBFMethod.aTTCBF,
        CBFMethod.PACBF,
        CBFMethod.RACBF,
    ]


    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    hist_list = []

    # -------------------------
    # Run simulations
    # -------------------------
    for CBF_METHOD in CBF_METHOD_list:
        print(f"...Running {CBF_METHOD.name}...")

        hist = simulate(
            param=None,
            CBF_METHOD=CBF_METHOD,
            P2_CONST=5,
            T_MAX=T_MAX,
            is_terminate_on_collision=False,
            CLASS_K=CLASS_K,
            IS_SLACK_CBF=IS_SLACK_CBF,
        )

        # -------------------------
        # Save history
        # -------------------------
        save_path = data_dir / f"hist_{CBF_METHOD.name}.pkl"
        save_history(hist, save_path)

        hist_list.append(hist)
        
    plot_main()

if __name__ == "__main__":
    main()