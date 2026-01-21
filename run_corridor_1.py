# Copyright (c) 2026, Professorship for Adaptive Behavior of Autonomous Vehicles, University of the Bundeswehr Munich.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from scenarios.corridor_navigation import (
    simulate,
    save_history,
    plot_multi_pos_trajectories,
    CBFMethod,
    ClassKType,
    DT,
    export_video,
)

from plot_corridor_1 import main as plot_main
# ---------------------------------------------------------------------
# Compare different Class-K functions (LINEAR, EXP, RATIONAL)
# ---------------------------------------------------------------------

def main() -> None:

    print("\n" + "=" * 80)
    print(f"Comparing Comparing aTTCBF with TTCBF for different Class-K functions")
    print("=" * 80)
    # -------------------------
    # Global experiment setup
    # -------------------------
    T_MAX = 8.0 + DT
    IS_SLACK_CBF = True

    # TTCBF parameters
    parameter_list = [0.2, 0.3, 0.4, None]  # None -> aTTCBF

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------
    # Loop over all Class-K
    # -------------------------
    for CLASS_K in [
        ClassKType.LINEAR,
        ClassKType.EXP,
        ClassKType.RATIONAL,
    ]:
        print(f"CLASS_K = {CLASS_K.name}")
    
        hist_list = []

        for idx, param in enumerate(parameter_list):
            if param is None:
                CBF_METHOD = CBFMethod.aTTCBF
                tag = "aTTCBF"
            else:
                CBF_METHOD = CBFMethod.TTCBF
                tag = f"TTCBF_a{param}"

            print(f"  → Class K = {CLASS_K.name}, param = {param}")

            hist = simulate(
                CBF_METHOD=CBF_METHOD,
                param=param,
                T_MAX=T_MAX,
                is_terminate_on_collision=True,
                CLASS_K=CLASS_K,
                IS_SLACK_CBF=IS_SLACK_CBF,
            )
                        
            # -------------------------
            # Save history
            # -------------------------
            save_path = data_dir / f"hist_{tag}_{CLASS_K.name}.pkl"
            save_history(hist, save_path)
            print("\n")

            hist_list.append(hist)

    plot_main()
        
if __name__ == "__main__":
    main()
