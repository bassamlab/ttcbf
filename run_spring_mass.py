
# Copyright (c) 2026, Professorship for Adaptive Behavior of Autonomous Vehicles, University of the Bundeswehr Munich.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from scenarios.spring_mass import (
    CBFMethod,
    run_simulation,
    hist_to_traj,
    save_trajectory,
)
from plot_spring_mass import main as plot_main

def main() -> None:
    print("\n" + "=" * 80)
    print(f"Running spring-mass system with relative degree 6")
    print("=" * 80)
    
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
        
    hist_list = []
    cbf_methods = [CBFMethod.TTCBF, CBFMethod.NONE]

    for cbf_method in cbf_methods:
        print(f"...Running with CBF method: {cbf_method.name.upper()}...")
        hist = run_simulation(cbf_method)
        hist_list.append(hist)

        traj = hist_to_traj(hist, cbf_method)
        save_trajectory(traj, data_dir / f"hist_spring_mass_{cbf_method.name.lower()}.pkl")
        
    plot_main()
    
if __name__ == "__main__":
    main()
