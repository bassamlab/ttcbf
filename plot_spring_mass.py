# Copyright (c) 2026, Professorship for Adaptive Behavior of Autonomous Vehicles, University of the Bundeswehr Munich.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

# import everything you need from your main script module
# e.g., if your main file is named spring_mass.py:
from scenarios.spring_mass import load_trajectory, plot_results, export_video

def main(is_export_video: bool = True) -> None:
    traj_1 = load_trajectory(Path("data") / "hist_spring_mass_ttcbf.pkl")
    traj_2 = load_trajectory(Path("data") / "hist_spring_mass_none.pkl")
    
    traj_list = [traj_1, traj_2]
    
    Path("fig").mkdir(parents=True, exist_ok=True)
    plot_results(Path("fig"), traj_list)

    if is_export_video:
        Path("video").mkdir(parents=True, exist_ok=True)
        export_video(
            traj=traj_1,
            video_path=Path("video") / "video_spring_mass_ttcbf.mp4",
            fps=None,
            dpi=150,
            labels=[r"$x_1$", r"$x_2$", r"$x_3$"],
            legend_title=traj_1.meta.get("cbf_method", None),
        )
        
        export_video(
            traj=traj_2,
            video_path=Path("video") / "video_spring_mass_none.mp4",
            fps=None,
            dpi=150,
            labels=[r"$x_1$", r"$x_2$", r"$x_3$"],
            legend_title=traj_2.meta.get("cbf_method", None),
        )

if __name__ == "__main__":
    main()