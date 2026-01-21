# Copyright (c) 2026, Professorship for Adaptive Behavior of Autonomous Vehicles, University of the Bundeswehr Munich.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

# Import everything from your simulation module
# Assume your original file is named: circle_corridor.py
from scenarios.corridor_navigation import (
    export_video,
    load_history,
    plot_single_pos_trajectories,
    CBFMethod,
    ClassKType,
)

# ---------------------------------------------------------------------
# Reload results and plot
# ---------------------------------------------------------------------

def main(is_export_video: bool = True, is_all_traj_one_video: bool = True) -> None:

    data_dir = Path("data")

    hist_aTTCBF = load_history(data_dir / "hist_aTTCBF.pkl")
    hist_PACBF  = load_history(data_dir / "hist_PACBF.pkl")
    hist_RACBF  = load_history(data_dir / "hist_RACBF.pkl")

    hist_list = [hist_aTTCBF, hist_PACBF, hist_RACBF]
    cbf_list  = [CBFMethod.aTTCBF, CBFMethod.PACBF, CBFMethod.RACBF]

    fig_dir = Path("fig")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_name_prefix = fig_dir / "fig_corridor_acbfs"

    plot_single_pos_trajectories(
        hist_list=hist_list,
        cbf_list=cbf_list,
        fig_name_prefix=fig_name_prefix,
    )

    is_save_video = True
    if is_save_video:
        video_dir = Path("video")
        video_dir.mkdir(parents=True, exist_ok=True)
        cm = [
            "#008B8B",
            "tab:purple",
            "tab:orange",
        ]
        ls = ["-", "--", "-."]
        
        if is_export_video:
            if is_all_traj_one_video:
                labels = [
                    "aTTCBF",
                    "PACBF",
                    "RACBF",
                ]
                video_path = video_dir / "video_all_acbfs.mp4"
                legend_title = rf"Class $\mathcal{{K}}$: {ClassKType.LINEAR.name.capitalize()}"
                export_video(hist=hist_list, video_path=video_path, traj_color=cm, ls=ls, is_full_map=True, labels=labels, legend_title=legend_title)
            else:
                video_paths = [
                    video_dir / f"video_aTTCBF.mp4",
                    video_dir / f"video_PACBF.mp4",
                    video_dir / f"video_RACBF.mp4",
                ]

                for idx, hist in enumerate(hist_list):
                    export_video(hist=hist, video_path=video_paths[idx], traj_color=cm[idx], ls=ls[idx], is_full_map=True)

if __name__ == "__main__":
    main()