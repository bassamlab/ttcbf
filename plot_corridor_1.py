# Copyright (c) 2026, Professorship for Adaptive Behavior of Autonomous Vehicles, University of the Bundeswehr Munich.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from scenarios.corridor_navigation import (
    export_video,
    load_history,
    plot_multi_pos_trajectories,
    CBFMethod,
    ClassKType,
)


# ---------------------------------------------------------------------
# Load saved TTCBF / aTTCBF results
# ---------------------------------------------------------------------

def ask_class_k() -> ClassKType:
    print("\nSelect Class-K type:")
    for idx, k in enumerate(ClassKType):
        print(f"  [{idx}] {k.name}")

    while True:
        user_in = input("Enter index: ").strip()
        if not user_in.isdigit():
            print("Invalid input. Enter an integer index.")
            continue

        idx = int(user_in)
        if 0 <= idx < len(ClassKType):
            return list(ClassKType)[idx]

        print("Index out of range.")


def load_histories(class_k: ClassKType):
    """
    Load histories for the selected Class-K.
    Assumes filenames:
        data/hist_TTCBF_a0.2_<CLASS_K>.pkl
        data/hist_TTCBF_a0.3_<CLASS_K>.pkl
        data/hist_TTCBF_a0.4_<CLASS_K>.pkl
        data/hist_aTTCBF_<CLASS_K>.pkl
    """
    data_dir = Path("data")

    parameter_list = [0.2, 0.3, 0.4, None]

    hist_paths = [
        data_dir / f"hist_TTCBF_a0.2_{class_k.name}.pkl",
        data_dir / f"hist_TTCBF_a0.3_{class_k.name}.pkl",
        data_dir / f"hist_TTCBF_a0.4_{class_k.name}.pkl",
        data_dir / f"hist_aTTCBF_{class_k.name}.pkl",
    ]

    hist_list = []
    for p in hist_paths:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing file: {p}. "
                "Make sure you have generated and saved relevant data."
            )
        hist_list.append(load_history(p))

    return hist_list, parameter_list


def main(is_export_video: bool = True, is_all_traj_one_video: bool = True) -> None:
    CLASS_K_list = [
        ClassKType.LINEAR,
        ClassKType.EXP,
        ClassKType.RATIONAL,
    ]
    
    for CLASS_K in CLASS_K_list:
        print("\n" + "=" * 80)
        print(f"Processing Class-K: {CLASS_K.name}")
        print("=" * 80)
        # -------------------------
        # Load data
        # -------------------------
        hist_list, parameter_list = load_histories(CLASS_K)
        
        
        fig_dir = Path("fig")
        fig_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------
        # Plot
        # -------------------------
        fig_name_prefix = fig_dir / f"fig_corridor_attcbf_{CLASS_K.name.lower()}"

        plot_multi_pos_trajectories(
            CLASS_K=CLASS_K,
            hist_list=hist_list,
            fig_name_prefix=fig_name_prefix,
            parameter_list=parameter_list,
        )
        
                            
        if is_export_video:
            video_dir = Path("video")
            video_dir.mkdir(parents=True, exist_ok=True)

            cm = [
                "darkgoldenrod",
                "darkviolet",
                "darkred",
                "#008B8B",
            ]
            
            if is_all_traj_one_video:
                # All trajectories in one video
                video_path = video_dir / f"video_all_ttcbf_{CLASS_K.name}.mp4"
                labels = [
                    r"$a=0.2$",
                    r"$a=0.3$",
                    r"$a=0.4$",
                    "aTTCBF",
                ]
                ls = ["-", "-", "-", "-"]
                legend_title = rf"Class $\mathcal{{K}}$: {CLASS_K.name.capitalize()}"
                export_video(hist=hist_list, video_path=video_path, traj_color=cm, ls=ls, is_full_map=False, labels=labels, legend_title=legend_title)
            else:
                # One trajectory per video
                video_paths = [
                    video_dir / f"video_TTCBF_a0.2_{CLASS_K.name}.mp4",
                    video_dir / f"video_TTCBF_a0.3_{CLASS_K.name}.mp4",
                    video_dir / f"video_TTCBF_a0.4_{CLASS_K.name}.mp4",
                    video_dir / f"video_aTTCBF_{CLASS_K.name}.mp4",
                ]
                for idx, hist in enumerate(hist_list):
                    export_video(hist=hist, video_path=video_paths[idx], traj_color=cm[idx], is_full_map=False)


if __name__ == "__main__":
    main()