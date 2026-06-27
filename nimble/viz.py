"""Nimble GUI motion playback."""

from __future__ import annotations

import argparse
import time

import nimblephysics as nimble

from common.paths import ATHLETES_183_RETARGETED_DIR
from nimble.b3d_io import read_trial_poses


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a .b3d motion with Nimble GUI")
    parser.add_argument(
        "--b3d-path",
        type=str,
        default=f"{ATHLETES_183_RETARGETED_DIR}/927.b3d",
        help="Path to .b3d file",
    )
    parser.add_argument("--trial-idx", type=int, default=16, help="Trial index to play")
    parser.add_argument("--port", type=int, default=8000, help="Nimble GUI port")
    args = parser.parse_args()

    poses = read_trial_poses(args.b3d_path, trial_idx=args.trial_idx)
    zeros = __import__("numpy").zeros((poses.shape[0], 37 - poses.shape[1]))
    expanded = __import__("numpy").hstack([poses, zeros])

    gui = nimble.NimbleGUI()
    gui.serve(args.port)
    rajagopal_opensim = nimble.RajagopalHumanBodyModel()
    skeleton = rajagopal_opensim.skeleton
    gui.nativeAPI().renderSkeleton(skeleton)

    timestep = 1.0 / 120.0
    try:
        while True:
            for t in range(len(expanded)):
                skeleton.setPositions(expanded[t])
                gui.nativeAPI().renderSkeleton(skeleton)
                time.sleep(timestep)
    finally:
        gui.blockWhileServing()


if __name__ == "__main__":
    main()
