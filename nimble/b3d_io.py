"""Shared B3D read helpers."""

from __future__ import annotations

import nimblephysics as nimble
import numpy as np

__all__ = ["read_trial_poses", "load_subject"]


def load_subject(b3d_path: str):
    return nimble.biomechanics.SubjectOnDisk(b3d_path)


def read_trial_poses(b3d_path: str, trial_idx: int = 0) -> np.ndarray:
    subject = load_subject(b3d_path)
    trial_length = subject.getTrialLength(trial_idx)
    frames = subject.readFrames(
        trial=trial_idx,
        startFrame=0,
        numFramesToRead=trial_length,
        includeSensorData=False,
        includeProcessingPasses=True,
    )
    kin_passes = [frame.processingPasses[0] for frame in frames if frame.processingPasses]
    return np.array([kp.pos for kp in kin_passes if hasattr(kp, "pos")])
