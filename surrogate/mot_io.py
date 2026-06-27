"""MOT file I/O for muscle activations and coordinates."""

from __future__ import annotations

import numpy as np

__all__ = [
    "write_mot",
    "write_mot_file",
    "write_muscle_activations",
    "write_mot33",
    "write_mot35",
    "write_mot33_simulation",
]


def write_mot_file(t, data, labels, filepath):
    """Write OpenSim .mot file."""
    nframes = t.shape[0]
    ncols = len(labels)

    with open(filepath, "w") as f:
        f.write("name Motion\n")
        f.write("version=1\n")
        f.write("nRows=%d\n" % nframes)
        f.write("nColumns=%d\n" % (ncols + 1))
        f.write("inDegrees=no\n")
        f.write("endheader\n")
        f.write("time\t" + "\t".join(labels) + "\n")
        for i in range(nframes):
            row = [f"{t[i]:.6f}"] + [f"{data[i, j]:.6f}" for j in range(ncols)]
            f.write("\t".join(row) + "\n")


def write_muscle_activations(path: str, data: np.ndarray, framerate: int = 60) -> None:
    """Write muscle activation array to a simple MOT-style file."""
    n_rows = data.shape[0]
    n_cols = data.shape[1] if data.ndim > 1 else 1
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    labels = [f"activation_{i}" for i in range(n_cols)]
    times = np.arange(n_rows) / float(framerate)
    write_mot_file(times, data, labels, path)


def write_mot(path, data, framerate=60):
    header_string = (
        f"Coordinates\nversion=1\nnRows={data.shape[0]}\nnColumns=36\ninDegrees=yes\n\n"
        "Units are S.I. units (second, meters, Newtons, ...)\n"
        "If the header above contains a line with 'inDegrees', this indicates whether "
        "rotational values are in degrees (yes) or radians (no).\n\n"
        "endheader\n"
        "time\tpelvis_tilt\tpelvis_list\tpelvis_rotation\tpelvis_tx\tpelvis_ty\tpelvis_tz\t"
        "hip_flexion_r\thip_adduction_r\thip_rotation_r\tknee_angle_r\tknee_angle_r_beta\t"
        "ankle_angle_r\tsubtalar_angle_r\tmtp_angle_r\thip_flexion_l\thip_adduction_l\t"
        "hip_rotation_l\tknee_angle_l\tknee_angle_l_beta\tankle_angle_l\tsubtalar_angle_l\t"
        "mtp_angle_l\tlumbar_extension\tlumbar_bending\tlumbar_rotation\tarm_flex_r\t"
        "arm_add_r\tarm_rot_r\telbow_flex_r\tpro_sup_r\tarm_flex_l\tarm_add_l\tarm_rot_l\t"
        "elbow_flex_l\tpro_sup_l\n"
    )
    with open(path, "w") as f:
        f.write(header_string)
        for i, row in enumerate(data):
            values = [str(i / framerate)] + [str(x) for x in row]
            f.write("      " + "\t     ".join(values) + "\n")


def write_mot33(path, data, framerate=60):
    write_mot(path, data, framerate=framerate)


def write_mot35(path, data, framerate=60):
    write_mot(path, data, framerate=framerate)


def write_mot33_simulation(path, data, framerate=60):
    write_mot(path, data, framerate=framerate)
