from pathlib import Path
import sys

import numpy as np
from ovito.io import import_file
from ovito.modifiers import PolyhedralTemplateMatchingModifier

sys.path.append(str(Path(__file__).resolve().parent))
from omega_identification_cr50 import (
    FCC_111,
    HCP_C,
    PARENT_111,
    cell_matrix,
    max_alignment,
    quat_xyzw_to_matrix,
    transformed_parent_111,
)


ROOT = Path(__file__).resolve().parents[1]
traj = ROOT / "zenodo_large_files" / "raw_trajectory" / "Cr50_S1_md.lammpstrj"
pipeline = import_file(str(traj), multiple_frames=True)
pipeline.modifiers.append(
    PolyhedralTemplateMatchingModifier(
        output_orientation=True,
        output_interatomic_distance=True,
        output_rmsd=True,
    )
)

ref = pipeline.compute(0)
h0, _ = cell_matrix(ref)
lines = []
for frame in [0, 100, 125, 150, 300]:
    data = pipeline.compute(frame)
    ht, _ = cell_matrix(data)
    parent = transformed_parent_111(h0, ht)
    st = np.asarray(data.particles["Structure Type"])
    q = np.asarray(data.particles["Orientation"])
    r = quat_xyzw_to_matrix(q)

    for label, stype, axes in [
        ("BCC_111", 3, PARENT_111),
        ("FCC_111", 1, FCC_111),
        ("HCP_0001", 2, HCP_C),
    ]:
        idx = np.where(st == stype)[0]
        if len(idx) == 0:
            lines.append(f"frame {frame} {label} no_atoms")
            continue
        idx = idx[: min(len(idx), 5000)]
        scores = np.zeros(len(idx))
        rr = r[idx]
        for axis in axes:
            lab = np.einsum("nij,j->ni", rr, axis)
            lab /= np.linalg.norm(lab, axis=1)[:, None]
            scores = np.maximum(scores, max_alignment(lab, parent))
        lines.append(
            f"frame {frame} {label} n_sample {len(idx)} "
            f"p10 {np.percentile(scores,10):.4f} p50 {np.percentile(scores,50):.4f} "
            f"p90 {np.percentile(scores,90):.4f} max {np.max(scores):.4f}"
        )

(ROOT / "data" / "omega_identification" / "ptm_orientation_score_validation.txt").write_text(
    "\n".join(lines),
    encoding="utf-8",
)
