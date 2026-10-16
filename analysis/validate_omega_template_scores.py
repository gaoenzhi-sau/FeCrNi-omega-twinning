from pathlib import Path
import sys

import numpy as np
from ovito.io import import_file
from ovito.modifiers import PolyhedralTemplateMatchingModifier

sys.path.append(str(Path(__file__).resolve().parent))
from omega_identification_cr50 import (  # noqa: E402
    TRAJ,
    cell_matrix,
    omega_template_distance_sets,
    omega_template_scores,
    omega_template_vector_sets,
    omega_vector_template_scores,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "omega_identification" / "template_score_negative_validation.txt"
arrays = np.load(ROOT / "data" / "omega_identification" / "cr50_omega_candidate_arrays.npz")
pipeline = import_file(str(TRAJ), multiple_frames=True)
pipeline.modifiers.append(
    PolyhedralTemplateMatchingModifier(
        output_orientation=True,
        output_interatomic_distance=True,
        output_rmsd=True,
    )
)

lines = []
for frame in [0, 100, 125, 150, 300]:
    data = pipeline.compute(frame)
    ids = np.asarray(data.particles["Particle Identifier"], dtype=np.int64)
    order = np.argsort(ids)
    st = np.asarray(data.particles["Structure Type"], dtype=np.int32)[order]
    candidate = arrays[f"frame_{frame}_omega_candidate"].astype(bool)
    ht, _ = cell_matrix(data)
    distance_templates = omega_template_distance_sets(ht)
    vector_templates = omega_template_vector_sets(ht)

    groups = {
        "candidate": candidate,
        "bcc": st == 3,
        "nonbcc_noncandidate": (st != 3) & (~candidate),
    }
    for name, mask in groups.items():
        idx = np.where(mask)[0]
        if len(idx) == 0:
            lines.append(f"frame {frame} {name} n 0")
            continue
        # Deterministic down-sample for speed.
        if len(idx) > 5000:
            idx = idx[np.linspace(0, len(idx) - 1, 5000, dtype=int)]
        sample_mask = np.zeros_like(mask, dtype=bool)
        sample_mask[idx] = True
        distance_score = omega_template_scores(data, sample_mask, order, distance_templates)
        vector_score = omega_vector_template_scores(data, sample_mask, order, vector_templates)
        dvalues = distance_score[sample_mask]
        dvalues = dvalues[np.isfinite(dvalues)]
        vvalues = vector_score[sample_mask]
        vvalues = vvalues[np.isfinite(vvalues)]
        lines.append(
            f"frame {frame} {name} n {len(vvalues)} "
            f"dist_p50 {np.percentile(dvalues,50):.5f} "
            f"vec_p10 {np.percentile(vvalues,10):.5f} vec_p50 {np.percentile(vvalues,50):.5f} "
            f"vec_p90 {np.percentile(vvalues,90):.5f}"
        )

OUT.write_text("\n".join(lines), encoding="utf-8")
