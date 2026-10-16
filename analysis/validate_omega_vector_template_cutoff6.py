from pathlib import Path
import sys

import numpy as np
from ovito.io import import_file
from ovito.modifiers import PolyhedralTemplateMatchingModifier

sys.path.append(str(Path(__file__).resolve().parent))
from omega_identification_cr50 import (  # noqa: E402
    TRAJ,
    cell_matrix,
    omega_template_vector_sets,
    omega_vector_template_scores,
)


ROOT = Path(__file__).resolve().parents[1]
arrays = np.load(ROOT / "data" / "omega_identification" / "cr50_omega_candidate_arrays.npz")
pipeline = import_file(str(TRAJ), multiple_frames=True)
pipeline.modifiers.append(PolyhedralTemplateMatchingModifier(output_orientation=True, output_interatomic_distance=True, output_rmsd=True))
lines = []
for cutoff in [3.8, 5.0, 6.0]:
    for frame in [0, 125, 150, 300]:
        data = pipeline.compute(frame)
        ids = np.asarray(data.particles["Particle Identifier"], dtype=np.int64)
        order = np.argsort(ids)
        st = np.asarray(data.particles["Structure Type"], dtype=np.int32)[order]
        cand = arrays[f"frame_{frame}_omega_candidate"].astype(bool)
        ht, _ = cell_matrix(data)
        templates = omega_template_vector_sets(ht, cutoff=cutoff, max_shell=36)
        for name, mask in {
            "candidate": cand,
            "bcc": st == 3,
            "nonbcc_noncandidate": (st != 3) & (~cand),
        }.items():
            idx = np.where(mask)[0]
            if len(idx) == 0:
                lines.append(f"cutoff {cutoff} frame {frame} {name} n 0")
                continue
            if len(idx) > 1200:
                idx = idx[np.linspace(0, len(idx) - 1, 1200, dtype=int)]
            sample = np.zeros_like(mask, dtype=bool)
            sample[idx] = True
            score = omega_vector_template_scores(data, sample, order, templates, cutoff=cutoff, min_shell=12, max_shell=36)
            values = score[sample]
            values = values[np.isfinite(values)]
            lines.append(
                f"cutoff {cutoff} frame {frame} {name} n {len(values)} "
                f"p10 {np.percentile(values,10):.5f} p50 {np.percentile(values,50):.5f} "
                f"p90 {np.percentile(values,90):.5f}"
            )

(ROOT / "data" / "omega_identification" / "vector_template_cutoff_sensitivity.txt").write_text(
    "\n".join(lines),
    encoding="utf-8",
)
