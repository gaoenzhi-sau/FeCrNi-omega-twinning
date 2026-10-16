from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
from ovito.data import CutoffNeighborFinder
from ovito.io import import_file
from ovito.modifiers import PolyhedralTemplateMatchingModifier


ROOT = Path(__file__).resolve().parents[1]
TRAJ = ROOT / "zenodo_large_files" / "raw_trajectory" / "Cr50_S1_md.lammpstrj"
OMEGA = ROOT / "data" / "omega_identification" / "cr50_omega_candidate_arrays.npz"
OUT = ROOT / "data" / "omega_identification" / "cr50_local_chemistry_summary.csv"

FRAMES = [100, 125, 150, 300]
TYPE_NAMES = {1: "Fe", 2: "Cr", 3: "Ni"}


def fractions(types: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    n = int(np.sum(mask))
    values: dict[str, float] = {"n": float(n)}
    if n == 0:
        for name in TYPE_NAMES.values():
            values[f"{name}_at_pct"] = float("nan")
        return values
    for typ, name in TYPE_NAMES.items():
        values[f"{name}_at_pct"] = 100.0 * float(np.sum(types[mask] == typ)) / n
    return values


def neighbor_shell_mask(data, candidate_sorted: np.ndarray, order: np.ndarray, cutoff: float = 3.8) -> np.ndarray:
    candidate_original = np.zeros_like(candidate_sorted, dtype=bool)
    candidate_original[order] = candidate_sorted
    shell_original = candidate_original.copy()
    finder = CutoffNeighborFinder(cutoff, data)
    for original_idx in np.where(candidate_original)[0]:
        for neigh in finder.find(int(original_idx)):
            shell_original[neigh.index] = True
    return shell_original[order]


def main() -> None:
    arrays = np.load(OMEGA)
    pipeline = import_file(str(TRAJ), multiple_frames=True)
    pipeline.modifiers.append(PolyhedralTemplateMatchingModifier())

    rows: list[dict[str, str]] = []
    for frame in FRAMES:
        data = pipeline.compute(frame)
        ids = np.asarray(data.particles["Particle Identifier"], dtype=np.int64)
        order = np.argsort(ids)
        types = np.asarray(data.particles["Particle Type"], dtype=np.int32)[order]
        structure = np.asarray(data.particles["Structure Type"], dtype=np.int32)[order]

        candidate = arrays[f"frame_{frame}_omega_candidate"].astype(bool)
        non_bcc = structure != 3
        groups = {
            "nominal_all": np.ones_like(candidate, dtype=bool),
            "BCC_matrix": structure == 3,
            "nonBCC_noncandidate": non_bcc & (~candidate),
            "omega_candidate": candidate,
            "omega_candidate_plus_3p8A_shell": neighbor_shell_mask(data, candidate, order, cutoff=3.8),
        }
        strain = float(data.attributes.get("Timestep", frame * 1000)) * 1e-6
        for name, mask in groups.items():
            vals = fractions(types, mask)
            rows.append(
                {
                    "frame": str(frame),
                    "strain": f"{strain:.6f}",
                    "group": name,
                    "n": str(int(vals["n"])),
                    "Fe_at_pct": f"{vals['Fe_at_pct']:.3f}",
                    "Cr_at_pct": f"{vals['Cr_at_pct']:.3f}",
                    "Ni_at_pct": f"{vals['Ni_at_pct']:.3f}",
                }
            )

    with OUT.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(OUT)


if __name__ == "__main__":
    main()
