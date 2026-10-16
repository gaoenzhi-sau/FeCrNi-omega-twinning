from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
from ovito.data import CutoffNeighborFinder
from ovito.io import import_file
from ovito.modifiers import PolyhedralTemplateMatchingModifier


ROOT = Path(__file__).resolve().parents[1]
TRAJ = ROOT / "zenodo_large_files" / "raw_trajectory" / "Cr50_S1_md.lammpstrj"
ARRAYS = ROOT / "data" / "omega_identification" / "cr50_omega_candidate_arrays.npz"
OUT = ROOT / "data" / "omega_identification" / "cr50_cluster_tracking_summary.csv"

REFERENCE_FRAME = 125
TRACK_FRAMES = [100, 125, 150, 300]
N_CLUSTERS = 3
CUTOFF = 3.8


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int32)
        self.size = np.ones(n, dtype=np.int32)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = int(self.parent[x])
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def connected_components(data, candidate_sorted: np.ndarray, order: np.ndarray) -> list[np.ndarray]:
    candidate_original = np.zeros_like(candidate_sorted, dtype=bool)
    candidate_original[order] = candidate_sorted
    original_to_sorted = np.empty_like(order)
    original_to_sorted[order] = np.arange(len(order), dtype=order.dtype)
    candidate_sorted_idx = np.where(candidate_sorted)[0]
    local_index = {int(sidx): i for i, sidx in enumerate(candidate_sorted_idx)}
    dsu = DSU(len(candidate_sorted_idx))
    finder = CutoffNeighborFinder(CUTOFF, data)
    for sidx in candidate_sorted_idx:
        oi = int(order[sidx])
        ai = local_index[int(sidx)]
        for neigh in finder.find(oi):
            nsidx = int(original_to_sorted[neigh.index])
            if candidate_sorted[nsidx]:
                dsu.union(ai, local_index[nsidx])

    buckets: dict[int, list[int]] = {}
    for local_i, sidx in enumerate(candidate_sorted_idx):
        root = dsu.find(local_i)
        buckets.setdefault(root, []).append(int(sidx))
    return [np.array(v, dtype=np.int64) for v in sorted(buckets.values(), key=len, reverse=True)]


def pct(mask: np.ndarray) -> float:
    return 100.0 * float(np.mean(mask)) if mask.size else float("nan")


def main() -> None:
    arrays = np.load(ARRAYS)
    pipeline = import_file(str(TRAJ), multiple_frames=True)
    pipeline.modifiers.append(PolyhedralTemplateMatchingModifier())
    data = pipeline.compute(REFERENCE_FRAME)
    ids = np.asarray(data.particles["Particle Identifier"], dtype=np.int64)
    order = np.argsort(ids)
    ref_candidate = arrays[f"frame_{REFERENCE_FRAME}_omega_candidate"].astype(bool)
    comps = connected_components(data, ref_candidate, order)[:N_CLUSTERS]

    rows: list[dict[str, str]] = []
    for rank, comp in enumerate(comps, start=1):
        comp_mask = np.zeros_like(ref_candidate, dtype=bool)
        comp_mask[comp] = True
        for frame in TRACK_FRAMES:
            candidate = arrays[f"frame_{frame}_omega_candidate"].astype(bool)
            structure = arrays[f"frame_{frame}_structure"].astype(np.uint8)
            non_bcc = structure != 3
            rows.append(
                {
                    "reference_frame": str(REFERENCE_FRAME),
                    "cluster_rank": str(rank),
                    "cluster_size_at_0p125": str(int(comp_mask.sum())),
                    "tracked_frame": str(frame),
                    "tracked_strain": f"{frame * 0.001:.3f}",
                    "same_atoms_candidate_pct": f"{pct(candidate[comp_mask]):.2f}",
                    "same_atoms_nonBCC_pct": f"{pct(non_bcc[comp_mask]):.2f}",
                    "same_atoms_BCC_pct": f"{pct(structure[comp_mask] == 3):.2f}",
                    "same_atoms_FCC_pct": f"{pct(structure[comp_mask] == 1):.2f}",
                    "same_atoms_HCP_pct": f"{pct(structure[comp_mask] == 2):.2f}",
                    "same_atoms_OTHER_pct": f"{pct(structure[comp_mask] == 0):.2f}",
                }
            )

    with OUT.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(OUT)


if __name__ == "__main__":
    main()
