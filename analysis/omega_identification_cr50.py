from __future__ import annotations

from pathlib import Path
import csv
import math

import numpy as np
from ovito.data import CutoffNeighborFinder
from ovito.io import import_file
from ovito.modifiers import AtomicStrainModifier, PolyhedralTemplateMatchingModifier


ROOT = Path(__file__).resolve().parents[1]
TRAJ = ROOT / "zenodo_large_files" / "raw_trajectory" / "Cr50_S1_md.lammpstrj"
OUT_DIR = ROOT / "data" / "omega_identification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRAMES = [0, 50, 75, 100, 125, 150, 200, 250, 300]
STRUCTURE_NAMES = {
    0: "OTHER",
    1: "FCC",
    2: "HCP",
    3: "BCC",
}

# The tensile model has 69 x 34 x 69 BCC unit cells.
NCELLS = np.array([69.0, 34.0, 69.0])
PARENT_111 = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ]
)
PARENT_111 /= np.linalg.norm(PARENT_111, axis=1)[:, None]
FCC_111 = PARENT_111.copy()
HCP_C = np.array([[0.0, 0.0, 1.0]])


def cell_matrix(data):
    cell = np.asarray(data.cell[:], dtype=float)
    return cell[:, :3], cell[:, 3]


def fractional_positions(positions, h, origin):
    return np.linalg.solve(h, (positions - origin).T).T


def quat_xyzw_to_matrix(q):
    q = np.asarray(q, dtype=float)
    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2]
    w = q[:, 3]
    n = np.sqrt(x * x + y * y + z * z + w * w)
    n[n == 0.0] = 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    r = np.empty((len(q), 3, 3), dtype=float)
    r[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    r[:, 0, 1] = 2.0 * (x * y - z * w)
    r[:, 0, 2] = 2.0 * (x * z + y * w)
    r[:, 1, 0] = 2.0 * (x * y + z * w)
    r[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    r[:, 1, 2] = 2.0 * (y * z - x * w)
    r[:, 2, 0] = 2.0 * (x * z - y * w)
    r[:, 2, 1] = 2.0 * (y * z + x * w)
    r[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return r


def transformed_parent_111(h0, ht):
    f = ht @ np.linalg.inv(h0)
    dirs = PARENT_111 @ f.T
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    return dirs


def b12_length(ht):
    # Length of an affine-transformed a/12<111> shuffle.
    a = np.linalg.norm(ht[:, :3], axis=0) / NCELLS
    return float(np.sqrt(np.sum(a * a)) / 12.0)


def max_alignment(vectors, parent_dirs):
    dots = np.abs(vectors @ parent_dirs.T)
    return np.max(dots, axis=1)


def orientation_scores(q, structure, parent_dirs):
    score = np.zeros(len(structure), dtype=float)
    valid = np.isin(structure, [1, 2])
    if not np.any(valid):
        return score
    idx = np.where(valid)[0]
    r = quat_xyzw_to_matrix(q[idx])
    local_scores = np.zeros(len(idx), dtype=float)
    for local_axis_set, stype in [(FCC_111, 1), (HCP_C, 2)]:
        sub = np.where(structure[idx] == stype)[0]
        if not len(sub):
            continue
        rr = r[sub]
        best = np.zeros(len(sub), dtype=float)
        for axis in local_axis_set:
            lab = np.einsum("nij,j->ni", rr, axis)
            lab /= np.linalg.norm(lab, axis=1)[:, None]
            best = np.maximum(best, max_alignment(lab, parent_dirs))
        local_scores[sub] = best
    score[idx] = local_scores
    return score


def connected_neighbor_counts(data, candidate_mask, cutoff=3.8, max_atoms=None):
    finder = CutoffNeighborFinder(cutoff, data)
    counts = np.zeros(data.particles.count, dtype=np.int16)
    idxs = np.where(candidate_mask)[0]
    if max_atoms is not None:
        idxs = idxs[:max_atoms]
    for i in idxs:
        c = 0
        for neigh in finder.find(i):
            if candidate_mask[neigh.index]:
                c += 1
        counts[i] = c
    return counts


def clustered_mask_from_sorted(data, gate_sorted, order, min_neighbors=4, cutoff=3.8):
    mask_original = np.zeros_like(gate_sorted, dtype=bool)
    mask_original[order] = gate_sorted
    neighbor_counts_original = connected_neighbor_counts(data, mask_original, cutoff=cutoff)
    neighbor_counts_sorted = neighbor_counts_original[order]
    return gate_sorted & (neighbor_counts_sorted >= min_neighbors), neighbor_counts_sorted


def percentile(values, qs):
    if len(values) == 0:
        return {f"p{q}": np.nan for q in qs}
    return {f"p{q}": float(np.percentile(values, q)) for q in qs}


def lattice_vector(ht, cubic_components):
    return ht @ (np.asarray(cubic_components, dtype=float) / NCELLS)


def perpendicular_hex_pair(n):
    candidates = []
    for i in [-1.0, 0.0, 1.0]:
        for j in [-1.0, 0.0, 1.0]:
            for k in [-1.0, 0.0, 1.0]:
                v = np.array([i, j, k], dtype=float)
                if np.linalg.norm(v) == 0:
                    continue
                if abs(np.dot(v, n)) < 1.0e-9 and abs(np.linalg.norm(v) - math.sqrt(2.0)) < 1.0e-9:
                    candidates.append(v)
    for a in candidates:
        for b in candidates:
            if abs(np.dot(a, b) + 1.0) < 1.0e-9:
                return a, b
    raise RuntimeError(f"No hexagonal in-plane basis found for {n}")


def omega_template_vector_sets(ht, cutoff=3.8, max_shell=14):
    templates = []
    for n in PARENT_111 * math.sqrt(3.0):
        e1, e2 = perpendicular_hex_pair(n)
        a1 = lattice_vector(ht, e1)
        a2 = lattice_vector(ht, e2)
        cvec = 0.5 * lattice_vector(ht, n)
        basis = [
            np.zeros(3),
            (2.0 * a1 + a2) / 3.0 + 0.5 * cvec,
            (a1 + 2.0 * a2) / 3.0 + 0.5 * cvec,
        ]
        for b0 in basis:
            vectors = []
            for i in range(-2, 3):
                for j in range(-2, 3):
                    for k in range(-2, 3):
                        translation = i * a1 + j * a2 + k * cvec
                        for b in basis:
                            r = translation + b - b0
                            d = float(np.linalg.norm(r))
                            if 1.0e-6 < d <= cutoff:
                                vectors.append(r)
            if vectors:
                vectors = sorted(vectors, key=lambda x: float(np.linalg.norm(x)))[:max_shell]
                templates.append(np.array(vectors, dtype=float))
    return templates


def omega_template_distance_sets(ht, cutoff=3.8, max_shell=14):
    return [
        np.array(sorted(np.linalg.norm(vectors, axis=1))[:max_shell], dtype=float)
        for vectors in omega_template_vector_sets(ht, cutoff=cutoff, max_shell=max_shell)
    ]


def omega_template_scores(data, candidate_mask_sorted, order, templates, cutoff=3.8, min_shell=8):
    scores_sorted = np.full(len(candidate_mask_sorted), np.nan, dtype=float)
    if not np.any(candidate_mask_sorted):
        return scores_sorted
    mask_original = np.zeros_like(candidate_mask_sorted, dtype=bool)
    mask_original[order] = candidate_mask_sorted
    finder = CutoffNeighborFinder(cutoff, data)
    for sorted_idx, original_idx in enumerate(order):
        if not candidate_mask_sorted[sorted_idx]:
            continue
        distances = sorted(float(neigh.distance) for neigh in finder.find(int(original_idx)))
        if len(distances) < min_shell:
            continue
        measured = np.array(distances, dtype=float)
        best = np.inf
        for template in templates:
            k = min(len(measured), len(template), 14)
            if k < min_shell:
                continue
            denom = max(float(np.mean(template[:k])), 1.0e-12)
            score = float(np.sqrt(np.mean((measured[:k] - template[:k]) ** 2)) / denom)
            if score < best:
                best = score
        if np.isfinite(best):
            scores_sorted[sorted_idx] = best
    return scores_sorted


def omega_vector_template_scores(data, candidate_mask_sorted, order, vector_templates, cutoff=3.8, min_shell=8, max_shell=14):
    scores_sorted = np.full(len(candidate_mask_sorted), np.nan, dtype=float)
    if not np.any(candidate_mask_sorted):
        return scores_sorted
    finder = CutoffNeighborFinder(cutoff, data)
    for sorted_idx, original_idx in enumerate(order):
        if not candidate_mask_sorted[sorted_idx]:
            continue
        vectors = []
        for neigh in finder.find(int(original_idx)):
            vectors.append(np.asarray(neigh.delta, dtype=float))
        if len(vectors) < min_shell:
            continue
        measured = np.array(sorted(vectors, key=lambda x: float(np.linalg.norm(x)))[:max_shell], dtype=float)
        best = np.inf
        for template in vector_templates:
            k = min(len(measured), len(template), max_shell)
            if k < min_shell:
                continue
            m = measured[:k]
            t = template[:k]
            d_mt = np.sqrt(np.sum((m[:, None, :] - t[None, :, :]) ** 2, axis=2))
            chamfer = 0.5 * (np.mean(np.min(d_mt, axis=1) ** 2) + np.mean(np.min(d_mt, axis=0) ** 2))
            denom = max(float(np.mean(np.linalg.norm(t, axis=1))), 1.0e-12)
            score = float(np.sqrt(chamfer) / denom)
            if score < best:
                best = score
        if np.isfinite(best):
            scores_sorted[sorted_idx] = best
    return scores_sorted


def main():
    pipeline = import_file(str(TRAJ), multiple_frames=True)
    ptm = PolyhedralTemplateMatchingModifier(
        output_orientation=True,
        output_interatomic_distance=True,
        output_rmsd=True,
    )
    pipeline.modifiers.append(ptm)
    strain = AtomicStrainModifier(cutoff=3.8)
    strain.reference_frame = 0
    strain.eliminate_cell_deformation = True
    strain.output_nonaffine_squared_displacements = True
    strain.output_strain_tensors = True
    pipeline.modifiers.append(strain)

    ref = pipeline.compute(0)
    ref_ids = np.asarray(ref.particles["Particle Identifier"], dtype=np.int64)
    ref_order = np.argsort(ref_ids)
    ref_pos = np.asarray(ref.particles["Position"], dtype=float)[ref_order]
    h0, o0 = cell_matrix(ref)
    ref_frac = fractional_positions(ref_pos, h0, o0)
    ref_id_sorted = ref_ids[ref_order]

    rows = []
    candidate_npz = {}

    for frame in FRAMES:
        data = pipeline.compute(frame)
        timestep = int(data.attributes.get("Timestep", frame * 1000))
        strain_value = timestep * 1e-6
        ids = np.asarray(data.particles["Particle Identifier"], dtype=np.int64)
        order = np.argsort(ids)
        if not np.array_equal(ids[order], ref_id_sorted):
            raise RuntimeError(f"Particle identifiers do not match reference at frame {frame}")

        pos = np.asarray(data.particles["Position"], dtype=float)[order]
        structure = np.asarray(data.particles["Structure Type"], dtype=np.int32)[order]
        rmsd = np.asarray(data.particles["RMSD"], dtype=float)[order]
        q = np.asarray(data.particles["Orientation"], dtype=float)[order]
        d2 = np.asarray(data.particles["Nonaffine Squared Displacement"], dtype=float)[order]
        shear = np.asarray(data.particles["Shear Strain"], dtype=float)[order]

        ht, ot = cell_matrix(data)
        cur_frac = fractional_positions(pos, ht, ot)
        delta_frac = cur_frac - ref_frac
        delta_frac -= np.round(delta_frac)
        disp = (ht @ delta_frac.T).T
        disp_norm = np.linalg.norm(disp, axis=1)

        parent_dirs = transformed_parent_111(h0, ht)
        projections = np.abs(disp @ parent_dirs.T)
        max_proj = np.max(projections, axis=1)
        variant = np.argmax(projections, axis=1)
        projection_collinearity = max_proj / np.maximum(disp_norm, 1.0e-12)
        b12 = b12_length(ht)
        partial_units = max_proj / b12
        orient_score = orientation_scores(q, structure, parent_dirs)

        non_bcc = structure != 3
        # Omega-compatible shuffle gate: non-BCC PTM environment, strong <111>
        # directionality, non-trivial D2min, and a projected displacement
        # compatible with the first few a/12<111> shuffle increments. Three
        # lower thresholds are retained as a sensitivity check. The lo=0.45
        # connected set is used as the main omega-like candidate population.
        gates = {}
        clustered = {}
        neighbor_counts_by_threshold = {}
        for lo in [0.35, 0.45, 0.55]:
            gate = (
                non_bcc
                & (projection_collinearity >= 0.70)
                & (partial_units >= lo)
                & (partial_units <= 4.25)
                & (d2 >= 0.02)
            )
            gates[lo] = gate
            clustered[lo], neighbor_counts_by_threshold[lo] = clustered_mask_from_sorted(data, gate, order)

        orientation_gate = (structure == 0) | (orient_score >= 0.86)
        oriented_gate_045 = gates[0.45] & orientation_gate
        oriented_clustered_045, oriented_neighbor_counts = clustered_mask_from_sorted(data, oriented_gate_045, order)

        omega_candidate = clustered[0.45]
        omega_orientation_subset = oriented_clustered_045
        distance_templates = omega_template_distance_sets(ht)
        vector_templates = omega_template_vector_sets(ht)
        template_score = omega_template_scores(data, omega_candidate, order, distance_templates)
        vector_template_score = omega_vector_template_scores(data, omega_candidate, order, vector_templates)
        template_subset = omega_candidate & np.isfinite(vector_template_score) & (vector_template_score <= 0.28)

        total = len(ids)
        candidate_npz[f"frame_{frame}_ids"] = ref_id_sorted
        candidate_npz[f"frame_{frame}_omega_candidate"] = omega_candidate.astype(np.uint8)
        candidate_npz[f"frame_{frame}_omega_orientation_subset"] = omega_orientation_subset.astype(np.uint8)
        candidate_npz[f"frame_{frame}_omega_template_subset"] = template_subset.astype(np.uint8)
        candidate_npz[f"frame_{frame}_omega_distance_template_score"] = template_score.astype(np.float32)
        candidate_npz[f"frame_{frame}_omega_vector_template_score"] = vector_template_score.astype(np.float32)
        candidate_npz[f"frame_{frame}_variant"] = variant.astype(np.uint8)
        candidate_npz[f"frame_{frame}_partial_units"] = partial_units.astype(np.float32)
        candidate_npz[f"frame_{frame}_projection_collinearity"] = projection_collinearity.astype(np.float32)
        candidate_npz[f"frame_{frame}_orientation_score"] = orient_score.astype(np.float32)
        candidate_npz[f"frame_{frame}_d2min"] = d2.astype(np.float32)
        candidate_npz[f"frame_{frame}_shear"] = shear.astype(np.float32)
        candidate_npz[f"frame_{frame}_structure"] = structure.astype(np.uint8)

        row = {
            "Cr_label": 50,
            "frame": frame,
            "timestep": timestep,
            "strain": f"{strain_value:.6f}",
            "total_atoms": total,
            "bcc_atoms": int(np.sum(structure == 3)),
            "fcc_atoms": int(np.sum(structure == 1)),
            "hcp_atoms": int(np.sum(structure == 2)),
            "other_atoms": int(np.sum(structure == 0)),
            "non_bcc_atoms": int(np.sum(non_bcc)),
            "non_bcc_frac": f"{np.mean(non_bcc):.6f}",
            "omega_candidate_atoms": int(np.sum(omega_candidate)),
            "omega_candidate_frac": f"{np.mean(omega_candidate):.6f}",
            "omega_among_non_bcc_frac": f"{np.sum(omega_candidate) / max(np.sum(non_bcc), 1):.6f}",
            "orientation_subset_atoms": int(np.sum(omega_orientation_subset)),
            "orientation_subset_frac": f"{np.mean(omega_orientation_subset):.6f}",
            "orientation_subset_among_omega_frac": f"{np.sum(omega_orientation_subset) / max(np.sum(omega_candidate), 1):.6f}",
            "template_subset_atoms": int(np.sum(template_subset)),
            "template_subset_frac": f"{np.mean(template_subset):.6f}",
            "template_subset_among_omega_frac": f"{np.sum(template_subset) / max(np.sum(omega_candidate), 1):.6f}",
            "b12_A": f"{b12:.6f}",
            "candidate_mean_partial_units": f"{np.mean(partial_units[omega_candidate]) if np.any(omega_candidate) else np.nan:.6f}",
            "candidate_mean_collinearity": f"{np.mean(projection_collinearity[omega_candidate]) if np.any(omega_candidate) else np.nan:.6f}",
            "candidate_mean_orientation_score": f"{np.mean(orient_score[omega_candidate & np.isin(structure, [1, 2])]) if np.any(omega_candidate & np.isin(structure, [1, 2])) else np.nan:.6f}",
            "candidate_mean_shear": f"{np.mean(shear[omega_candidate]) if np.any(omega_candidate) else np.nan:.6f}",
            "candidate_distance_template_score_p10": f"{np.nanpercentile(template_score[omega_candidate], 10) if np.any(np.isfinite(template_score[omega_candidate])) else np.nan:.6f}",
            "candidate_distance_template_score_p50": f"{np.nanpercentile(template_score[omega_candidate], 50) if np.any(np.isfinite(template_score[omega_candidate])) else np.nan:.6f}",
            "candidate_distance_template_score_p90": f"{np.nanpercentile(template_score[omega_candidate], 90) if np.any(np.isfinite(template_score[omega_candidate])) else np.nan:.6f}",
            "candidate_vector_template_score_p10": f"{np.nanpercentile(vector_template_score[omega_candidate], 10) if np.any(np.isfinite(vector_template_score[omega_candidate])) else np.nan:.6f}",
            "candidate_vector_template_score_p50": f"{np.nanpercentile(vector_template_score[omega_candidate], 50) if np.any(np.isfinite(vector_template_score[omega_candidate])) else np.nan:.6f}",
            "candidate_vector_template_score_p90": f"{np.nanpercentile(vector_template_score[omega_candidate], 90) if np.any(np.isfinite(vector_template_score[omega_candidate])) else np.nan:.6f}",
        }
        for lo in [0.35, 0.45, 0.55]:
            row[f"projected_gate_lo{int(lo*100):02d}_atoms"] = int(np.sum(gates[lo]))
            row[f"projected_gate_lo{int(lo*100):02d}_frac"] = f"{np.mean(gates[lo]):.6f}"
            row[f"projected_cluster_lo{int(lo*100):02d}_atoms"] = int(np.sum(clustered[lo]))
            row[f"projected_cluster_lo{int(lo*100):02d}_frac"] = f"{np.mean(clustered[lo]):.6f}"
        for stype, sname in STRUCTURE_NAMES.items():
            mask = omega_candidate & (structure == stype)
            row[f"omega_{sname}_atoms"] = int(np.sum(mask))
            row[f"omega_{sname}_frac_total"] = f"{np.mean(mask):.6f}"
        for k, v in percentile(partial_units[non_bcc], [10, 50, 90]).items():
            row[f"nonbcc_partial_{k}"] = f"{v:.6f}"
        for k, v in percentile(partial_units[omega_candidate], [10, 50, 90]).items():
            row[f"omega_partial_{k}"] = f"{v:.6f}"
        rows.append(row)

    summary_path = OUT_DIR / "cr50_omega_candidate_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as fp:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(OUT_DIR / "cr50_omega_candidate_arrays.npz", **candidate_npz)
    (OUT_DIR / "omega_identification_readme.txt").write_text(
        "Main omega-like candidate criterion: non-BCC PTM + <111> non-affine "
        "displacement collinearity >= 0.70 + projected displacement 0.45-4.25 "
        "times |a/12<111>| + D2min >= 0.02 A^2 + at least 4 candidate neighbors "
        "within 3.8 A. The 0.35 and 0.55 lower projected-displacement thresholds "
        "are retained for sensitivity. The orientation-confirmed subset additionally "
        "requires PTM orientation score >= 0.86 for FCC/HCP atoms; PTM OTHER atoms "
        "are kept in the main projected class because PTM does not define an omega "
        "orientation template. The template subset additionally requires a local "
        "direction-sensitive omega vector-template Chamfer score <= 0.28 for the "
        "four [0001]omega || BCC<111> variants. The distance-only template score "
        "is exported for diagnostics but is not used as a discriminator because "
        "BCC and omega nearest-neighbor distances overlap. Atom fraction is used "
        "as the MD volume-fraction proxy.\n",
        encoding="utf-8",
    )
    (OUT_DIR / "omega_identification_cr50_done.txt").write_text(
        f"wrote {summary_path}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
