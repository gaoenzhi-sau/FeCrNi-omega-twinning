from __future__ import annotations

from pathlib import Path

import numpy as np
from ovito.io import import_file
from ovito.modifiers import AtomicStrainModifier, PolyhedralTemplateMatchingModifier


ROOT = Path(__file__).resolve().parents[1]
TRAJ = ROOT / "zenodo_large_files" / "raw_trajectory" / "Cr50_S1_md.lammpstrj"
OUT = ROOT / "data" / "omega_identification"
ARRAYS = np.load(OUT / "cr50_omega_candidate_arrays.npz")
FRAMES = [75, 100, 125, 150, 300]


pipeline = import_file(str(TRAJ), multiple_frames=True)
pipeline.modifiers.append(
    PolyhedralTemplateMatchingModifier(
        output_orientation=True,
        output_interatomic_distance=True,
        output_rmsd=True,
    )
)
strain = AtomicStrainModifier(cutoff=3.8)
strain.reference_frame = 0
strain.eliminate_cell_deformation = True
strain.output_nonaffine_squared_displacements = True
strain.output_strain_tensors = True
pipeline.modifiers.append(strain)

out_path = OUT / "Cr50_omega_candidate_selected_frames.dump"
with out_path.open("w", encoding="utf-8", newline="\n") as fp:
    for frame in FRAMES:
        data = pipeline.compute(frame)
        timestep = int(data.attributes.get("Timestep", frame * 1000))
        ids = np.asarray(data.particles["Particle Identifier"], dtype=np.int64)
        types = np.asarray(data.particles["Particle Type"], dtype=np.int32)
        pos = np.asarray(data.particles["Position"], dtype=float)
        structure = np.asarray(data.particles["Structure Type"], dtype=np.int32)
        d2 = np.asarray(data.particles["Nonaffine Squared Displacement"], dtype=float)
        shear = np.asarray(data.particles["Shear Strain"], dtype=float)

        sorted_ids = ARRAYS[f"frame_{frame}_ids"]
        order = np.searchsorted(sorted_ids, ids)
        valid = (order >= 0) & (order < len(sorted_ids)) & (sorted_ids[order] == ids)
        omega = np.zeros(len(ids), dtype=np.int32)
        template = np.zeros(len(ids), dtype=np.int32)
        variant = np.zeros(len(ids), dtype=np.int32)
        partial = np.full(len(ids), np.nan, dtype=float)
        collinearity = np.full(len(ids), np.nan, dtype=float)
        orient_score = np.full(len(ids), np.nan, dtype=float)
        vector_score = np.full(len(ids), np.nan, dtype=float)
        omega[valid] = ARRAYS[f"frame_{frame}_omega_candidate"][order[valid]].astype(np.int32)
        template[valid] = ARRAYS[f"frame_{frame}_omega_template_subset"][order[valid]].astype(np.int32)
        variant[valid] = ARRAYS[f"frame_{frame}_variant"][order[valid]].astype(np.int32) + 1
        partial[valid] = ARRAYS[f"frame_{frame}_partial_units"][order[valid]]
        collinearity[valid] = ARRAYS[f"frame_{frame}_projection_collinearity"][order[valid]]
        orient_score[valid] = ARRAYS[f"frame_{frame}_orientation_score"][order[valid]]
        vector_score[valid] = ARRAYS[f"frame_{frame}_omega_vector_template_score"][order[valid]]

        cell = np.asarray(data.cell[:], dtype=float)
        xlo, xhi = cell[0, 3], cell[0, 3] + cell[0, 0]
        ylo, yhi = cell[1, 3], cell[1, 3] + cell[1, 1]
        zlo, zhi = cell[2, 3], cell[2, 3] + cell[2, 2]
        fp.write("ITEM: TIMESTEP\n")
        fp.write(f"{timestep}\n")
        fp.write("ITEM: NUMBER OF ATOMS\n")
        fp.write(f"{len(ids)}\n")
        fp.write("ITEM: BOX BOUNDS pp pp pp\n")
        fp.write(f"{xlo:.10f} {xhi:.10f}\n")
        fp.write(f"{ylo:.10f} {yhi:.10f}\n")
        fp.write(f"{zlo:.10f} {zhi:.10f}\n")
        fp.write(
            "ITEM: ATOMS id type x y z structure_type omega_candidate "
            "omega_template_subset omega_variant partial111 collinearity111 "
            "orientation_score omega_vector_template_score d2min shear_strain\n"
        )
        for i in range(len(ids)):
            fp.write(
                f"{ids[i]} {types[i]} {pos[i,0]:.6f} {pos[i,1]:.6f} {pos[i,2]:.6f} "
                f"{structure[i]} {omega[i]} {template[i]} {variant[i]} "
                f"{partial[i]:.6f} {collinearity[i]:.6f} {orient_score[i]:.6f} "
                f"{vector_score[i]:.6f} {d2[i]:.6f} {shear[i]:.6f}\n"
            )

print(out_path)
