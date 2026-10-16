from __future__ import annotations

from pathlib import Path

import numpy as np
from ovito.io import import_file
from ovito.modifiers import PolyhedralTemplateMatchingModifier, PythonScriptModifier, SliceModifier
from ovito.vis import TachyonRenderer, Viewport


ROOT = Path(__file__).resolve().parents[1]
TRAJ = ROOT / "zenodo_large_files" / "raw_trajectory" / "Cr50_S1_md.lammpstrj"
OUT = ROOT / "data" / "omega_identification"
OUT.mkdir(parents=True, exist_ok=True)
ARRAYS = np.load(OUT / "cr50_omega_candidate_arrays.npz")
FRAMES = [75, 100, 125, 150, 300]


PTM_COLORS = {
    0: np.array([0.58, 0.58, 0.58]),  # Other/distorted
    1: np.array([0.13, 0.68, 0.30]),  # FCC-like
    2: np.array([0.86, 0.22, 0.16]),  # HCP-like
    3: np.array([0.24, 0.34, 0.92]),  # BCC
}
OMEGA_COLOR = np.array([1.00, 0.77, 0.05])
TEMPLATE_COLOR = np.array([1.00, 0.40, 0.05])


def color_by_omega_candidate(frame, data):
    structure = np.asarray(data.particles["Structure Type"], dtype=np.int32)
    ids = np.asarray(data.particles["Particle Identifier"], dtype=np.int64)
    colors = np.empty((len(ids), 3), dtype=float)
    for stype, color in PTM_COLORS.items():
        colors[structure == stype] = color

    key = f"frame_{frame}_ids"
    if key in ARRAYS:
        sorted_ids = ARRAYS[key]
        order = np.searchsorted(sorted_ids, ids)
        valid = (order >= 0) & (order < len(sorted_ids)) & (sorted_ids[order] == ids)
        candidate = np.zeros(len(ids), dtype=bool)
        template = np.zeros(len(ids), dtype=bool)
        candidate[valid] = ARRAYS[f"frame_{frame}_omega_candidate"][order[valid]].astype(bool)
        template[valid] = ARRAYS[f"frame_{frame}_omega_template_subset"][order[valid]].astype(bool)
        colors[candidate] = OMEGA_COLOR
        data.particles_.create_property("OmegaCandidate", data=candidate.astype(np.int32))
        data.particles_.create_property("OmegaTemplateSubset", data=template.astype(np.int32))

    data.particles_.create_property("Color", data=colors)


pipeline = import_file(str(TRAJ), multiple_frames=True)
pipeline.modifiers.append(PolyhedralTemplateMatchingModifier(output_orientation=True, output_interatomic_distance=True, output_rmsd=True))
pipeline.modifiers.append(PythonScriptModifier(function=color_by_omega_candidate))

data0 = pipeline.compute(0)
cell = data0.cell
center_y = cell[1, 3] + cell[1, 1] / 2.0
pipeline.modifiers.append(SliceModifier(normal=(0, 1, 0), distance=center_y, slab_width=15.0))
pipeline.add_to_scene()

mins = cell[:, 3]
lengths = [cell[0, 0], cell[1, 1], cell[2, 2]]
center = (mins[0] + lengths[0] / 2.0, mins[1] + lengths[1] / 2.0, mins[2] + lengths[2] / 2.0)
vp = Viewport(type=Viewport.Type.Ortho)
vp.camera_dir = (0.0, 1.0, 0.0)
vp.camera_up = (0.0, 0.0, 1.0)
vp.camera_pos = (center[0], center[1] - 300.0, center[2])
vp.fov = max(lengths[0], lengths[2]) * 1.08

lines = []
for frame in FRAMES:
    eps = frame / 1000.0
    fname = OUT / f"Cr50_omega_candidate_slice_{eps:.3f}.png"
    vp.render_image(
        filename=str(fname),
        size=(1000, 1000),
        frame=frame,
        renderer=TachyonRenderer(shadows=False, ambient_occlusion=False),
    )
    lines.append(f"{fname}|{eps:.3f}")

(OUT / "render_cr50_omega_candidate_slices.txt").write_text("\n".join(lines), encoding="utf-8")
