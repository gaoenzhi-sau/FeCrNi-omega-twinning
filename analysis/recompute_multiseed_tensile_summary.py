from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import math

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SETS = [(name, ROOT / "tensile" / name) for name in ("S1", "S2", "S3")]
OUT = ROOT / "data" / "summary"
OUT.mkdir(parents=True, exist_ok=True)


def first_frame_composition(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as fp:
        fp.readline(); fp.readline(); fp.readline()
        n = int(fp.readline())
        fp.readline(); fp.readline(); fp.readline(); fp.readline()
        counts = Counter()
        for _ in range(n):
            parts = fp.readline().split()
            counts[parts[1]] += 1
    return counts["1"] / n * 100, counts["2"] / n * 100, counts["3"] / n * 100, n


def read_stress(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        values = line.split()
        if len(values) >= 4:
            x, z = float(values[0]), float(values[3])
            if math.isfinite(x) and math.isfinite(z):
                rows.append((x, z))
    data = np.asarray(rows)
    return data[:, 0], data[:, 1]


def smooth(y, window=101):
    pad = window // 2
    return np.convolve(np.pad(y, (pad, pad), mode="edge"), np.ones(window) / window, mode="valid")


def summarize(set_name, directory):
    cr_label = int(directory.name.removeprefix("Cr"))
    initial_frame = ROOT / "data" / "tensile_initial_frames" / f"{set_name}_Cr{cr_label}_initial.lammpstrj"
    fe, cr_actual, ni, n = first_frame_composition(initial_frame)
    x, y_raw = read_stress(directory / "stress_strain.dat")
    y = smooth(y_raw)
    elastic = (x >= 0.005) & (x <= 0.030)
    peak_region = x <= 0.15
    peak_idx = np.flatnonzero(peak_region)[np.argmax(y[peak_region])]
    flow = (x >= 0.15) & (x <= 0.25)
    return {
        "set": set_name,
        "Cr_label": cr_label,
        "Fe_at_pct": fe,
        "Cr_at_pct": cr_actual,
        "Ni_at_pct": ni,
        "n_atoms": n,
        "E_GPa": float(np.polyfit(x[elastic], y[elastic], 1)[0]),
        "peak_stress_GPa": float(y[peak_idx]),
        "peak_strain": float(x[peak_idx]),
        "flow_stress_GPa": float(y[flow].mean()),
        "x": x,
        "stress_raw": y_raw,
        "stress_smooth": y,
    }


rows = []
for set_name, root in SETS:
    for cr in range(45, 51):
        rows.append(summarize(set_name, root / f"Cr{cr}"))

raw_fields = ["set", "Cr_label", "Fe_at_pct", "Cr_at_pct", "Ni_at_pct", "n_atoms", "E_GPa", "peak_stress_GPa", "peak_strain", "flow_stress_GPa"]
with (OUT / "multiseed_metrics_raw.csv").open("w", newline="", encoding="utf-8-sig") as fp:
    writer = csv.DictWriter(fp, fieldnames=raw_fields)
    writer.writeheader()
    writer.writerows({k: row[k] for k in raw_fields} for row in rows)

stats = []
for cr in range(45, 51):
    group = [row for row in rows if row["Cr_label"] == cr]
    stat = {"Cr_label": cr, "n": len(group)}
    for field in ("Fe_at_pct", "Cr_at_pct", "Ni_at_pct", "E_GPa", "peak_stress_GPa", "peak_strain", "flow_stress_GPa"):
        values = np.asarray([row[field] for row in group])
        stat[field + "_mean"] = float(values.mean())
        stat[field + "_sd"] = float(values.std(ddof=1))
    stats.append(stat)

stat_fields = list(stats[0])
with (OUT / "multiseed_metrics_summary.csv").open("w", newline="", encoding="utf-8-sig") as fp:
    writer = csv.DictWriter(fp, fieldnames=stat_fields)
    writer.writeheader(); writer.writerows(stats)

# Mean curves and pointwise sample standard deviations for plotting.
with (OUT / "multiseed_mean_curves.csv").open("w", newline="", encoding="utf-8-sig") as fp:
    writer = csv.writer(fp)
    header = ["strain"]
    for cr in range(45, 51):
        header += [f"Cr{cr}_mean_GPa", f"Cr{cr}_sd_GPa"]
    writer.writerow(header)
    grid = rows[0]["x"]
    for i, strain in enumerate(grid):
        record = [strain]
        for cr in range(45, 51):
            curves = np.asarray([row["stress_smooth"] for row in rows if row["Cr_label"] == cr])
            record += [curves[:, i].mean(), curves[:, i].std(ddof=1)]
        writer.writerow(record)

for row in rows:
    print(f"{row['set']} Cr{row['Cr_label']}: peak={row['peak_stress_GPa']:.4f} @ {row['peak_strain']:.4f}; flow={row['flow_stress_GPa']:.4f}; E={row['E_GPa']:.3f}")
print("--- mean +/- SD ---")
for row in stats:
    print(f"Cr{row['Cr_label']}: peak={row['peak_stress_GPa_mean']:.4f}+/-{row['peak_stress_GPa_sd']:.4f}; flow={row['flow_stress_GPa_mean']:.4f}+/-{row['flow_stress_GPa_sd']:.4f}; E={row['E_GPa_mean']:.3f}+/-{row['E_GPa_sd']:.3f}; strain={row['peak_strain_mean']:.5f}+/-{row['peak_strain_sd']:.5f}")
