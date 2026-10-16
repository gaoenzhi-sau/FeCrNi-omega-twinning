from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[1]
GSFE_DIR = ROOT / "gsfe"
OUT = ROOT / "data" / "summary" / "gsfe_composition_status.csv"


PATHS = {
    45: {
        "full": GSFE_DIR / "Cr45" / "results" / "full_slip.dat",
        "twin": GSFE_DIR / "Cr45" / "results" / "twinning.dat",
        "omega": GSFE_DIR / "Cr45" / "results" / "omega_path.dat",
    },
}
for cr in [46, 47, 48, 49, 50]:
    PATHS[cr] = {
        "full": GSFE_DIR / f"Cr{cr}" / "full_slip" / "gsfe.dat",
        "twin": GSFE_DIR / f"Cr{cr}" / "twinning" / "gsfe.dat",
        "omega": GSFE_DIR / f"Cr{cr}" / "omega_path" / "gsfe.dat",
    }


def infer_from_input(cr: int, path_name: str) -> tuple[float | None, float | None, float | None, str]:
    if cr == 45:
        return None, None, None, "composition not re-parsed from 45 dat-only folder"
    folder = PATHS[cr][path_name].parent
    inputs = list(folder.glob("input.lammps"))
    if not inputs:
        return None, None, None, "missing input file"
    text = inputs[0].read_text(encoding="utf-8", errors="ignore")
    cr_frac = ni_frac_remaining = None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 6 and parts[:4] == ["set", "type", "1", "type/fraction"]:
            if parts[4] == "2":
                cr_frac = float(parts[5])
            if parts[4] == "3":
                ni_frac_remaining = float(parts[5])
    if cr_frac is None or ni_frac_remaining is None:
        return None, None, None, "set commands not found"
    cr_at = 100.0 * cr_frac
    ni_at = 100.0 * (1.0 - cr_frac) * ni_frac_remaining
    fe_at = 100.0 - cr_at - ni_at
    return fe_at, cr_at, ni_at, f"input {inputs[0].name}"


def parse_dump0(cr: int, path_name: str) -> tuple[float | None, float | None, float | None, str]:
    if cr == 45:
        return None, None, None, "no dump0 path inferred for 45"
    folder = PATHS[cr][path_name].parent
    dump = folder / "dump.comp.0.lammpstrj"
    if not dump.exists():
        return None, None, None, "missing dump.comp.0.lammpstrj"
    with dump.open("r", encoding="utf-8", errors="ignore") as fp:
        fp.readline()
        fp.readline()
        fp.readline()
        n = int(fp.readline().strip())
        fp.readline()
        for _ in range(3):
            fp.readline()
        fp.readline()
        counts: Counter[str] = Counter()
        for _ in range(n):
            parts = fp.readline().split()
            if len(parts) >= 2:
                counts[parts[1]] += 1
    return counts["1"] / n * 100.0, counts["2"] / n * 100.0, counts["3"] / n * 100.0, "dump.comp.0"


rows = []
for cr in sorted(PATHS):
    for path_name in ["full", "twin", "omega"]:
        fe_i, cr_i, ni_i, src_i = infer_from_input(cr, path_name)
        fe_d, cr_d, ni_d, src_d = parse_dump0(cr, path_name)
        ni_check = ni_d if ni_d is not None else ni_i
        valid = "" if ni_check is None else abs(ni_check - 15.0) <= 0.6
        rows.append({
            "Cr_label": cr,
            "path": path_name,
            "Fe_input_at_pct": fe_i,
            "Cr_input_at_pct": cr_i,
            "Ni_input_at_pct": ni_i,
            "Fe_dump_at_pct": fe_d,
            "Cr_dump_at_pct": cr_d,
            "Ni_dump_at_pct": ni_d,
            "valid_fixed_Ni15": valid,
            "note": f"{src_i}; {src_d}",
        })

fields = list(rows[0].keys())
with OUT.open("w", newline="", encoding="utf-8-sig") as fp:
    writer = csv.DictWriter(fp, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(OUT)
for row in rows:
    if row["Cr_label"] == 50:
        print(row)
