from __future__ import annotations

from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "omega_identification"
SUMMARY = OUT / "cr50_omega_candidate_summary.csv"
ARRAYS = np.load(OUT / "cr50_omega_candidate_arrays.npz")

COLORS = {
    "ink": (35, 35, 35),
    "muted": (95, 95, 95),
    "grid": (226, 226, 226),
    "blue": (47, 85, 151),
    "gold": (211, 156, 47),
    "pink": (177, 63, 114),
    "olive": (106, 143, 58),
    "green": (45, 170, 80),
    "red": (216, 74, 56),
    "gray": (117, 117, 117),
    "lightgray": (235, 235, 235),
}


def load_font(size: int, bold: bool = False):
    names = ["arialbd.ttf" if bold else "arial.ttf", "calibrib.ttf" if bold else "calibri.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_TITLE = load_font(42, True)
FONT_SUB = load_font(24)
FONT_PANEL = load_font(28, True)
FONT = load_font(22)
FONT_SMALL = load_font(18)


def read_summary():
    rows = []
    with SUMMARY.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append({k: float(v) if v not in ("", "nan", "NaN") else math.nan for k, v in row.items()})
    return rows


def draw_text(draw, xy, text, font=FONT, fill=None, anchor=None):
    draw.text(xy, text, font=font, fill=fill or COLORS["ink"], anchor=anchor)


def plot_area(panel, xvals, series, y_label, x_label, y_max=None, legend=True):
    draw = ImageDraw.Draw(panel)
    w, h = panel.size
    left, right, top, bottom = 90, 28, 52, 78
    pw, ph = w - left - right, h - top - bottom
    xmin, xmax = min(xvals), max(xvals)
    ymax = y_max if y_max is not None else max(max(vals) for _, vals, _ in series) * 1.08
    ymax = max(ymax, 1.0e-6)
    for i in range(6):
        y = top + ph - i * ph / 5
        draw.line((left, y, left + pw, y), fill=COLORS["grid"], width=1)
        val = ymax * i / 5
        draw_text(draw, (left - 8, y), f"{val:.2f}", FONT_SMALL, COLORS["muted"], anchor="rm")
    for i, xv in enumerate(np.linspace(xmin, xmax, 7)):
        x = left + (xv - xmin) / (xmax - xmin) * pw
        draw.line((x, top, x, top + ph), fill=(240, 240, 240), width=1)
        draw_text(draw, (x, top + ph + 16), f"{xv:.2f}", FONT_SMALL, COLORS["muted"], anchor="ma")
    draw.line((left, top, left, top + ph), fill=COLORS["ink"], width=2)
    draw.line((left, top + ph, left + pw, top + ph), fill=COLORS["ink"], width=2)
    for label, vals, color in series:
        pts = []
        for xv, yv in zip(xvals, vals):
            x = left + (xv - xmin) / (xmax - xmin) * pw
            y = top + ph - yv / ymax * ph
            pts.append((x, y))
        if len(pts) > 1:
            draw.line(pts, fill=color, width=4)
        for x, y in pts:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="white", outline=color, width=3)
    draw_text(draw, (left + pw / 2, h - 24), x_label, FONT_SMALL, COLORS["muted"], anchor="ma")
    draw_text(draw, (left, top - 18), y_label, FONT_SMALL, COLORS["muted"], anchor="ls")
    if legend:
        lx, ly = left + pw - 230, top + 4
        for i, (label, _, color) in enumerate(series):
            yy = ly + i * 25
            draw.line((lx, yy + 9, lx + 28, yy + 9), fill=color, width=4)
            draw.ellipse((lx + 10, yy + 4, lx + 20, yy + 14), fill="white", outline=color, width=2)
            draw_text(draw, (lx + 36, yy), label, FONT_SMALL, COLORS["ink"])


def stacked_bars(panel, xvals, components, y_label, x_label):
    draw = ImageDraw.Draw(panel)
    w, h = panel.size
    left, right, top, bottom = 90, 28, 52, 78
    pw, ph = w - left - right, h - top - bottom
    ymax = max(sum(vals[i] for _, vals, _ in components) for i in range(len(xvals))) * 1.12
    ymax = max(ymax, 0.01)
    xmin, xmax = min(xvals), max(xvals)
    for i in range(6):
        y = top + ph - i * ph / 5
        draw.line((left, y, left + pw, y), fill=COLORS["grid"], width=1)
        draw_text(draw, (left - 8, y), f"{ymax*i/5:.2f}", FONT_SMALL, COLORS["muted"], anchor="rm")
    bar_w = max(14, pw / (len(xvals) * 2.5))
    for i, xv in enumerate(xvals):
        x = left + (xv - xmin) / (xmax - xmin) * pw
        base = top + ph
        for label, vals, color in components:
            y = base - vals[i] / ymax * ph
            draw.rectangle((x - bar_w / 2, y, x + bar_w / 2, base), fill=color)
            base = y
        draw_text(draw, (x, top + ph + 16), f"{xv:.3f}", FONT_SMALL, COLORS["muted"], anchor="ma")
    draw.line((left, top, left, top + ph), fill=COLORS["ink"], width=2)
    draw.line((left, top + ph, left + pw, top + ph), fill=COLORS["ink"], width=2)
    draw_text(draw, (left + pw / 2, h - 24), x_label, FONT_SMALL, COLORS["muted"], anchor="ma")
    draw_text(draw, (left, top - 18), y_label, FONT_SMALL, COLORS["muted"], anchor="ls")
    lx, ly = left + pw - 210, top + 4
    for i, (label, _, color) in enumerate(components):
        yy = ly + i * 25
        draw.rectangle((lx, yy + 3, lx + 18, yy + 18), fill=color)
        draw_text(draw, (lx + 26, yy), label, FONT_SMALL, COLORS["ink"])


def histogram_panel(panel, frames):
    draw = ImageDraw.Draw(panel)
    w, h = panel.size
    left, right, top, bottom = 90, 28, 52, 78
    pw, ph = w - left - right, h - top - bottom
    xmax = 5.0
    bins = np.linspace(0.0, xmax, 56)
    densities = []
    for frame, _color in frames:
        partial = ARRAYS[f"frame_{frame}_partial_units"]
        st = ARRAYS[f"frame_{frame}_structure"]
        values = partial[st != 3]
        hist, edges = np.histogram(values, bins=bins, density=True)
        densities.append((hist, edges))
    ymax = max(float(np.max(hist)) for hist, _ in densities) * 1.10
    for i in range(6):
        y = top + ph - i * ph / 5
        draw.line((left, y, left + pw, y), fill=COLORS["grid"], width=1)
        draw_text(draw, (left - 8, y), f"{ymax*i/5:.1f}", FONT_SMALL, COLORS["muted"], anchor="rm")
    # Shaded main threshold window.
    x0 = left + 0.45 / xmax * pw
    x1 = left + 4.25 / xmax * pw
    draw.rectangle((x0, top, x1, top + ph), fill=(235, 242, 252))
    for (frame, color), (hist, edges) in zip(frames, densities):
        pts = []
        for j, val in enumerate(hist):
            x = left + ((edges[j] + edges[j + 1]) / 2) / xmax * pw
            y = top + ph - val / ymax * ph
            pts.append((x, y))
        draw.line(pts, fill=color, width=4)
    draw.line((left, top, left, top + ph), fill=COLORS["ink"], width=2)
    draw.line((left, top + ph, left + pw, top + ph), fill=COLORS["ink"], width=2)
    for xv in range(0, 6):
        x = left + xv / xmax * pw
        draw_text(draw, (x, top + ph + 16), str(xv), FONT_SMALL, COLORS["muted"], anchor="ma")
    draw_text(draw, (left + pw / 2, h - 24), "Projected displacement / |a/12<111>|", FONT_SMALL, COLORS["muted"], anchor="ma")
    draw_text(draw, (left, top - 18), "Density", FONT_SMALL, COLORS["muted"], anchor="ls")
    lx, ly = left + pw - 190, top + 4
    for i, (frame, color) in enumerate(frames):
        yy = ly + i * 25
        draw.line((lx, yy + 9, lx + 28, yy + 9), fill=color, width=4)
        draw_text(draw, (lx + 36, yy), f"epsilon={frame/1000:.3f}", FONT_SMALL, COLORS["ink"])


def make_figure():
    rows = read_summary()
    x = np.array([r["strain"] for r in rows], dtype=float)
    total = np.array([r["total_atoms"] for r in rows], dtype=float)
    img = Image.new("RGB", (2600, 1900), "white")
    draw = ImageDraw.Draw(img)
    draw_text(draw, (1300, 54), "Cr50 omega-like identification from tensile MD", FONT_TITLE, anchor="ma")
    draw_text(
        draw,
        (1300, 95),
        "Main criterion: PTM non-BCC + <111> non-affine projection + connected-neighbor filter; atom fraction approximates volume fraction",
        FONT_SUB,
        COLORS["muted"],
        anchor="ma",
    )

    panel_w, panel_h = 1180, 760
    positions = [(110, 160), (1310, 160), (110, 970), (1310, 970)]
    panels = []
    for pos in positions:
        panel = Image.new("RGB", (panel_w, panel_h), "white")
        panels.append(panel)

    pdraw = ImageDraw.Draw(panels[0])
    draw_text(pdraw, (20, 14), "a  Fraction evolution", FONT_PANEL)
    plot_area(
        panels[0],
        x,
        [
            ("PTM non-BCC", [r["non_bcc_frac"] for r in rows], COLORS["gray"]),
            ("omega-like candidate", [r["omega_candidate_frac"] for r in rows], COLORS["blue"]),
            ("vector-template subset", [r["template_subset_frac"] for r in rows], COLORS["gold"]),
            ("orientation subset", [r["orientation_subset_frac"] for r in rows], COLORS["pink"]),
        ],
        "Atom fraction",
        "Engineering strain",
    )

    pdraw = ImageDraw.Draw(panels[1])
    draw_text(pdraw, (20, 14), "b  Projection-threshold sensitivity", FONT_PANEL)
    plot_area(
        panels[1],
        x,
        [
            ("0.35 x a/12<111>", [r["projected_cluster_lo35_frac"] for r in rows], COLORS["olive"]),
            ("0.45 x a/12<111>", [r["projected_cluster_lo45_frac"] for r in rows], COLORS["blue"]),
            ("0.55 x a/12<111>", [r["projected_cluster_lo55_frac"] for r in rows], COLORS["gold"]),
        ],
        "Connected candidate fraction",
        "Engineering strain",
    )

    pdraw = ImageDraw.Draw(panels[2])
    draw_text(pdraw, (20, 14), "c  PTM labels inside omega-like candidates", FONT_PANEL)
    stacked_bars(
        panels[2],
        x,
        [
            ("FCC-like", np.array([r["omega_FCC_atoms"] for r in rows]) / total, COLORS["green"]),
            ("Other", np.array([r["omega_OTHER_atoms"] for r in rows]) / total, COLORS["gray"]),
            ("HCP-like", np.array([r["omega_HCP_atoms"] for r in rows]) / total, COLORS["red"]),
        ],
        "Atom fraction",
        "Engineering strain",
    )

    pdraw = ImageDraw.Draw(panels[3])
    draw_text(pdraw, (20, 14), "d  <111> projected displacement in non-BCC atoms", FONT_PANEL)
    histogram_panel(
        panels[3],
        [
            (100, COLORS["blue"]),
            (125, COLORS["gold"]),
            (150, COLORS["pink"]),
            (300, COLORS["olive"]),
        ],
    )

    for panel, pos in zip(panels, positions):
        img.paste(panel, pos)
    out = OUT / "cr50_omega_identification_summary.png"
    img.save(out, dpi=(320, 320))

    compact_cols = [
        "frame",
        "strain",
        "non_bcc_frac",
        "omega_candidate_frac",
        "template_subset_frac",
        "orientation_subset_frac",
        "omega_among_non_bcc_frac",
        "candidate_mean_partial_units",
        "candidate_mean_collinearity",
        "candidate_mean_shear",
    ]
    with (OUT / "cr50_omega_identification_compact.csv").open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=compact_cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row[col] for col in compact_cols})
    print(out)


if __name__ == "__main__":
    make_figure()
