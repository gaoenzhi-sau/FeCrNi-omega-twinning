# Data and code for COMMAT-D-26-04506

This repository contains the simulation inputs, analysis code, processed data,
and selected atomistic trajectories supporting the manuscript:

> GSFE constrained atomistic analysis of transient omega like shuffling and
> twinning in Cr rich BCC FeCrNi alloys

## Directory structure

- `potentials/`: the two Fe-Cr-Ni EAM component files used with
  `pair_style hybrid/overlay`.
- `gsfe/`: generalized stacking-fault-energy inputs, logs, and curve data for
  the full-slip, twinning, and omega-like pathways.
- `tensile/`: three statistically independent tensile realizations (S1-S3),
  each covering the Cr45-Cr50 compositions.
- `analysis/`: scripts used for multi-seed tensile statistics, omega-like
  candidate identification, cluster tracking, local chemistry, validation,
  and plotting.
- `data/summary/`: processed numerical data underlying the reported trends.
- `data/omega_identification/`: structural-identification outputs and
  sensitivity checks.
- `data/tensile_initial_frames/`: the first equilibrated trajectory frame for
  every tensile simulation, used to verify realized compositions.
- `zenodo_large_files/`: files larger than GitHub's normal file-size limit;
  deposit these files on Zenodo and link the DOI from the GitHub repository.

## Three tensile realizations

- `S1`, `S2`, and `S3` denote three statistically independent realizations.
- Each realization uses independent random seeds for chemical substitution and
  initial velocities.
- The exact random seeds are preserved in the corresponding `input.lammps`
  files.

## Software

- LAMMPS, 11 February 2026 build (as recorded in the deposited log files).
- Python 3.
- NumPy.
- OVITO Python module.
- Pillow, required only for the supplied raster plotting script.

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Running the tensile calculations

Run LAMMPS from an individual case directory, for example:

```bash
cd tensile/S1/Cr50
lmp -in input.lammps
```

The input files use repository-relative paths to the common starting model and
potential files. The tensile simulation uses a 0.001 ps timestep and 300000
deformation steps, corresponding to 300 ps and a target engineering strain of
approximately 0.30. Comments in the public copies were corrected accordingly;
the numerical commands and original output data were not changed.

## Running the GSFE calculations

Run LAMMPS from a pathway directory, for example:

```bash
cd gsfe/Cr50/twinning
lmp -in input.lammps
```

For Cr46-Cr50, the deposited files are copies of the original inputs with only
repository-relative file paths substituted. The original Cr45 input files were
not retained. The Cr45 inputs included here are transparently marked
reconstructions from the otherwise identical Cr46-Cr50 input series, using
Cr=0.45, nominal Ni=0.15, and the same series random seeds. The original Cr45
curve data are preserved in `gsfe/Cr45/results/`; no original Cr45 log files
are claimed or fabricated.

## Reproducing processed results

Recompute multi-seed tensile statistics:

```bash
python analysis/recompute_multiseed_tensile_summary.py
```

Recompute the Cr50-S1 omega-like candidate analysis:

```bash
python analysis/omega_identification_cr50.py
python analysis/track_cr50_omega_clusters.py
python analysis/compute_cr50_local_chemistry.py
```

The complete Cr50-S1 trajectory used by these scripts is stored under
`zenodo_large_files/raw_trajectory/`. It should be uploaded to Zenodo rather
than committed to ordinary Git history. A smaller selected-frame trajectory is
also provided for inspection.

## Potential

The simulations use the two supplied files with the element mapping
`Fe Cr Ni` and `pair_style hybrid/overlay eam/alloy eam/fs`. The potential is
described and validated in the manuscript; see the cited work by Bonny et al.,
Journal of Nuclear Materials 484 (2017) 42-50. Checksums are provided in
`SHA256SUMS.txt` so that the exact files used can be verified.

## Data scope

All input files, random seeds, scalar outputs, processed figure data, and the
complete trajectory used for the detailed omega-like structural analysis are
included. The other 17 full tensile trajectories and all per-displacement GSFE
relaxation dumps are not duplicated because the original collection is about
620 GB. They can be regenerated from the deposited inputs and seeds. Initial
frames and the numerical outputs used for statistics are included for all
18 tensile cases.

## Before public release

1. Upload all files except `zenodo_large_files/` to a public GitHub repository.
2. Upload `zenodo_large_files/` to Zenodo and publish the record.
3. Add the final GitHub URL and Zenodo DOI to the manuscript's Data Availability
   statement and to the response to the editor.
4. Select licenses with all co-authors. MIT is commonly used for code and
   CC BY 4.0 or CC0 for author-owned data. The potential files should follow
   their original terms and attribution.

