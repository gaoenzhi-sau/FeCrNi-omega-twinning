# Interatomic-potential files

The simulations call these files together using:

```text
pair_style hybrid/overlay eam/alloy eam/fs
pair_coeff * * eam/alloy FeCrNi_d.eam.alloy Fe Cr Ni
pair_coeff * * eam/fs FeCrNi_s.eam.fs Fe Cr Ni
```

They are the exact files used for the reported calculations. The manuscript
describes the potential construction and validation and cites G. Bonny et al.,
Journal of Nuclear Materials 484 (2017) 42-50. Verify redistribution rights
and add the original download or generation provenance before public release.
Exact SHA-256 checksums are recorded in the repository-level
`SHA256SUMS.txt`.
