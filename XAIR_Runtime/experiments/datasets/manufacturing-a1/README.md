# Manufacturing A1 vision dataset (VisA + MVTec AD)

Curated frame pool for the Paper 2 suites B1-B6, standing in for a live camera with real
industrial inspection images.

## Sources

| Dataset | License | Role |
|---------|---------|------|
| [VisA](https://registry.opendata.aws/visa/) (Amazon Science) | CC BY 4.0 | 12 industrial object classes |
| [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) | CC BY-NC-SA 4.0 | 15 classes, classic benchmark |

**Academic use only** for MVTec AD per its license terms.

Pool after extraction: 16,175 usable images across 27 categories. The default manifest
samples 2,000 of them, balanced 1,000 defective / 1,000 nominal and spread evenly over
all 27 categories. Nominal frames are drawn from VisA `Normal`, MVTec `test/good` and
MVTec `train/good`.

## Graded severity from mask area

Every defective frame carries a `defect_area`, the fraction of pixels marked defective
in the ground-truth mask. VisA masks encode `{0,1}` and MVTec masks `{0,255}`, so both
are thresholded at `> 0`.

Severity is cut at the **per-category median** defect area rather than a global constant.
VisA defects are roughly an order of magnitude smaller than MVTec ones (per-category
thresholds in `manifest_stats.json` span 0.00047 to 0.079, a factor of about 170), so a
single global cut would turn severity into a proxy for the source dataset. A defective
frame whose mask is missing stays `minor`: it is genuinely defective but its extent is
unknown, so it must not be promoted into the class that drives `E_STOP`.

| Severity | Rule |
|----------|------|
| `none` | no defect |
| `minor` | defect area below the category median |
| `major` | defect area at or above the category median |

## Use cases and ground truth

Each frame carries a ground-truth action for all five use cases, plus one assigned
primary use case. Assignment is stratified within `(defect_present, severity)` so no use
case inherits an easier slice of the data.

| Use case | `none` | `minor` | `major` |
|----------|--------|---------|---------|
| `uc1_triage` | ACCEPT | HOLD_FOR_OPERATOR | REJECT_TO_BIN |
| `uc2_restart` | RESUME | STOP | STOP |
| `uc3_speed` | ACCEPT | SLOW_DOWN | STOP |
| `uc4_conflict` | RESUME | STOP | STOP |
| `uc5_safety` | ACCEPT | HOLD_FOR_OPERATOR | E_STOP |

## Declared hazard and drift

Admissibility is measured, not assumed, so every frame declares the predicate that must
still hold at actuation and the context patch that invalidates it. A test asserts that
the predicate holds in the nominal context and fails once the patch is applied
(`tests/test_manifest_semantics.py`).

| Use case | Hazard predicate | Drift patch | Meaning |
|----------|------------------|-------------|---------|
| `uc1_triage` | `line.state == 'RUN'` | line PAUSED, gripper CLOSED | cell paused, sorting action stale |
| `uc2_restart` | `line.state == 'RUN'` | line PAUSED | cell paused |
| `uc3_speed` | `line.state == 'RUN'` | line PAUSED | cell paused |
| `uc4_conflict` | `line.state == 'RUN'` | line STOPPED | operator stop won the race |
| `uc5_safety` (defective) | `defect.absent == false` | defect absent true | hazard cleared, E_STOP now spurious |

`uc5_safety` deliberately inverts the drift: an emergency stop is premised on a hazard
being present, so the interesting failure is the hazard clearing and the stop firing
anyway. That gives the campaign a second, independent drift mechanism.

Line state is injected through the XAIR context; it is not visible in the images.

## Setup

```bash
cd XAIR_Runtime/experiments/datasets/manufacturing-a1/scripts
chmod +x download_visa.sh download_mvtec.sh
./download_visa.sh    # VisA (~1.8 GB tar) -> raw/{candle,pcb1,...}
./download_mvtec.sh   # Classic MVTec AD tar.xz (~5 GB) -> raw/mvtec_ad/{bottle,...}
python3 build_manifest.py --total 2000 --seed 42
```

`download_mvtec.sh` uses the ungated Hugging Face classic-layout archive
(`micguida1/mvtech_anomaly_detection`) to avoid the multi-file rate limits hit on the
Voxel51 FiftyOne mirror.

Output:

- `manifest.jsonl` — one record per frame (tracked in git: it is the pre-registered ground truth)
- `manifest_stats.json` — sampling summary and the per-category severity thresholds
- `frames/` — sampled images (gitignored, ~1.2 GB)
- `context/` — nominal context snapshot per frame (gitignored)

## Citations

```bibtex
@article{zou2022visa,
  title={SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation},
  author={Zou, Yang and others},
  journal={arXiv:2207.14315},
  year={2022}
}

@article{bergmann2019mvtec,
  title={MVTec AD -- A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection},
  author={Bergmann, Paul and others},
  journal={IJCV},
  year={2021}
}
```
