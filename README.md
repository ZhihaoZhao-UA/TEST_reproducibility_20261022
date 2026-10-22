# Risk Bounds for Neural Network Path Averaging

Reproducibility archive for the research project **Risk Bounds for Neural Network Path Averaging**, imported from `TEST_reproducibility_20261022.zip`.

The project studies convex averaging of neural-network predictions along training paths, with independent-sample aggregation, population risk bounds, leverage and effective-dimension diagnostics, and comparisons with checkpoint selection, equal averaging, and cross-validation. This repository contains NumPy implementations, experiment protocols, stored numerical results, prediction caches, public regression inputs, and supplementary material.

## Scope and provenance

This is a research archive rather than an installable Python package. Stored results allow inspection without rerunning all training studies. Unfavorable comparisons, unsuccessful criteria, and vacuous bounds are retained.

The archive contains records from several manuscript revisions. In particular, [the October 16 closure note](closure_20261016/README.txt) identifies the current submission's S11 as the finite-population guarantee; older local-training and curvature studies remain for provenance. Section and table numbers in [the original README](README.txt) and the bundled supplementary source refer to their respective revisions and should not be treated as a single synchronized submission. Use the filenames below to identify experiments.

## Environment and dependencies

The archived environment in [MANIFEST.json](MANIFEST.json) records:

| Component | Recorded version / requirement |
| --- | --- |
| Python | 3.12.14; use Python 3.12 for reproduction |
| NumPy | 2.3.5 |
| Other Python imports | Standard library and modules included in this repository |
| Compute | CPU; no GPU, PyTorch, TensorFlow, or external optimization solver is required by the included Python scripts |

LaTeX is unnecessary for numerical reproduction. Compiling `TEST_submission/supplement.tex` is a separate task requiring the `sn-jnl` class and its LaTeX packages; the repository does not include a complete manuscript build environment.

### Setup

The following commands use a POSIX shell (Linux/macOS):

```bash
git clone https://github.com/ZhihaoZhao-UA/TEST_reproducibility_20261022.git
cd TEST_reproducibility_20261022
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install "numpy==2.3.5"
python -c "import sys, numpy; print(sys.version); print(numpy.__version__)"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell and set the thread variables with `$env:OPENBLAS_NUM_THREADS="1"` (and equivalently for the other two variables).

Set thread limits **before** starting Python. Some training scripts also start multiple Python processes: the n96 and predictability generators use four workers, and the nested experiment uses three. Full runs can be lengthy; no portable wall-time or memory guarantee is supplied. Floating-point details may vary across operating systems and BLAS implementations.

## Repository map

| Files | Purpose |
| --- | --- |
| `neural_path_average.py`, `neural_path_factor_train.py`, `neural_path_major_revision.py` | Shared training, simplex optimization, and experiment routines |
| `*_protocol.json` | Experiment settings and seeds; retain alongside scripts |
| `*_results.json`, other result/validation JSONs | Archived numerical outputs, summaries, and diagnostics |
| `neural_path_n96_predictions.npz` | Included historical prediction cache |
| `neural_path_nested_bank*_cache.npz` | Included nested-experiment caches for banks 0–7 |
| `neural_path_realdata_inputs/` | UCI Airfoil and Yacht input ZIPs, with source and license notes |
| `closure_20261016/` | Finite-support algebra checks and historical submission checks |
| `TEST_submission/supplement.tex` | Bundled supplementary source; also read by the historical strong-curvature validator |
| `MANIFEST.json` | SHA-256 checksums for 116 original archive files and recorded environment |
| `README.txt` | Original chronological reproduction notes, retained unchanged |

## Reproduction workflow

Run commands from the repository root, keeping the directory structure intact. **Use a disposable clone or copy:** generators and validators write JSON/NPZ outputs beside their source files and can overwrite archived results. Check original hashes before running anything. The new `README.md` and the later closure additions are outside the original manifest.

### 1. Verify the original archive files

This check uses only the Python standard library:

```bash
python - <<'PY'
from pathlib import Path
import hashlib
import json

root = Path.cwd()
files = json.loads((root / "MANIFEST.json").read_text())["files"]
bad = [name for name, expected in files.items()
       if not (root / name).is_file()
       or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected]
if bad:
    raise SystemExit("Missing or modified files: " + ", ".join(bad))
print(f"Verified {len(files)} original archive files.")
PY
```

After regeneration, checksum differences can reflect rewritten results or platform-dependent numerical details. Keep an untouched clone for comparison.

### 2. Start with algebra checks and stored-result validation

These commands avoid regenerating the full training studies:

```bash
# Current finite-support leverage and deterministic bridge checks (120 cases).
python closure_20261016/neural_path_finite_support_check.py

# Output-layer path representation, dimension/leverage checks, analytic example.
python neural_path_output_layer_check.py

# Validate the archived piecewise-path audit and calibration diagnostics.
python neural_path_final_validate.py
python neural_path_clarity_validate.py

# Validate the stored score audit and additional deterministic inequalities.
python neural_path_score_validate.py
```

Successful checks exit without assertion failures and write their report JSONs. The score validator checks 48 dictionaries and 4,608 aggregation fits; the output-layer script checks 30 random algebra cases. These are numerical checks, not proofs or independent predictive experiments. Compare generated reports with the archived `*_validation.json` and check JSONs, accounting for floating-point tolerances.

### 3. Regenerate the main numerical studies

Each block can be run separately after setup; protocols must remain beside the scripts.

```bash
# Finite-population residual-score audit.
python neural_path_score_audit.py
python neural_path_score_validate.py

# Fresh equal-budget experiment and fixed-dictionary sample-size experiment.
python neural_path_major_revision.py budget
python neural_path_major_revision.py fixed

# Public-data regression pilot; reads the included ZIPs directly.
python neural_path_realdata.py --data neural_path_realdata_inputs

# Piecewise-path audit and subsequent calibration diagnostics.
python neural_path_final_round.py
python neural_path_final_validate.py
python neural_path_clarity_diagnostics.py
python neural_path_clarity_validate.py

# Directional-certificate checks.
python neural_path_directional_check.py
```

| Study | Principal outputs |
| --- | --- |
| Score audit | `neural_path_score_results.json`, `neural_path_score_validation.json` |
| Equal budget (`budget`) | `neural_path_fresh_budget_results.json` |
| Fixed dictionary (`fixed`) | `neural_path_fixed_dictionary_results.json` |
| UCI pilot | `neural_path_realdata_results.json` |
| Piecewise path | `neural_path_final_results.json`, `neural_path_final_validation.json` |
| Calibration | `neural_path_clarity_diagnostics.json`, `neural_path_clarity_validation.json` |
| Directional checks | `neural_path_directional_check.json` |

### 4. Reproduce historical cache-based analyses

The n96 cache is included, so the following analyses can start from the archived predictions:

```bash
python neural_path_n96_analysis.py
python neural_path_n96_last_control.py
python neural_path_TEST_independent.py
python neural_path_local_geometry.py
python neural_path_distance_transfer.py
python neural_path_progressive_experiment.py
python neural_path_nested_decomposition.py
python neural_path_risk_target_audit.py
python neural_path_risk_target_crosscheck.py
python -c "from neural_path_risk_correction import run; run('n96')"
python -c "from neural_path_transport import run; run('n96')"
```

To rebuild caches from training, run `python neural_path_n96_confirm.py` first, followed by `python neural_path_nested.py` before the nested decomposition. The nested decomposition otherwise uses the eight included bank caches. The exploratory fold-ensemble pilot is available through `python neural_path_fold_ensemble_pilot.py`.

The larger `neural_path_predictability_predictions.npz` cache is **not included**. To reproduce analyses that need it:

```bash
python neural_path_predictability_confirm.py
python neural_path_predictability_analysis.py
python -c "from neural_path_risk_correction import run; run('predictability')"
python -c "from neural_path_transport import run; run('predictability')"
```

Running `neural_path_risk_correction.py` or `neural_path_transport.py` directly processes both datasets and therefore requires both prediction caches. Some legacy helper entry points, such as `neural_path_calibrated_analysis.py` and the default entry point of `neural_path_stabilize.py`, expect additional historical artifacts not included here; they are not standalone quick-start commands.

### 5. Optional historical mechanism studies

These studies are retained for provenance; see the scope note above before relating them to a particular submission:

```bash
python neural_path_training_mechanism.py
python neural_path_training_check.py
python neural_path_curvature_separation.py
python neural_path_strong_curvature.py
python neural_path_strong_trained.py
python neural_path_strong_check.py
```

The strong-curvature validator also reads the bundled supplementary table. These studies concern specific local-training or curvature mechanisms, not a general superiority claim or an independent CV benchmark.

## Data, interpretation, and reuse

The Airfoil Self-Noise and Yacht Hydrodynamics inputs are included as unchanged UCI ZIPs. Dataset sources, attribution, and CC BY 4.0 terms are recorded in [LICENSE_AND_SOURCES.txt](neural_path_realdata_inputs/LICENSE_AND_SOURCES.txt); input hashes are in `neural_path_realdata_results.json`. Numerical reproduction uses these local inputs without downloading datasets.

Aggregation weights do not use evaluation true means. Population oracles and some theoretical-bound diagnostics do use population information and should not be interpreted as confidence procedures computed solely from arbitrary observed labels. Overlapping folds, repeated aggregation samples, and overlapping real-data splits are not independent training-study replicates. Later calibration and mechanism analyses must be distinguished from fresh predictive confirmation.

No repository-wide software license is included. Dataset licenses apply to their respective inputs; making the repository public does not itself assign an open-source license to the code. When referencing results, identify the repository commit and the relevant protocol/result files. The archive does not provide a DOI.
