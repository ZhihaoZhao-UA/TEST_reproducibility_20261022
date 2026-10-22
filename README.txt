TEST reproducibility archive, 2026-10-05

Python 3.12 and NumPy 2.3.5 were used. Keep the extracted directory structure.
Run scripts from a disposable copy: analysis scripts may overwrite result JSONs.
For BLAS reproducibility and limited resource use, set:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
Some complete neural training studies are lengthy. Stored outputs and necessary
historical caches are included for inspection; results have not been replaced
with newly favorable runs. MANIFEST.json supplies SHA-256 hashes.

MANUSCRIPT / RECORD MAP AND RUN COMMANDS
Main Table 1: literature/theorem comparison; no numerical data.
Main Table 2, Figure 1: neural_path_local_geometry.json and
neural_path_distance_transfer.json; regenerate with the corresponding .py files.
Main Table 3 and Supplement S7 Tables S8-S9:
  python3 neural_path_score_audit.py
  python3 neural_path_score_validate.py
48 trained dictionaries, 4608 aggregation fits; known finite input population.
Main Table 4 (new equal-budget experiment):
  python3 neural_path_major_revision.py budget
  -> neural_path_fresh_budget_results.json
Main Table 5 (UCI regression):
  python3 neural_path_realdata.py --data neural_path_realdata_inputs
  -> neural_path_realdata_results.json
Input ZIP files, original URLs, checksums and license/source notes are included.

Supplement S1-S3 and S5 (historical/exploratory comparisons):
  neural_path_n96_results.json, neural_path_n96_analysis.json,
  neural_path_n96_last_control.json, neural_path_TEST_independent.json,
  neural_path_progressive_results.json, neural_path_nested_results.json,
  neural_path_nested_decomposition.json and neural_path_risk_correction_n96.json.
The core historical cache is neural_path_n96_predictions.npz. Regenerate it with
neural_path_n96_confirm.py, then run n96_analysis.py, n96_last_control.py,
TEST_independent.py, progressive_experiment.py, nested.py and
nested_decomposition.py (all names prefixed neural_path_). The nested cache
files are included. The earlier full-domain negative result is retained in
neural_path_predictability_results.json and its associated records; its larger
cache is omitted and can be recreated by neural_path_predictability_confirm.py.
To recompute only the n96 risk correction and inverse-size transport without
regenerating the larger historical cache, use:
  python3 -c "from neural_path_risk_correction import run; run('n96')"
  python3 -c "from neural_path_transport import run; run('n96')"
Direct execution of those two scripts processes both datasets and requires the
regenerated predictability cache.
Those experiments preceded the present theoretical refinement and must not be
relabelled as fresh confirmations. See protocols and Supplement S3.

Supplement S6 (piecewise path audit):
  python3 neural_path_final_round.py
  python3 neural_path_final_validate.py
  -> neural_path_final_results.json and validation file.
Supplement S7.4 (exploratory fold ensemble):
  python3 neural_path_fold_ensemble_pilot.py
Supplement S7.5 / Table S10 (fixed-dictionary sample sizes):
  python3 neural_path_major_revision.py fixed
  -> neural_path_fixed_dictionary_results.json
Supplement S8 (output-layer lemma and analytic example):
  python3 neural_path_output_layer_check.py
30 random algebra checks and direct evaluation of the closed-form constants;
these are not 30 new predictive experiments and are not the proof of the lemma.

INTERPRETATION
Weights never use evaluation true means. Numerical population oracles and risk
bounds do use them for diagnostics. The score audit uses exact finite-population
geometry; the S8 example specifies its population analytically. Neither is a
confidence procedure computed solely from arbitrary observed labels. Infinite
bounds, unsuccessful prespecified criteria and unfavorable CV comparisons are
retained. Overlapping folds, repeated aggregation samples and real-data random
splits do not constitute independent training-study replicates.

This archive is prepared for accompanying a submission. It has no public DOI
and has not been uploaded. Public UCI data retain their original licenses.


2026-10-07: Supplement S9 calibration diagnostics
Run python3 neural_path_clarity_diagnostics.py, then python3 neural_path_clarity_validate.py.
These post-outcome diagnostics replay the stored S6/S7 seeds and use saved public-data weights. They do not define a new estimator or provide independent predictive confirmation. Outputs cover the fixed-lambda geometry/envelope decomposition (48 dictionaries), analysis-lambda sensitivity (864 calculations), and weight concentration (Tables S11-S13). Validation checks input hashes, variance algebra, endpoint bounds, and response derivatives against finite differences.


2026-10-14: Corollary 9 and Supplement S10. Run python3 neural_path_directional_check.py. This verifies matrix allowances, the population bridge, designed nonlinear parameter segments, and the tangent-versus-joint leverage example. No new trained-path or predictive performance experiment is claimed.

S11 local training mechanism: run neural_path_training_mechanism.py, then neural_path_training_check.py (Python + NumPy). 1536 exploratory runs; all-parameter bias-free width-one tanh; fixed replicated training sites. Gradient control was added after initial outcomes. This is a pairwise mechanism check, not a CV benchmark or an initial-basin certificate.

S12: run neural_path_curvature_separation.py for the deterministic misspecification/curvature separation check. NumPy only. It checks population weight and risk expansions on an actual full-parameter path. It is NOT a finite-sample fitted-weight superiority experiment. The excess over gradient prediction mixing has a small coefficient.

S12 stronger mechanism: run neural_path_strong_curvature.py (stage 1), neural_path_strong_trained.py (stage 2 added after stage-1 inspection), neural_path_strong_check.py. The check also reads TEST_submission/supplement.tex included in this archive; the simulation scripts run independently. All 4608 outcomes retained. Population-designed coordinate map is shared by all comparators; not a data-only training method. New table reports paired gains and Monte Carlo SE, not universal superiority.
