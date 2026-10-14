# Day 2 notes: training, selection, evaluation, saved artifacts

## How to run

```powershell
python -m pytest -q
python -m src.train          # writes models/v1/
```

`src.train` stops if `models/v1` already exists, so an old version is never overwritten by accident:
- to replace it: add `--overwrite`
- to keep it: change `model.version` in `config.yaml` to `v2`

It also stops if the git tree has uncommitted changes, so `metadata.json` always names the exact
code that produced the model. `--allow-dirty` exists for throwaway development runs only.

## Order of operations (the order is the leakage protection)

1. Each model family runs a small grid: every setting is fit on **train** and scored on **val**
   with `selection_metric` (PR-AUC by default).
2. The best setting per family, then the best family, are chosen on that **val** score.
3. The cutoff is chosen on **val**: the highest cutoff with sensitivity ≥ 0.80.
4. Every decision is now frozen. From here **test** is used for reporting only: both families'
   scores, CIs, calibration, missing-vital subgroups, masking, ablations, operating points and
   subgroups. Test is scored many times, but no test number feeds back into a model, a setting
   or the cutoff.
5. Masking is also run on **val** (`masking_val`). The API's missing-vital warnings and the
   monitoring rules rely on these validation numbers, not on test.

## Design choices

**1. Why select on PR-AUC?**
Deaths are about 14%, a minority class. AUROC is pulled up by the many true negatives.
PR-AUC measures how many flagged patients really are high risk. Baselines to quote alongside:
a random score gives PR-AUC ≈ 0.14 and AUROC 0.5.
`selection_metric` in `config.yaml` can be `pr_auc`, `auroc` or `brier` (lower is better, so it
is negated). Any other value is rejected, so the metadata never names a metric that wasn't used.

**2. Why these two models?**
- **Logistic regression**: interpretable, usually well calibrated, the baseline. Grid over `C`.
- **HistGradientBoosting**: captures non-linear effects and interactions, handles NaN natively,
  needs no extra package. Grid over learning rate, depth and leaf size: 12 settings.
- The grids are small on purpose. Validation has about 83 deaths; a large grid starts to fit
  the validation set.

**3. Why is preprocessing fit on train only?**
Median imputation and scaling for logistic regression are inside the Pipeline, so `fit` sees
train only. `test_imputer_learns_from_train_only` checks the learned medians equal the train
medians, not the medians of all data. The tree model needs no imputation.

**4. How is the cutoff chosen?**
On validation: the highest cutoff whose sensitivity is still ≥ 0.80. A higher cutoff means
fewer alerts, so among the cutoffs that catch enough deaths this one raises the fewest false
alarms. Every ROC point is considered (`drop_intermediate=False`), and the value is stored
unrounded so the API uses exactly the evaluated cutoff.
0.80 is a **business assumption**. The clinical team should own it; it is one line in `config.yaml`.
Test sensitivity can land below 0.80 because the cutoff was set on validation. That is expected,
and it is the honest number.

**5. Why not refit the final model on train + val?**
The cutoff chosen on validation would no longer belong to the refit model, and with this little
data the gain is small. Stated in the README as a trade-off.

**6. Why does the RespRate ablation drop all four columns?**
Day 1 showed every ventilated patient has no RespRate, so "RespRate missing" ≈ "ventilated".
Dropping only `RespRate_missing` changes nothing: `RespRate_count == 0` and the NaN mean and last
carry the same information. So all `RespRate_*` columns go, and the PR-AUC drop shows how much the
model relies on the charting rule rather than on breathing rate itself.
Two more ablations: `no_counts` (all `*_count` columns) and `admission_only` (age, sex, ICU type).

**7. Two kinds of missing-vital analysis**
- **Subgroups** (`missing_subgroups_test`): how the model does on patients who really had a vital missing.
- **Masking** (`masking_val`, `masking_test`): one vital set to missing for everyone, as if a feed
  went down. Masking RespRate makes everyone look ventilated, so predicted risk and alerts rise.
  The model cannot tell "not measured" from "feed broken".

**8. Calibration**
`calibration_test` groups test patients into 10 bins by predicted risk and compares predicted
with observed death rate. No recalibration was done: validation already chose the model and the
cutoff, and calibrating on it too would put all three decisions on the same 600 patients.
Recalibration is listed as a next step.

**9. What is saved, and how**
`models/v1/`:
- `model.joblib`: the whole Pipeline, preprocessing included
- `metadata.json`: `git_commit`, `data_version`, SHA-256 of `features.csv`, `config.yaml` and each
  split file, seed, hyperparameters, cutoff and rule, feature columns, train missing rates and
  median reading counts (baselines for the API's `low_count` flag and for monitoring), Python and
  library versions
- `metrics.json`: every result
- `test_predictions.csv`: per-patient test scores, for audit

Files are written to a temp folder and renamed together, so a half-written version never exists.
`test_artifacts_saved_and_loadable` checks that the reloaded model reproduces the saved scores.
`features.csv` is read back with `float_precision="round_trip"`; a test checks the features
survive the CSV round trip bit for bit. Without it, 1% of `HR_mean` values changed in the last
digit, which can move a tree split and make API and offline scores disagree.

**10. Logistic regression coefficients: read with care**
`logreg_coefficients` in `metrics.json` are on scaled inputs, sorted by size. The count features
had some of the largest coefficients, but the `no_counts` ablation barely lost anything
(val PR-AUC 0.381 -> 0.363). Correlated features share signal, so coefficient size is not
importance. The ablations are the evidence; the coefficients are not.

## Results on the real data (test set: n = 600, 83 deaths)

Selected: logistic regression, `C = 0.01`, cutoff 0.131.

| Model | val PR-AUC | test PR-AUC | test AUROC | test Brier |
|---|---|---|---|---|
| Logistic regression | **0.381** | 0.303 | 0.741 | 0.110 |
| HistGradientBoosting | 0.368 | 0.354 | 0.772 | 0.105 |

- 95% CI for the selected model: PR-AUC 0.23–0.40, AUROC 0.68–0.80. Sensitivity 0.77, specificity 0.58.
- The tree model did better on test but lost on validation, so it was not chosen. Switching after
  seeing test would make test a selection set. The gap is inside the test CI.
- `C = 0.01` is at the edge of the grid. Widening it is future work, decided on validation.
- Ablations (test AUROC): all features 0.741, admission fields only 0.700, no RespRate 0.689,
  no counts 0.754. Most of what vitals add comes from whether RespRate was recorded.
- Operating points (cutoff from val, applied to test): sensitivity target 0.6 -> 27% alerted,
  PPV 0.29; 0.8 -> 47%, PPV 0.23; 0.9 -> 61%, PPV 0.20.
- Masking on test: HR missing for everyone halves alerts (47% -> 23%); RespRate missing raises
  them to 63%. Validation numbers are in `masking_val`.
- Calibration: scores are compressed. Lowest decile 5% predicted vs 2% observed; highest 29% vs 37%.
- Cardiac surgery recovery: AUROC 0.52, 2 of 7 deaths caught.

## Limitations from Day 2

- Small test set: every metric needs its CI.
- The model uses ventilation status indirectly and depends on the source hospital's charting.
- The 0.80 sensitivity target is an assumption, not a clinical decision.
- No recalibration (see point 8).
- One seed, one split. Repeating with other seeds would show stability; it must not be used to
  pick the best-looking result.