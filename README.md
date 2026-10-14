# Corvita ICU: 24-hour in-hospital mortality risk

A small, complete ML system. It predicts in-hospital death from the first 24 hours of an
ICU stay, using three vitals (HR, RespRate, Temp) and three admission fields (age, sex,
ICU type). Data: PhysioNet/CinC Challenge 2012, set-a (4,000 adults).

**Research prototype. Not for clinical use. Not validated for neonates.**

The model is modest (test AUROC 0.74). Most of the work went into the parts around it:
no leakage, reproducible runs, a tested API, and an honest account of where the model fails.

---

## 1. Reproduce

Python 3.13 (models/v1 was trained on 3.13.1). Install from `requirements.lock`, which pins
every package including indirect ones; `requirements*.txt` list only the top-level packages.

Each `make` target below is one command; Windows users without `make` can run the command
shown in the `Makefile` (e.g. `make day1` = `python -m src.ingest` then `python -m src.run_day1`).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1                   # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.lock

python -m pytest -q                    # make test: 84 tests on synthetic data, ~1 min
python -m src.ingest                   # make data: download, check PhysioNet SHA256SUMS, manifest
python -m src.run_day1                 # make day1: parse, validate, 24 h features, splits
python -m src.train                    # make train: choose on val, report on test -> models/v1/
python -m src.monitor --make-baseline  # make baseline: monitoring reference for v1
```

If the download is blocked, put `set-a.tar.gz` and `Outcomes-a.txt` in `data/raw/` and run
`python -m src.ingest --no-download`.

**What a rebuild gives you.**
- `data/manifest.json` (checksums) and `data/splits/*_ids.csv` (split IDs) are committed.
  `make day1` on the same source files reproduces the same `data_version` and the same splits;
  `src.run_day1` fails if a group would differ in death rate by more than 2 points.
- `make train` needs a clean, committed git tree, so `metadata.json -> git_commit` always names
  the exact code. `--allow-dirty` exists for throwaway runs and records the commit as `-dirty`.
- Expected differences from the committed `models/v1`: `created_at`, `git_commit`, and
  `config_sha256` if you changed settings. With the same `requirements.lock` and Python 3.13
  the scores and cutoff match. Another BLAS, CPU or library version can move the last decimals.

---

## 2. System design

```
set-a/*.txt + Outcomes-a.txt
  ingest.py     SHA-256 of sources and every record file -> data/manifest.json, data_version
  parse.py      admission table + long observation table; -1 -> missing
  validate.py   IDs, labels, outcome linkage, times, ranges, units -> reports/data_quality.json
  features.py   keep 00:00-24:00 inclusive, then count / mean / last / missing per vital
  split.py      70/15/15 by patient, stratified, seed 42 -> data/splits/*.csv
  train.py      grid on train, choose on val, cutoff on val, then report on test -> models/v1/
  api/          FastAPI; imports features.py and validate.py, no second copy of the logic
  monitor.py    request log vs training baseline -> alerts
```

Key choices:

- **One feature module.** Training and the API call the same `build_features()`.
  Tests send every test-set patient's raw file through the API and compare with the scores
  saved at training time, for a linear and a tree model. Training reads `features.csv` with
  `float_precision="round_trip"`, and a test checks the loaded features are bit-identical to
  the in-memory ones: a changed last digit can move a tree split.
- **Everything learned is fit on train only.** Median imputation and scaling live inside the
  saved sklearn Pipeline. A test checks the learned medians equal the train medians.
- **Outcome fields never reach features.** `load_outcomes()` returns only the ID and the label.
  A test asserts no outcome column is a feature.
- **Settings in one file.** Cutoff, seed, split ratios, ranges, grids, `selection_metric`
  (pr_auc, auroc or brier) and sensitivity target are in `config.yaml`. Unknown values are
  rejected, so `metadata.json` never describes a setting that was not used.
- **Every model says what made it.** `metadata.json` holds `git_commit`, `data_version`,
  and SHA-256 of `features.csv`, `config.yaml` and each split file.
- **Model versions are immutable.** `train.py` refuses to overwrite `models/v1` unless told to.
  Files are written to a temp folder and renamed together.

---

## 3. Data checks (real data)

| Check | Result |
|---|---|
| Source files vs PhysioNet `SHA256SUMS.txt` | checked on download; result in `data/manifest.json` |
| Records / deaths | 4,000 / 554 (13.85%) |
| Outcome rows without a record / records without an outcome | 0 / 0 |
| Rows after 24:00 removed | 780,715 |
| Implausible values removed | HR 11, RespRate 62, Temp 130 |
| Temp in Fahrenheit | none found |
| Missing sex | 3 |
| Splits (n / death rate) | train 2,800 / 13.9%, val 600 / 13.8%, test 600 / 13.8% |

Temp had a minimum of -17.8 °C. That is 0 °F converted, so a "not measured" code, not a reading.
Values outside the ranges in `config.yaml` are treated as missing and counted, not clipped.
Full report: `reports/data_quality.json` (committed).

**Patients with no measurement in the 24 h window:**

| HR | RespRate | Temp | All three |
|---|---|---|---|
| 1.8% | 72.6% | 1.8% | 1.8% (71) |

**RespRate missingness is a charting rule, not chance.** Every patient on mechanical
ventilation in the first 24 h (2,405 of 2,405) has no RespRate. Among non-ventilated
patients, 31% have none. Death rate: 16.2% without RespRate, 7.7% with it.
So "RespRate missing" is close to "ventilated". `MechVent` was used to check this only;
it is not a model input.

---

## 4. Features

Per vital, from 00:00 to 24:00 inclusive (24:00 kept, 24:01 dropped):

| Feature | Meaning |
|---|---|
| `{v}_count` | number of valid readings (0 if none) |
| `{v}_mean` | mean (missing if none) |
| `{v}_last` | latest reading by time, not file order |
| `{v}_missing` | 1 if no reading |

Plus `Age`, one-hot `Gender` and `ICUType` with fixed categories and a missing indicator.
22 columns. Fixed category lists mean nothing is learned at this step.

---

## 5. Model and results

Two families, small grids, chosen by **validation PR-AUC** (deaths are the minority class):
logistic regression (`C`), and HistGradientBoosting (learning rate, depth, leaf size).

The cutoff is the highest score with **validation sensitivity ≥ 0.80**. 0.80 is a business
assumption: missing a death is treated as worse than a false alarm. It belongs to the
clinical team and is one line in `config.yaml`.

**Selected: logistic regression, C = 0.01, cutoff 0.131.**

Test set (n = 600, 83 deaths). Every choice was frozen on validation before these were computed:

| Metric | Value | 95% CI (bootstrap) |
|---|---|---|
| PR-AUC | 0.30 | 0.23–0.40 |
| AUROC | 0.74 | 0.68–0.80 |
| Brier | 0.110 | 0.091–0.128 |
| Sensitivity | 0.77 | 0.68–0.85 |
| Specificity | 0.58 | 0.54–0.62 |
| Alert rate / PPV | 47% / 0.23 | |

Reference points: a random score has PR-AUC 0.14. Predicting 13.85% for everyone gives Brier 0.119.

| Model | val PR-AUC | test PR-AUC | test AUROC |
|---|---|---|---|
| Logistic regression | **0.381** | 0.303 | 0.741 |
| HistGradientBoosting | 0.368 | 0.354 | 0.772 |

The tree model did better on test. It lost on validation, so it was not chosen.
The gap sits inside the test CI. Switching after seeing test would turn test into a
selection set. `C = 0.01` is the edge of the grid; widening it is listed as future work.

### What the score is made of

Ablations retrain the same family with the same grid on fewer columns:

| Inputs | Features | val PR-AUC | test PR-AUC | test AUROC |
|---|---|---|---|---|
| All | 22 | 0.381 | 0.303 | 0.741 |
| No RespRate columns | 18 | 0.302 | 0.288 | 0.689 |
| No count columns | 19 | 0.363 | 0.309 | 0.754 |
| Age, sex, ICU type only | 10 | 0.228 | 0.239 | 0.700 |

- Admission fields alone reach AUROC 0.70. Vitals add about 0.04 on test.
- Most of that gain is whether RespRate was recorded, i.e. ventilation.
  HR and Temp add nothing measurable on test (the CI is about ±0.06).
- Count columns had large coefficients but removing them costs little. The coefficients
  shared signal with correlated columns. Ablation, not coefficient size, is the evidence here.

### Operating points

Cutoff chosen on validation for each target, applied unchanged to test:

| Target sensitivity | Cutoff | Test sensitivity | Test alert rate | Test PPV |
|---|---|---|---|---|
| 0.6 | 0.172 | 0.57 | 27% | 0.29 |
| 0.7 | 0.150 | 0.69 | 37% | 0.26 |
| **0.8** | **0.131** | **0.77** | **47%** | **0.23** |
| 0.9 | 0.110 | 0.88 | 61% | 0.20 |

PPV stays between 0.20 and 0.29. This model can rank patients for a broad screen.
It cannot produce a short, precise list.

### Missing vitals

Real missingness, test set, same cutoff:

| Group | n | Deaths | AUROC | Sensitivity | Specificity |
|---|---|---|---|---|---|
| RespRate recorded (none ventilated) | 179 | 13 | 0.77 | 0.62 | 0.81 |
| RespRate missing (mostly ventilated) | 421 | 70 | 0.70 | 0.80 | 0.47 |
| All vitals missing | 11 | 0 | n/a | n/a | 0.45 |

Everyone with RespRate also has HR and Temp, so "RespRate recorded" equals "all vitals present".
One cutoff behaves very differently in the two groups. The mostly ventilated group gets most alerts.
The RespRate-recorded group gets few, and 5 of its 13 deaths are missed.

Masking one vital for every test patient (as if a feed went down):

| Masked | Alert rate | Sensitivity | AUROC |
|---|---|---|---|
| none | 47% | 0.77 | 0.74 |
| HR | **23%** | **0.51** | 0.73 |
| RespRate | 63% | 0.82 | 0.69 |
| Temp | 67% | 0.88 | 0.73 |
| all | 51% | 0.80 | 0.71 |

**If the HR feed drops, alerts halve and no error is raised.** The shift comes mostly from
`HR_count` falling to 0, not from the missing flag. Losing RespRate makes everyone look
ventilated and raises alerts. Masking one vital also creates input combinations that almost
never occur in training.

The same masking is run on the validation set (`metrics.json -> masking_val`). The API's
missing-HR warning and the monitoring rules rest on the validation numbers, so no product
decision depends on test results. The test table above is for reporting only.

### Calibration and subgroups

| Predicted risk decile | Lowest | 4th | 9th | Highest |
|---|---|---|---|---|
| Mean predicted | 5.2% | 10.4% | 21.4% | 29.0% |
| Observed | 1.7% | 5.0% | 28.3% | 36.7% |

Scores are compressed: low risk is over-predicted, high risk under-predicted. Strong
regularization does this, and PR-AUC selection ignores calibration.
Do not show raw probabilities to clinicians without recalibration.

| ICU type | n | Deaths | AUROC | Sensitivity |
|---|---|---|---|---|
| Coronary | 92 | 15 | 0.86 | 0.87 |
| Cardiac surgery recovery | 114 | 7 | **0.52** | **0.29** |
| Medical | 238 | 38 | 0.74 | 0.87 |
| Surgical | 156 | 23 | 0.72 | 0.70 |

The model does not work in cardiac surgery recovery. 95% of those patients are ventilated,
so the ventilation signal cannot separate them. Women: sensitivity 0.83 (30 deaths);
men: 0.74 (53 deaths). That gap is within noise.

All numbers: `models/v1/metrics.json`. Test predictions: `models/v1/test_predictions.csv`.

---

## 6. API

`POST /predict` takes raw observations and builds features with the training code.

```json
{"record_id": "demo-1", "age": 67, "gender": 1, "icu_type": 3,
 "observations": [{"time": "01:30", "parameter": "HR", "value": 92}]}
```

Returns `probability`, `high_risk`, `threshold`, `model_version`, `data_version`,
`data_quality_flag`, `missing_vitals`, `low_count_vitals`, `warnings` and `request_id`.
Also `GET /health` and `GET /model-info` (which adds `git_commit`).
`model_version` + `data_version` in every response, and `git_commit` in `/model-info`,
identify the data, code and model behind a prediction.

**Input contract: send every charted reading from the first 24 h, not summaries.** The number
of readings per vital is a model input, so sending fewer readings than were charted changes
the score (same patient, HR and Temp only: 24 HR + 6 Temp readings p = 0.18; 6 + 2 p = 0.16; 1 + 1 p = 0.16).

| Input | Response |
|---|---|
| Wrong type (strict: `"60"` for age, `true` for gender, `3.0` for ICU type), unknown field, unknown vital, bad time, NaN/Infinity, age outside 15–100, ICU type not 1–4 | 422; error body lists fields and reasons, never the submitted values |
| Observations after 24:00 | ignored, warning |
| Implausible value (e.g. HR 0) | ignored, warning |
| A vital with far fewer readings than typical (< 25% of the training median: HR 29, RespRate 25, Temp 7) | scored, `low_count`, warning |
| A vital not measured | scored, `missing_vitals` |
| HR not measured | scored, extra warning that a low score is unreliable |
| No vitals at all | scored from age, sex, ICU type; `insufficient_data`; **`high_risk: null`** |

Quality flag, most severe first: `insufficient_data` > `missing_vitals` > `low_count` > `ok`.

Why `high_risk` is null without vitals: on age, sex and ICU type alone, 39% of combinations
crossed the cutoff (e.g. age 60, medical ICU), and in the test set 6 of the 11 patients with no
vitals were flagged although all survived. An alert with no physiological basis is worse than
no alert. The probability is still returned for reference.

Rule: a request built wrong is rejected. Clinical data that is incomplete is scored and
flagged, because training data looked the same way.

The API refuses to start if the model's cutoff or feature list differs from the code.

**Examples** (`examples/`), each with the real response from `models/v1`
(regenerate with `python -m scripts.make_examples` after retraining):

| Request | Status | Shows |
|---|---|---|
| `valid.json` | 200 | 12 HR, 8 RespRate, 6 Temp readings; one reading after 24:00 ignored |
| `missing_vitals.json` | 200 | no RespRate (typical ventilated patient): `missing_vitals` |
| `invalid.json` | 422 | age as text, gender as `true`, ICU type 7, bad time, unknown vital |

**Logs.** One JSON line per request to stdout (and to `LOG_PATH` if set): request ID, path,
status, latency, model version, and for predictions the quality flag, missing vitals,
per-vital counts and the score. Never logged: record ID, age, sex, vital values.
Validation errors log field names only. A test checks this.
If the log file cannot be written (read-only mount, wrong owner), the API warns once on
stderr and keeps logging to stdout; a logging failure never fails a request.

---

## 7. Tests and CI

84 tests, all on synthetic data in PhysioNet format (`tests/conftest.py`). CI never sees patient data.

- Cutoff: 24:00 kept, 24:01 dropped, in features and in the API
- Splits: disjoint, complete, reproducible, balanced; overlap raises `DataError` (not `assert`)
- Outcome linkage: an outcome row without a record is reported
- No outcome fields in features; imputer fit on train only
- Cutoff chosen on validation, stored unrounded, considers every ROC point
- `selection_metric` from config is the one used; unknown settings are rejected
- Training refuses uncommitted code; metadata carries commit, config and split hashes
- Saved model reloads and reproduces saved scores; features survive the CSV round trip bit for bit
- API: every test-set patient scored through the API matches the saved offline score
  (logistic regression and gradient boosting); valid, invalid (422, incl. strict types and NaN),
  missing vitals, no vitals (`high_risk: null`), low counts, late and implausible values
- Logs carry the right fields and no patient data; an unwritable log path does not fail requests
- Checksums catch an edited source file; downloads are atomic; PhysioNet hash mismatch is an error
- Monitor: alerts on HR outage, charting-frequency change, score shift, 422 spikes; reads
  container output with non-JSON lines; empty log -> `insufficient_traffic`; missing log -> exit 2

`.github/workflows/ci.yml`, on every push and pull request:
1. **test**: Python 3.13.1, install `requirements.lock`, `make test`.
2. **terraform**: `fmt -check`, `init -backend=false`, `validate`.
3. **docker**: build the image, run it with a logs volume, send the three example requests
   (expect 200, 200 with `missing_vitals`, 422), then run the monitor on the request log.

`.github/workflows/deploy.yml` is a separate, manual-only release workflow (section 8). Both
workflow files pass `actionlint`.

---

## 8. Monitoring, retraining, release, rollback

`python -m src.monitor` compares the last 500 predictions with the v1 baseline.
Exit code 0 = ok or not enough traffic yet, 1 = alert, 2 = log or baseline file missing.
Run it on a schedule (e.g. hourly) and page on exit code 1.

Demo with Docker (request logs land in `./logs` on the host):

```powershell
docker build -t corvita-icu:v1 .
mkdir logs
docker run --rm -p 8000:8000 -e LOG_PATH=/app/logs/requests.jsonl -v "${PWD}/logs:/app/logs" corvita-icu:v1
# second window
curl.exe -s -X POST localhost:8000/predict -H "content-type: application/json" -d "@examples/valid.json"
curl.exe -s -X POST localhost:8000/predict -H "content-type: application/json" -d "@examples/missing_vitals.json"
curl.exe -s -X POST localhost:8000/predict -H "content-type: application/json" -d "@examples/invalid.json"
python -m src.monitor --log logs/requests.jsonl
```

On a Linux host, run `chmod 777 logs` first (or add `--user "$(id -u)"`): the container user
cannot write to a folder owned by your user. If it can't, the API still answers and logs to stdout.
`docker logs <container> > requests.log` also works as monitor input: the container runs uvicorn
with `--no-access-log`, and any other non-JSON lines are skipped and counted.

| Check | Alert when | Why |
|---|---|---|
| Missing rate per vital | moves > 10 pts from train | HR outage halves alerts silently |
| Median count per vital | < 0.5x or > 2x train | counts track charting frequency |
| Mean score, alert rate | move > 0.05 / > 0.15 from val | case mix or input drift |
| 5xx, 422 share | > 1% / > 10% | bugs, or a client changed its format |

No drift check runs under 50 predictions. Thresholds can be set under `monitoring:` in `config.yaml`.

Not built, by design or for time:

- **Drift on values.** Add PSI on `HR_mean`, `Temp_mean` and the score against the train distribution.
- **Retraining is manual.** A person decides. Steps: new data -> new manifest -> new splits ->
  `model.version: v2` -> `python -m src.train` -> compare v2 with v1 on the same new test set.
- **Release** (`.github/workflows/deploy.yml`, started by hand, never on push):
  `deploy` builds the image tagged with the model version and creates a Cloud Run revision
  with 0% traffic, reachable at its own tag URL for checks. `canary` sends a share of traffic
  (default 10%); watch the monitor; `promote` sends 100%.
- **Rollback.** Image tag = model version, and registry tags are immutable, so `v1` always
  means the same image. Rollback = the `rollback` action with `v1`: traffic goes back to that
  revision. No rebuild. Terraform ignores image and traffic on the service, so a later
  `terraform apply` never undoes a release.
- **Outcome monitoring.** Deaths are known only at discharge. Join outcomes to logged
  request IDs later to track real PPV and calibration.

---

## 9. Infrastructure (Terraform, GCP)

`infra/` defines a private, versioned bucket; an image registry with immutable tags;
a Cloud Run service; two service accounts; IAM; Workload Identity Federation for CI
(`wif.tf`); log retention; and a budget alert.
`terraform init -backend=false` and `terraform validate` output: `infra/validate_output.txt`
(Terraform 1.16.5, hashicorp/google 6.50.0; versions pinned in `versions.tf` and
`.terraform.lock.hcl`). CI repeats both on every push, plus `terraform fmt -check`.
**Nothing was deployed.**

- **Who can do what.** ML engineers read and write the bucket. CI pushes images and deploys
  revisions as the runtime account only. The runtime account has no project roles: the model
  is in the image and logs go to stdout. Callers need `run.invoker`. The config rejects `allUsers`.
- **Secrets.** None today, and CI needs none: it signs in with Workload Identity Federation
  (`infra/wif.tf`). GitHub issues a short-lived OIDC token; Google accepts it only from
  `zzhu247/corvita-icu` on `main` and swaps it for a 1-hour token for the CI service account.
  No key file exists to leak or rotate; forks, other branches and pull requests cannot sign in.
  Any future runtime secret goes in Secret Manager, read by the runtime account. Never in git or tfvars.
- **State.** A separate, versioned GCS bucket created once by hand (backend block in
  `versions.tf`). State is git-ignored.
- **Retention.** Raw files deleted after 90 days. Old object versions after 30 days.
  Request logs kept 30 days. Models are kept.
- **Cost.** Scales to zero, max 2 instances, 512 MiB. Budget alerts at 50/90/100% when a
  billing account is set. Estimate at prototype traffic (≈10,000 predictions a month):

  | Item | Usage | Monthly |
  |---|---|---|
  | Cloud Run (Toronto, tier 2) | ~1,000 vCPU-s and ~500 GiB-s; free tier is 180,000 vCPU-s, 360,000 GiB-s, 2M requests | $0 |
  | Artifact Registry | ≤10 images × ~0.3 GB; 0.5 GB free, then ~$0.10/GB | ~$0.25 |
  | Cloud Storage | < 1 GB of data and models | < $0.05 |
  | Cloud Logging | well under the 50 GiB/project free allowance | $0 |
  | **Total** | | **≈ $0.30** |

  Free tiers are per billing account. Prices change; check the GCP pricing calculator before
  deploying. The main cost risks are a minimum instance (`min_instance_count > 0`, ~$15–40
  a month for one always-on instance) and many image versions; the config avoids the first
  and the registry cleanup policy keeps 10 versions.
- **Teardown.** `terraform destroy`. The bucket has `force_destroy = false`, so data has to
  be removed on purpose first.
- **Recovery.** Bucket versioning restores overwritten files. Immutable tags plus Cloud Run
  revisions restore the service.
- **Untested** (no cloud account; the brief says not to pay for one):
  - `terraform plan` and `apply`: nothing was created. `validate` passes locally and in CI.
  - IAM in a real project: the roles are least-privilege on paper, but no call was made to prove
    a denied action is really denied.
  - Workload Identity Federation and `deploy.yml`: written and checked (`terraform validate`,
    `actionlint`), never run. The workflow is skipped until the GCP repo variables exist.
  - Cold-start latency and load on Cloud Run.
  - The region (Toronto) is chosen for data residency; confirm against the actual data agreement.

---

## 10. Limitations

- **Selection bias.** Set-a only includes adults who stayed at least 48 h in the ICU. At 24 h
  you do not know who will reach 48 h. Early deaths and early transfers are missing.
  Real-world performance on those patients is unknown.
- **Single source, around 2012, adults only.** Not for neonates (NOA). Not for other hospitals
  without local validation.
- **The model learned charting habits.** RespRate missingness stands in for ventilation.
  Another site that charts RespRate on ventilated patients would break this silently.
- **Weak probabilities.** Brier is 8% better than predicting the base rate, and scores are
  compressed. Ranking is usable; raw probabilities are not.
- **Fails in cardiac surgery recovery** (AUROC 0.52, n = 114, 7 deaths).
- **Small test set.** 83 deaths. All metrics carry wide intervals.
- **Ranges and the 0.80 target are my assumptions.** Both need clinical sign-off.

## 11. Before real use

1. Clinical sign-off on value ranges, the sensitivity target and the alert budget.
2. Recalibrate (Platt or isotonic) on a held-out set; select on log-loss or Brier too.
3. Replace "RespRate missing" with an explicit ventilation input, or drop count features (v2).
4. Validate on data from the target site, including patients who leave the ICU before 48 h.
5. Auth through the hospital gateway, HTTPS, rate limits, audit logging, a privacy review.
6. Outcome feedback loop and value-drift checks.
7. Decide whether the model should be used at all in cardiac surgery recovery.

## 12. AI use

AI tooling was used as a coding assistant to draft boilerplate, tests and early doc text,
but the final project decisions were validated on the real data and checked by running the
project's own test and monitoring workflows. The modeling choices, the ventilation-based
missingness explanation, the prediction API and the monitoring rules were all reviewed and
verified against the repository outputs before being kept.

Which parts used AI (Claude, Anthropic):

| Part | AI role |
|---|---|
| `src/`, `api/`, `scripts/` | First drafts of the code; I ran every step on the real data and checked the outputs |
| `tests/` | Drafted test cases and the synthetic PhysioNet-format data |
| `infra/`, `Dockerfile`, workflows | Drafted Terraform, Dockerfile, CI and release workflow |
| Docs (README, notes, model card) | First drafts of the text; numbers come from my runs |
| Code review | An AI review (`REVIEW.md`) found 16 bugs; fixes in `REVIEW_FIXES.md`, each with a test |

Done by me: running the pipeline on the real data, reading the results, checking the
ventilation explanation with `MechVent`, keeping logistic regression after the tree model
looked better on test, and correcting an early reading of the coefficients after the ablations.

## 13. Project status and follow-up

Time used (approximate, within the 24 h limit):

| Day | Work | Hours |
|---|---|---|
| 1 | Data, checks, cutoff, features, splits, tests | 2.5 |
| 2 | Models, selection, cutoff, evaluation, missing-vital analysis | 1.5 |
| 3 | API, Docker, logs, monitoring, CI, Terraform, docs | 3 |
| After review | Fixes from `REVIEW.md`, Workload Identity, release workflow | 2 |
| **Total** | | **9** |

This repository is organized around a three-day project plan with the required data pipeline,
model training, API, tests, monitoring and Terraform scaffolding. The implementation is
complete for the local ML + MLOps workflow described in the original PDF brief, and it is
intended as a prototype rather than a production clinical deployment.

Recommended next steps before real-world use:

- cloud deployment and IAM checks in a real project account; first real run of the
  Workload Identity sign-in and `deploy.yml` (both written and validated, never run)
- value-drift checks such as PSI on the score and key features
- recalibration on a held-out clinical set
- broader model-search or re-optimisation for the `C` grid and feature set
- authentication, audit logging and privacy review for hospital deployment

See `MODEL_CARD.md` for the one-page summary, `TASK.md` for the original brief, and
`REVIEW_FIXES.md` for the post-review fixes and bug corrections.