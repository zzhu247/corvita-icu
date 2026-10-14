# Day 3 notes: API, Docker, logs, monitoring, CI

## How to run (Windows PowerShell)

```powershell
python -m pip install -r requirements.lock
python -m pytest -q

# monitoring baseline, once per model version
python -m src.monitor --make-baseline

# start the API, request logs to logs/requests.jsonl
$env:LOG_PATH = "logs/requests.jsonl"
python -m uvicorn api.main:create_app --factory --port 8000 --no-access-log
```

In a second window:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict `
  -ContentType "application/json" -InFile examples/valid.json
# interactive docs: http://localhost:8000/docs

python -m src.monitor --log logs/requests.jsonl
```

Docker (needs Docker Desktop):

```powershell
docker build -t corvita-icu:v1 .
mkdir logs
docker run --rm -p 8000:8000 -e LOG_PATH=/app/logs/requests.jsonl -v "${PWD}/logs:/app/logs" corvita-icu:v1
```

## Files

| File | Role |
|---|---|
| `api/schemas.py` | Request and response shapes; decides what returns 422 |
| `api/service.py` | Loads the model, startup checks, builds features with `features.py`, quality flags |
| `api/main.py` | FastAPI routes, request-log middleware, error handling |
| `src/monitor.py` | Baseline, then checks the log for missing rates, counts, scores, error rates |
| `src/validate.py` | `out_of_range()`, shared by training and the API |
| `examples/` | Valid, missing-vitals and invalid requests, each with its real response |
| `scripts/make_examples.py` | Regenerates the example responses after a retrain |
| `Dockerfile` | Pinned base image, installs `requirements.lock`, non-root user, writable log folder, healthcheck |
| `.github/workflows/ci.yml` | Every push: tests; Terraform fmt/init/validate; image build plus example requests and monitor |
| `.github/workflows/deploy.yml` | Manual only: release a model version to Cloud Run (deploy, canary, promote, rollback) |
| `infra/wif.tf` | Workload Identity Federation, so GitHub Actions signs in to GCP without a key file |

## Design choices

**1. What returns 422, and what returns 200 with a warning?**
- **422**: the client built the request wrong. Wrong types (strict: no `"60"` for age, no `true`
  for gender, no `3.0` for ICU type), unknown fields, unknown vitals, bad time strings,
  NaN/Infinity, age outside the training range (15–100), ICU type not 1–4.
  The error body lists fields and reasons, never the submitted values. That keeps patient data
  out of error responses, and it avoids a crash: echoing a `NaN` back made the API return 500.
- **200 with a warning**: clinical data that is incomplete or noisy. Readings after 24:00
  (ignored), implausible values (ignored), a vital not measured, very few readings.
- Reason: the second kind also happened in training, and the model was built for it. The first
  kind means the integration is broken and should fail loudly.

**2. How do training and serving stay identical?**
The API calls `build_features()` from `features.py` and `out_of_range()` from `validate.py`.
There is no second copy of the logic. A test sends every test-set patient's raw file through the
API and compares with the score saved at training time, for both a linear and a tree model.

**3. The 24:00 cutoff is tested in the API too**
A 24:00 reading is kept; 24:01 and 30:00 are ignored with a warning. The test checks the score
does not change when late readings are added.

**4. Startup checks (fail fast)**
The API refuses to start if the model's cutoff differs from `config.yaml`, or its feature
columns differ from what `features.py` builds. Better no service than a service that quietly
computes the wrong thing. `test_refuses_model_with_different_cutoff` checks it.

**5. Data-quality flags and alerts**
Most severe first:
- `insufficient_data`: no vitals at all. A probability is returned, but **`high_risk` is null**.
  On age, sex and ICU type alone, 39% of combinations crossed the cutoff (56% in the medical
  ICU), and in the test set 6 of the 11 patients with no vitals were flagged although all
  survived. An alert with no physiological basis is worse than no alert.
- `missing_vitals`: at least one vital never measured. Missing HR adds a warning: on the
  validation set, removing HR lowers scores and alerts (`masking_val`), so a low score is unreliable.
- `low_count`: every vital present, but one has fewer than 25% of the training median readings
  (HR 29, RespRate 25, Temp 7). The reading count is a model input, so clients must send every
  charted reading, not summaries. Same patient: 24 HR + 6 Temp readings p = 0.18; 1 + 1 p = 0.16.
- `ok`.

**6. What is logged, and what is not**
- One JSON line per request: request ID, path, status, latency, model version. Predictions add
  the quality flag, missing and low-count vitals, per-vital counts, the score and `high_risk`.
- **Never logged**: record ID, age, sex, vital values. Validation errors log field names only.
  A test checks this.
- The request ID is also in the `x-request-id` response header.
- Logs go to stdout, which the cloud platform collects. With `LOG_PATH` set, a copy goes to a file
  for the monitor. If that file cannot be written, the API warns once on stderr and keeps
  answering; a logging failure never fails a request.
- Uvicorn runs with `--no-access-log`. The app already logs every request, and uvicorn's
  plain-text lines would mix into the JSON stream.

**7. Why does monitoring watch these numbers?**
Each comes from a Day 1 or Day 2 finding:
- **Missing rate per vital**: a missing HR feed lowers alerts; RespRate missingness means
  ventilation. Neither raises an error.
- **Median reading count per vital**: counts track how often staff chart. A new site, or
  automatic monitor charting, moves them a lot.
- **Mean score and alert rate**: compared with validation.
- **5xx and 422 share**: a jump in 422s usually means a client changed its format.

Exit codes: 0 = ok or not enough traffic, 1 = alert, 2 = log or baseline missing.
No drift check runs under 50 predictions. Thresholds can be overridden under `monitoring:` in
`config.yaml`. The monitor reads the log file or saved `docker logs` output; non-JSON lines are
skipped and counted. It only alerts: retraining stays a human decision, as the brief requires.

**8. Why does CI use no real data?**
All tests use the synthetic PhysioNet-format data from `tests/conftest.py`. CI neither needs nor
should touch patient data. The Docker job needs `models/v1`, so the model is committed; it is a
few KB. In production the image build would pull the model from the bucket by version.

**9. Docker image**
- Base image pinned to `python:3.13.1-slim`, the Python that trained v1.
- Installs `requirements.lock`, so every package version matches CI and training.
- Runs as a non-root user that owns `/app/logs`.
- Copies only `model.joblib` and `metadata.json`.
- Image tag = model version. Rollback = point the service back at the old tag.

**10. Releasing to the cloud (written, not run)**
`.github/workflows/deploy.yml` runs only when started by hand:
- `deploy`: build the image tagged with the model version, push it, create a Cloud Run revision
  with 0% traffic. The revision gets its own tag URL, so `/health` and `/model-info` can be
  checked before any patient request reaches it.
- `canary`: send a share of traffic to it (default 10%) and watch the monitor.
- `promote`: 100% of traffic.
- `rollback`: 100% back to an earlier tag, e.g. `v1`. No rebuild: registry tags are immutable,
  so `v1` always means the same image.

Sign-in uses Workload Identity Federation (`infra/wif.tf`). GitHub issues a short-lived OIDC
token; Google accepts it only from `zzhu247/corvita-icu` on `main` and exchanges it for a
1-hour token for the CI service account. No service account key exists to leak or rotate;
forks, other branches and pull requests cannot sign in.

Terraform creates the Cloud Run service but ignores its image and traffic, so a later
`terraform apply` never undoes a release. The workflow is skipped until the GCP repo
variables are set.

## Limitations

- The Docker image is built and tested in CI on every push: build, start, three example
  requests, then the monitor on the request log. Locally the API was run with the same start
  command as the container.
- Nothing was deployed to the cloud (the brief says not to pay for an account):
  - no `terraform plan` or `apply`; `validate` passes locally and in CI
  - IAM roles are least-privilege on paper, but no real call proved a denied action is denied
  - Workload Identity Federation and `deploy.yml` are written and linted, never run
  - cold-start latency and load on Cloud Run are unmeasured
- No authentication, rate limits or HTTPS in the app. In deployment the gateway or Cloud Run
  IAM handles these.
- Monitoring is a batch script, not real time. In production it runs on a schedule and pages on exit code 1.
- The drift thresholds (e.g. a missing-rate shift above 10 points) are judgment calls, not calibrated.