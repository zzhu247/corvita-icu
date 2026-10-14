# Changes after code review

An external review (`REVIEW.md`, 16 bugs, 5 nits) was run on the GitHub copy of this repo.
Each item below lists the fix and the test that now covers it.

Root cause of the "missing" items: `.gitignore` had the `.dockerignore` rules pasted into it,
so `tests/`, `infra/`, every `*.md`, the split IDs and the manifest existed locally but were
never committed.

| # | Problem | Fix | Covered by |
|---|---|---|---|
| B1 | Makefile line started with spaces; every target failed | Rewritten with tabs; added `train`, `baseline`, `api`, `docker-build`, `docker-run`, `monitor`, `lock` | CI runs `make test` |
| B2 | `.gitignore` excluded tests, infra, docs, split IDs, manifest | Rewritten; only raw/processed data, logs, venv, Terraform state ignored | repo contents |
| B3 | `git_commit: "unknown"` in model metadata | Training refuses unknown or dirty code unless `--allow-dirty`; metadata adds `config_sha256` and `splits_sha256`; v1 retrained from a committed tree | `test_refuses_to_train_from_uncommitted_code`, `test_metadata_records_code_config_and_split_versions` |
| B4 | Unwritable `LOG_PATH` turned every request into a 500; container user couldn't write `/app/logs` | Logger catches `OSError`, warns once, falls back to stdout; Dockerfile creates `/app/logs` owned by the app user | `test_unwritable_log_path_does_not_break_requests`, `test_log_write_failure_mid_run_does_not_break_requests` |
| B5 | Monitor crashed on `docker logs` output, empty logs, missing files | Skips and counts non-JSON lines; empty log -> `insufficient_traffic`; missing file -> clear message, exit 2; container runs with `--no-access-log` | `test_read_log_skips_non_json_lines`, `test_empty_log_reports_insufficient_traffic`, `test_missing_log_file_exits_2`, CI docker job |
| B6 | `features.csv` read back with a lossy float parser | `float_precision="round_trip"` in training, monitor and exploration | `test_training_reads_features_bit_for_bit` (fails without the fix: 1% of `HR_mean` differ); API vs saved score for every test patient, linear and tree model |
| B7 | `insufficient_data` responses could still say `high_risk: true` | `high_risk` is null when no vitals were sent; new `low_count` flag when a vital has < 25% of the training median readings; input contract documented | `test_no_vitals_scored_but_never_alerts`, `test_sparse_readings_flagged_low_count` |
| B8 | `selection_metric` and `threshold_rule` ignored but recorded | `selection_metric` (pr_auc / auroc / brier) drives selection; unknown values for either setting raise | `test_selection_metric_from_config_is_used`, `test_unknown_settings_rejected` |
| B9 | Only top-level packages pinned; moving base image | `requirements.lock` (every package); Docker and CI install from it; base image `python:3.13.1-slim`; CI uses Python 3.13.1; training warns on other Python versions | CI test job |
| B10 | API warning cited "validation" numbers that came from test | Masking now also runs on validation (`masking_val`); warning and monitor text cite it | `test_masking_also_reported_on_validation`, `test_missing_hr_gets_specific_warning` |
| B11 | `"gender": true`, `"age": "60"`, `"icu_type": 3.0` accepted | Strict models; `StrictInt` with bounds for gender and ICU type | `test_invalid_requests_rejected` (6 new cases) |
| B12 | Outcome/record linkage check could never fire | Labels filtered after validation, so orphans are counted and reported | `test_outcome_without_record_is_reported` |
| B13 | Cutoff stored rounded to 6 decimals | Stored unrounded; rounded only for display | `test_threshold_chosen_on_val_not_test` (exact equality) |
| B14 | `roc_curve` could skip the exact cutoff | `drop_intermediate=False` | `test_threshold_considers_every_cutoff` |
| B15 | Data checks used `assert` (removed under `python -O`) | `raise DataError(...)` | `test_overlapping_splits_raise_data_error` |
| B16 | Download not atomic; no check against PhysioNet hashes | Download to `.part`, rename on success; compare with release `SHA256SUMS.txt` (mismatch = error, offline = recorded in manifest) | `test_download_is_atomic`, `test_upstream_mismatch_is_an_error`, `test_upstream_unreachable_is_recorded_not_fatal`, `test_parse_sha256sums` |
| N1 | ICU subgroup mislabelled missing ICU type | `ICUType_missing` included | — |
| N2 | Exploration script ran on import | Moved to `scripts/explore_missing.py` with a `main()` guard and config paths | — |
| N3 | Docstring said "score test once" | Reworded: decisions on validation; test used many times for reporting only | — |
| N4 | `sha256_file` read whole file into memory | Reuses chunked version from `ingest.py` | — |
| N5 | Starlette deprecation warning for `httpx` | Tests use `httpx2` | test run has no warnings |

Found while fixing, not in the review: a request containing `NaN` returned 500 instead of 422,
because the default error handler echoed the input back and `NaN` is not valid JSON. The 422
body now lists fields and reasons only, never submitted values (also keeps patient data out of
error responses). Covered by `test_non_finite_value_rejected`, `test_validation_error_does_not_echo_input`.

Part 2 of the review (missing requirements) is covered by: the restored `README.md`,
`MODEL_CARD.md`, `tests/`, `infra/`, `.github/workflows/ci.yml`; committed `data/splits/`,
`data/manifest.json`, `reports/data_quality.json`; `examples/` with real responses;
and the cost estimate in README section 9.
