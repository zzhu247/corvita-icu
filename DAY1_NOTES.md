# Day 1 notes: data preparation, 24 h cutoff, features, splits

## How to run

```powershell
python -m pip install -r requirements.lock
python -m pytest -q          # synthetic data, no download needed
python -m src.ingest         # download, check PhysioNet SHA256SUMS, write/verify manifest
python -m src.run_day1       # parse -> validate -> features -> splits
```

With `make`: `make test`, `make day1`.

If the download fails, put `set-a.tar.gz` and `Outcomes-a.txt` in `data/raw/` and run
`python -m src.ingest --no-download`.

The first run writes `data/manifest.json`. Every later run recomputes the hashes and stops if
anything changed. To change the source data on purpose: `python -m src.ingest --update`.

## Data flow

```
set-a/*.txt + Outcomes-a.txt
  ingest.py    SHA-256 -> data/manifest.json, data_version
  parse.py     admission table + long observation table; -1 -> NaN
  validate.py  IDs, labels, linkage, times, ranges, units -> reports/data_quality.json
  features.py  keep 0 <= minutes <= 1440, then count / mean / last / missing
  split.py     70/15/15, stratified on the label, seed 42 -> data/splits/*_ids.csv
```

## Design choices

**1. Why checksums at three levels?**
- Against PhysioNet's published `SHA256SUMS.txt`: catches a corrupt or wrong download.
- The tar file and the outcomes file: catches a changed source.
- Every extracted record file: catches edits made after extraction, which the tar hash cannot see.

`data_version` is the first 12 characters of the combined hash. Model metadata and every API
response carry it. Downloads go to a `.part` file and are renamed only when complete, so an
interrupted download never looks finished.

**2. Why is the cutoff in `features.py`, not `parse.py`?**
Parsing only reads and never drops data. The cutoff is part of the feature definition.
The API must apply the same cutoff, so it lives in the module the API imports.

**3. How is 24:00 handled?**
`minutes <= 1440`: 24:00 is kept, 24:01 is dropped.
`test_cutoff_keeps_2400_drops_2401` and `test_later_data_never_reaches_features` check this.

**4. What does "last" mean?**
The latest reading by time, not the last line in the file. The sort is stable, so with two
readings at the same minute the one later in the file wins. A test feeds readings out of order.

**5. How are implausible values handled?**
Values outside the ranges in `config.yaml` become missing and are removed, so they never count
as a reading. Each removal is counted in the report.
HR = 0 is usually a disconnected monitor. Temp values in 80–110 would suggest Fahrenheit;
they are counted separately as a unit check. Converting them was the alternative; I removed
them because there is no way to confirm they really are Fahrenheit.
The same check (`out_of_range()`) is used by the API.

**6. How is missingness tracked?**
- Each vital has a `_missing` flag; `count` is 0 and `mean` / `last` are NaN.
- Median imputation lives in the Day 2 sklearn Pipeline and is fit on train only.
- Day 1 learns nothing from the data. One-hot encoding uses fixed category lists from config,
  so train, val, test and the API encode identically.

**7. How are outcome fields kept out of the features?**
`load_outcomes()` returns only RecordID and the label; everything else is dropped on read.
The pipeline raises `DataError` if any outcome column appears in the features, and a test checks it.

**8. Why split by RecordID?**
The prediction unit is the patient, so a patient may appear in one group only.
Two stratified splits: first 70%, then the remaining 30% in half.
`check_splits()` raises `DataError` (not `assert`, which `python -O` would remove) unless:
groups do not overlap, together they cover every patient, and each group's death rate is
within 2 points of the overall rate. Split IDs are saved and committed; Day 2 reads them.

**9. What stops the run and what is only counted?**
- Stops (`DataError`): duplicate IDs, a record without an outcome, a label that is not 0/1,
  overlapping or incomplete splits.
- Counted and cleaned: bad time strings, out-of-range values, unknown categories,
  outcome rows without a record (reported as a warning).

## Results on the real data (data_version `60ef1f882790`)

- 4,000 patients, 554 in-hospital deaths (13.85%).
- Splits 2,800 / 600 / 600; death rate 13.8% in each.
- 780,715 observation rows after 24:00 removed.
- Implausible values removed: HR 11, RespRate 62, Temp 130. The Temp minimum was -17.8 °C,
  which is 0 °F converted: a "not measured" code, not a reading. No Fahrenheit values found.
- Patients with no reading in the 24 h window: HR 1.8%, Temp 1.8%, RespRate 72.6%.
- RespRate missing by ICU type: coronary 64.5%, cardiac surgery recovery 94.9%,
  medical 62.3%, surgical 73.0%.
- Death rate: 16.2% when RespRate is missing, 7.7% when recorded. Not missing at random.
- All three vitals missing: 71 patients, death rate 9.9% (too few to interpret).
- Sex missing for 3 patients; age and ICU type never missing.

**Ventilation check** (`scripts/explore_missing.py`): all 2,405 patients with `MechVent = 1`
in the first 24 h have no RespRate (100%). Among the 1,595 others, 31.2% have none.
So RespRate missingness mostly records a charting rule: RespRate is not charted for ventilated
patients. `RespRate_missing` therefore acts as a proxy for ventilation.
- This is not outcome leakage: ventilation status is known within 24 h.
- It does depend on how this hospital charts. A site that charts RespRate on ventilated
  patients would change what the flag means, silently.
- `MechVent` was used to explain the pattern only. It is not a model input.

## Limitations from Day 1

- Set A only includes adults who stayed at least 48 h in the ICU. At 24 h you cannot know who
  will reach 48 h, so the model carries selection bias. Early deaths and early transfers are
  not in the data, and performance on them is unknown.
- The value ranges are my assumptions and need clinical sign-off.
- Adults only. Not for neonates.

## Synthetic test data

`tests/conftest.py` writes 200 fake patients in the exact PhysioNet format: one third have
RespRate, about 5% have unknown sex, readings span 48 h. CI has no access to real data, so all
tests run on these.