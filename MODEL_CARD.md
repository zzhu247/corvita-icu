# Model card: Corvita ICU 24 h mortality risk, v1

**Status:** research prototype. Not for clinical decisions.

## Intended use
Rank adult ICU patients by risk of in-hospital death, 24 h after ICU admission,
as a broad screen for review. Output is a score plus data-quality warnings.

## Not intended for
Neonates or children. Individual treatment decisions. Showing raw probabilities to
clinicians. Hospitals other than the source without local validation. Cardiac surgery recovery units.

## Model
Logistic regression (C = 0.01), median imputation and scaling fit on train only.
Inputs: HR, RespRate and Temp from 00:00–24:00 (count, mean, last, missing), age, sex, ICU type.
Alert cutoff 0.131: highest cutoff with validation sensitivity ≥ 0.80.

## Data
PhysioNet/CinC Challenge 2012 set-a. 4,000 adults with ICU stays of 48 h or more.
13.85% in-hospital mortality. Split by patient 70/15/15, stratified, seed 42.
data_version `60ef1f882790`.

## Performance (test, n = 600, 83 deaths)
AUROC 0.74 (0.68–0.80). PR-AUC 0.30 (0.23–0.40); base rate 0.14.
Sensitivity 0.77, specificity 0.58. 47% of patients flagged; PPV 0.23.
Brier 0.110 vs 0.119 for predicting the base rate.

## Known failure modes
- RespRate missingness acts as a ventilation flag. Depends on how the site charts.
- Missing HR lowers scores: alerts fall from 47% to 23% with no error raised.
- Scores are compressed: lowest decile 5% predicted vs 2% observed, highest 29% vs 37%.
- Cardiac surgery recovery: AUROC 0.52, 2 of 7 deaths caught.
- Patients with RespRate recorded (none ventilated): 8 of 13 deaths caught at the default cutoff.

## Missing and sparse data
No vitals in 24 h: a probability is returned but `high_risk` is null (`insufficient_data`).
A vital never measured: scored, flagged `missing_vitals`; missing HR adds a warning.
Far fewer readings than typical: scored, flagged `low_count`. Clients must send every charted reading.

## Fairness
Sensitivity 0.83 for women (30 deaths) and 0.74 for men (53 deaths). Within noise at this sample size.
Not checked: age bands, race or ethnicity (not in the data).

## Limitations
48 h stay requirement creates selection bias at the 24 h prediction time.
One source, around 2012. Small test set. Value ranges and the sensitivity target are untested assumptions.
