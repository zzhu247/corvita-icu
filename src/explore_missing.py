import json
import pandas as pd
from src.parse import load_records
from src.features import apply_cutoff

X = pd.read_csv("data/processed/features.csv", index_col="RecordID")
y = pd.read_csv("data/processed/labels.csv", index_col="RecordID")["In-hospital_death"]
df = X.join(y)

rep = json.load(open("reports/data_quality.json"))
print(json.dumps({k: rep[k] for k in ["time", "vitals", "admission"]}, indent=2))

icu = df[[f"ICUType_{k}" for k in range(1, 5)] + ["ICUType_missing"]].idxmax(axis=1)
print("\nRespRate missing rate by ICU type:")
print(df.groupby(icu)["RespRate_missing"].mean().round(3))

print("\nDeath rate by RespRate_missing:")
print(df.groupby("RespRate_missing")["In-hospital_death"].agg(["size", "mean"]).round(3))

print("\nDeath rate by all-vitals-missing (HR_missing):")
print(df.groupby("HR_missing")["In-hospital_death"].agg(["size", "mean"]).round(3))



_, obs = load_records("data/raw/set-a")
w = apply_cutoff(obs, 1440)
vent = set(w.loc[(w["Parameter"] == "MechVent") & (w["Value"] == 1), "RecordID"])
df["vent_24h"] = df.index.isin(vent).astype(int)

print("\nRespRate missing rate by ventilation (exploration only, not a feature):")
print(df.groupby("vent_24h")["RespRate_missing"].agg(["size", "mean"]).round(3))