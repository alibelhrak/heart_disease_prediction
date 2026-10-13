import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=42)
N = 920

datasets = rng.choice(["Cleveland","Hungarian","Switzerland","VA Long Beach"], size=N, p=[0.31,0.29,0.20,0.20])
num = rng.choice([0,1,2,3,4], size=N, p=[0.45,0.28,0.14,0.08,0.05])
has_disease = num > 0

age = np.where(has_disease, rng.normal(56,9,N).clip(35,77), rng.normal(49,9,N).clip(29,72)).astype(int)
sex = rng.choice(["Male","Female"], size=N, p=[0.68,0.32])
cp_d = rng.choice(["typical angina","atypical angina","non-anginal","asymptomatic"], size=N, p=[0.08,0.15,0.20,0.57])
cp_n = rng.choice(["typical angina","atypical angina","non-anginal","asymptomatic"], size=N, p=[0.16,0.32,0.32,0.20])
cp = np.where(has_disease, cp_d, cp_n)

trestbps = np.where(has_disease, rng.normal(134,18,N).clip(90,200), rng.normal(129,16,N).clip(90,180)).round(0).astype(object)
trestbps[rng.random(N)<0.06] = np.nan
chol = np.where(has_disease, rng.normal(245,55,N).clip(100,420), rng.normal(235,50,N).clip(100,400)).round(0).astype(object)
chol[rng.random(N)<0.30] = np.nan
fbs = rng.choice([True,False], size=N, p=[0.15,0.85]).astype(object)
fbs[rng.random(N)<0.04] = np.nan
restecg = rng.choice(["normal","st-t abnormality","lv hypertrophy"], size=N, p=[0.50,0.17,0.33]).astype(object)
restecg[rng.random(N)<0.01] = np.nan
thalch = np.where(has_disease, rng.normal(138,24,N).clip(70,202), rng.normal(158,20,N).clip(90,202)).round(0).astype(object)
thalch[rng.random(N)<0.06] = np.nan
exang = (rng.random(N) < np.where(has_disease,0.55,0.15)).astype(object)
exang[rng.random(N)<0.06] = np.nan
oldpeak = np.where(has_disease, rng.exponential(1.6,N).clip(0,6.2), rng.exponential(0.6,N).clip(0,4.0)).round(1).astype(object)
oldpeak[rng.random(N)<0.07] = np.nan
slope_d = rng.choice(["upsloping","flat","downsloping"], size=N, p=[0.16,0.60,0.24])
slope_n = rng.choice(["upsloping","flat","downsloping"], size=N, p=[0.46,0.40,0.14])
slope = np.where(has_disease, slope_d, slope_n).astype(object)
slope[rng.random(N)<0.35] = np.nan
ca_d = rng.choice([0,1,2,3], size=N, p=[0.30,0.32,0.22,0.16]).astype(float)
ca_n = rng.choice([0,1,2,3], size=N, p=[0.65,0.22,0.09,0.04]).astype(float)
ca = np.where(has_disease, ca_d, ca_n).astype(object)
ca[rng.random(N)<0.33] = np.nan
thal_d = rng.choice(["normal","fixed defect","reversable defect"], size=N, p=[0.15,0.22,0.63])
thal_n = rng.choice(["normal","fixed defect","reversable defect"], size=N, p=[0.55,0.12,0.33])
thal = np.where(has_disease, thal_d, thal_n).astype(object)
thal[rng.random(N)<0.33] = np.nan

df = pd.DataFrame({"id":range(1,N+1),"age":age,"sex":sex,"dataset":datasets,"cp":cp,
    "trestbps":trestbps,"chol":chol,"fbs":fbs,"restecg":restecg,"thalch":thalch,
    "exang":exang,"oldpeak":oldpeak,"slope":slope,"ca":ca,"thal":thal,"num":num})
df.to_csv("data/raw/heart_disease_uci.csv", index=False)
print(f"Generated {N} rows -> data/raw/heart_disease_uci.csv")
print(df["num"].value_counts().sort_index())
