"""
test_ohe_vs_fe.py — Comparaison OHE vs Frequency Encoding
Fichier standalone, ne modifie rien (ni modeles, ni base).
"""

import sys
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding='utf-8')

from config import TABLE_VOLS_VALIDES, IF_CONFIG, MODEL_PATH
from db import read_table
from features import FlightFeatureEncoder, normalize_vols_valides
from predict import charger_vols_non_matches

print("=" * 65)
print("  COMPARAISON : One-Hot Encoding vs Frequency Encoding")
print("  (fichier standalone, aucune modification)")
print("=" * 65)

# -- Charger les donnees ----------------------------------------------------
print("\nChargement des donnees...")
raw_vv = read_table(TABLE_VOLS_VALIDES)
if "EtapeValidation" in raw_vv.columns:
    raw_vv = raw_vv[raw_vv["EtapeValidation"].astype(str) == "1"]
df_vv = normalize_vols_valides(raw_vv)
df_vv = df_vv.dropna(subset=["Date", "Matricule", "NumVol", "AeroDepart", "AeroArriv"])
print(f"  Vols valides : {len(df_vv)}")

df_nm = charger_vols_non_matches()
df_nm_enc = df_nm.rename(columns={
    "MatriculeAIMS": "Matricule", "NumVolAIMS": "NumVol",
    "AeroDepartAIMS": "AeroDepart", "AeroArrivAIMS": "AeroArriv",
    "DateAIMS": "Date",
}).copy()
df_nm_enc["Date"] = pd.to_datetime(df_nm_enc["Date"], errors="coerce")
print(f"  Vols non-matches : {len(df_nm)}")

# -- Meme split 80/20 pour les deux ----------------------------------------
X_train_df, _ = train_test_split(df_vv, test_size=0.2, random_state=42)
print(f"  Split 80/20 : {len(X_train_df)} entrainement")

# -- Reference : modele sauvegarde (FE) ------------------------------------
print("\n" + "-" * 65)
print("  REFERENCE : modele sauvegarde (Frequency Encoding, 8 features)")
print("-" * 65)
with open(MODEL_PATH, "rb") as f:
    iso_saved = pickle.load(f)
encoder_saved = FlightFeatureEncoder.load()
X_nm_fe = encoder_saved.transform(df_nm_enc)
scores_ref = iso_saved.score_samples(X_nm_fe)
preds_ref = np.where(scores_ref >= iso_saved.offset_, 1, -1)
n_ref = int((preds_ref == -1).sum())
print(f"  Anomalies detectees : {n_ref}")

# =========================================================================
#  1. FREQUENCY ENCODING (re-entraine, meme split)
# =========================================================================
print("\n" + "-" * 65)
print("  1) FREQUENCY ENCODING (8 features)")
print("-" * 65)

t0 = time.perf_counter()
enc_fe = FlightFeatureEncoder()
enc_fe.fit(X_train_df)
X_train_fe = enc_fe.transform(X_train_df)
t_enc_fe = time.perf_counter() - t0

t0 = time.perf_counter()
iso_fe = IsolationForest(**IF_CONFIG)
iso_fe.fit(X_train_fe)
t_train_fe = time.perf_counter() - t0

t0 = time.perf_counter()
X_nm_fe2 = enc_fe.transform(df_nm_enc)
scores_fe = iso_fe.score_samples(X_nm_fe2)
preds_fe = np.where(scores_fe >= iso_fe.offset_, 1, -1)
t_score_fe = time.perf_counter() - t0

n_fe = int((preds_fe == -1).sum())
print(f"  Dimensions       : {X_train_fe.shape[1]} features")
print(f"  Encodage         : {t_enc_fe:.3f} sec")
print(f"  Entrainement IF  : {t_train_fe:.3f} sec")
print(f"  Scoring          : {t_score_fe:.3f} sec")
print(f"  Anomalies        : {n_fe}")

# =========================================================================
#  2. ONE-HOT ENCODING
# =========================================================================
print("\n" + "-" * 65)
print("  2) ONE-HOT ENCODING")
print("-" * 65)

cat_cols = ["NumVol", "Matricule", "AeroDepart", "AeroArriv"]
df_train_cat = X_train_df[cat_cols].astype(str)
df_nm_cat = df_nm_enc[cat_cols].astype(str)

t0 = time.perf_counter()
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_train_ohe = ohe.fit_transform(df_train_cat)
t_enc_ohe = time.perf_counter() - t0

n_ohe_cols = X_train_ohe.shape[1]

t0 = time.perf_counter()
iso_ohe = IsolationForest(**IF_CONFIG)
iso_ohe.fit(X_train_ohe)
t_train_ohe = time.perf_counter() - t0

t0 = time.perf_counter()
X_nm_ohe = ohe.transform(df_nm_cat)
scores_ohe = iso_ohe.score_samples(X_nm_ohe)
preds_ohe = np.where(scores_ohe >= iso_ohe.offset_, 1, -1)
t_score_ohe = time.perf_counter() - t0

n_ohe = int((preds_ohe == -1).sum())
print(f"  Dimensions       : {n_ohe_cols} features")
print(f"  Encodage         : {t_enc_ohe:.3f} sec")
print(f"  Entrainement IF  : {t_train_ohe:.3f} sec")
print(f"  Scoring          : {t_score_ohe:.3f} sec")
print(f"  Anomalies        : {n_ohe}")

# -- Gestion valeurs inconnues ---------------------------------------------
known_cats = set(df_train_cat.values.flatten())
nm_cats = set(df_nm_cat.values.flatten())
unknown = nm_cats - known_cats
print(f"  Valeurs inconnues (non-matches) : {len(unknown)}")
print(f"    -> OHE les ignore (vecteur nul), FE les marque a -1.0")

# =========================================================================
#  TABLEAU RECAPITULATIF
# =========================================================================
print(f"\n{'=' * 65}")
print("  TABLEAU RECAPITULATIF")
print(f"{'=' * 65}")
print(f"  {'Critere':<30s}  {'Freq. Encoding':>16s}  {'One-Hot':>16s}")
print(f"  {'-'*30}  {'-'*16}  {'-'*16}")
print(f"  {'Dimensions':<30s}  {X_train_fe.shape[1]:>16d}  {n_ohe_cols:>16d}")
print(f"  {'Encodage (sec)':<30s}  {t_enc_fe:>16.3f}  {t_enc_ohe:>16.3f}")
print(f"  {'Entrainement IF (sec)':<30s}  {t_train_fe:>16.3f}  {t_train_ohe:>16.3f}")
print(f"  {'Scoring (sec)':<30s}  {t_score_fe:>16.3f}  {t_score_ohe:>16.3f}")
print(f"  {'Anomalies detectees':<30s}  {n_fe:>16d}  {n_ohe:>16d}")
print(f"  {'Ref (modele sauvegarde)':<30s}  {n_ref:>16d}  {'--':>16s}")
print(f"  {'Valeurs inconnues gerees':<30s}  {'Oui (-1.0)':>16s}  {'Non (ignore)':>16s}")
ratio = n_ohe_cols / X_train_fe.shape[1]
print(f"\n  Reduction dimensionnelle : {n_ohe_cols} -> {X_train_fe.shape[1]} = {(1 - 1/ratio)*100:.1f}%")

# -- Concordance des predictions -------------------------------------------
agree = int((preds_fe == preds_ohe).sum())
print(f"  Concordance predictions FE/OHE : {agree}/{len(preds_fe)} ({100*agree/len(preds_fe):.1f}%)")

print(f"\n{'=' * 65}")
