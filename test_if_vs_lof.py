"""
test_if_vs_lof.py — Comparaison Isolation Forest vs Local Outlier Factor
Fichier standalone, ne modifie rien (ni modeles, ni base).
"""

import sys
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding='utf-8')

from config import TABLE_VOLS_VALIDES, IF_CONFIG, MODEL_PATH
from db import read_table
from features import FlightFeatureEncoder, normalize_vols_valides
from predict import charger_vols_non_matches

print("=" * 65)
print("  COMPARAISON : Isolation Forest vs Local Outlier Factor")
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

# -- Meme split 80/20 ------------------------------------------------------
X_train_df, _ = train_test_split(df_vv, test_size=0.2, random_state=42)
print(f"  Split 80/20 : {len(X_train_df)} entrainement")

# -- Encodage ---------------------------------------------------------------
encoder = FlightFeatureEncoder()
encoder.fit(X_train_df)
X_train = encoder.transform(X_train_df)
X_nm = encoder.transform(df_nm_enc)
print(f"  Features : {X_train.shape[1]}")

# -- Reference : modele sauvegarde -----------------------------------------
print("\n" + "-" * 65)
print("  REFERENCE : modele IF sauvegarde")
print("-" * 65)
with open(MODEL_PATH, "rb") as f:
    iso_saved = pickle.load(f)
encoder_saved = FlightFeatureEncoder.load()
X_nm_saved = encoder_saved.transform(df_nm_enc)
preds_saved = np.where(iso_saved.score_samples(X_nm_saved) >= iso_saved.offset_, 1, -1)
n_ref = int((preds_saved == -1).sum())
print(f"  Anomalies : {n_ref}")

# =========================================================================
#  1. ISOLATION FOREST
# =========================================================================
print("\n" + "-" * 65)
print("  1) ISOLATION FOREST (contamination=5%)")
print("-" * 65)

t0 = time.perf_counter()
iso = IsolationForest(**IF_CONFIG)
iso.fit(X_train)
t_train_if = time.perf_counter() - t0

t0 = time.perf_counter()
scores_if = iso.score_samples(X_nm)
preds_if = np.where(scores_if >= iso.offset_, 1, -1)
t_score_if = time.perf_counter() - t0

n_if = int((preds_if == -1).sum())
print(f"  Entrainement : {t_train_if:.3f} sec")
print(f"  Scoring      : {t_score_if:.3f} sec")
print(f"  Anomalies    : {n_if}")

# =========================================================================
#  2. LOCAL OUTLIER FACTOR (novelty=True pour scorer de nouvelles donnees)
# =========================================================================
contam_values = [0.05, 0.10]

for contam in contam_values:
    print("\n" + "-" * 65)
    print(f"  2) LOCAL OUTLIER FACTOR (contamination={contam:.0%})")
    print("-" * 65)

    t0 = time.perf_counter()
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=contam,
        novelty=True,
        n_jobs=-1,
    )
    lof.fit(X_train)
    t_train_lof = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores_lof = lof.score_samples(X_nm)
    preds_lof = lof.predict(X_nm)
    t_score_lof = time.perf_counter() - t0

    n_lof = int((preds_lof == -1).sum())
    print(f"  Entrainement : {t_train_lof:.3f} sec")
    print(f"  Scoring      : {t_score_lof:.3f} sec")
    print(f"  Anomalies    : {n_lof}")

    # Concordance avec IF
    agree = int((preds_if == preds_lof).sum())
    print(f"  Concordance IF/LOF : {agree}/{len(preds_if)} ({100*agree/len(preds_if):.1f}%)")

    # Combien d'anomalies en commun
    both_anom = int(((preds_if == -1) & (preds_lof == -1)).sum())
    only_if = int(((preds_if == -1) & (preds_lof == 1)).sum())
    only_lof = int(((preds_if == 1) & (preds_lof == -1)).sum())
    print(f"  Anomalies communes : {both_anom}")
    print(f"  Uniquement IF      : {only_if}")
    print(f"  Uniquement LOF     : {only_lof}")

# =========================================================================
#  3. LOF avec differents n_neighbors
# =========================================================================
print("\n" + "-" * 65)
print("  3) SENSIBILITE LOF au parametre n_neighbors (contamination=5%)")
print("-" * 65)
print(f"  {'n_neighbors':<14s}  {'Anomalies':>10s}  {'Concordance IF':>15s}  {'Temps (sec)':>12s}")
print(f"  {'-'*14}  {'-'*10}  {'-'*15}  {'-'*12}")

for k in [5, 10, 20, 50, 100]:
    t0 = time.perf_counter()
    lof_k = LocalOutlierFactor(n_neighbors=k, contamination=0.05, novelty=True, n_jobs=-1)
    lof_k.fit(X_train)
    preds_k = lof_k.predict(X_nm)
    t_k = time.perf_counter() - t0
    n_k = int((preds_k == -1).sum())
    agree_k = int((preds_if == preds_k).sum())
    print(f"  {k:<14d}  {n_k:>10d}  {agree_k:>10d} ({100*agree_k/len(preds_if):.1f}%)  {t_k:>12.3f}")

# =========================================================================
#  TABLEAU RECAPITULATIF
# =========================================================================
print(f"\n{'=' * 65}")
print("  TABLEAU RECAPITULATIF (contamination=5%, n_neighbors=20)")
print(f"{'=' * 65}")

# Re-run LOF 5% for final comparison
lof_final = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True, n_jobs=-1)
t0 = time.perf_counter()
lof_final.fit(X_train)
t_lof_train = time.perf_counter() - t0
t0 = time.perf_counter()
preds_lof_final = lof_final.predict(X_nm)
t_lof_score = time.perf_counter() - t0
n_lof_final = int((preds_lof_final == -1).sum())
agree_final = int((preds_if == preds_lof_final).sum())

print(f"  {'Critere':<30s}  {'Isolation Forest':>18s}  {'LOF':>18s}")
print(f"  {'-'*30}  {'-'*18}  {'-'*18}")
print(f"  {'Approche':<30s}  {'Coupes aleatoires':>18s}  {'Densite locale':>18s}")
print(f"  {'Complexite entrainement':<30s}  {'O(n)':>18s}  {'O(n x k)':>18s}")
print(f"  {'Entrainement (sec)':<30s}  {t_train_if:>18.3f}  {t_lof_train:>18.3f}")
print(f"  {'Scoring (sec)':<30s}  {t_score_if:>18.3f}  {t_lof_score:>18.3f}")
print(f"  {'Anomalies detectees':<30s}  {n_if:>18d}  {n_lof_final:>18d}")
print(f"  {'Ref (modele sauvegarde)':<30s}  {n_ref:>18d}  {'--':>18s}")
print(f"  {'Sensibilite hyperparametre':<30s}  {'Faible':>18s}  {'Elevee (k)':>18s}")
print(f"  {'Concordance':<30s}  {'':>18s}  {f'{100*agree_final/len(preds_if):.1f}%':>18s}")

print(f"\n{'=' * 65}")
