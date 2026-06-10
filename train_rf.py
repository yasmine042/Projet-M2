"""
train_rf.py — Entraînement Random Forest supervisé (étape 2).

Labels :
  0 = Normal   ← VolsValidesEtape1 filtrée sur EtapeValidation='1'
  1 = Anomalie ← AnomaliesAIMS + AnomaliesMRO, sans 'Score IF anormal'

Features : 8 (FreqComboFamille exclue — même logique que l'IF l'utilise déjà).
Évaluation : cross-validation 5-fold avec encodeur refit par fold.
  → Chaque fold refit les tables de fréquence sans les données test,
    ce qui évite que les normaux aient systématiquement freq > 0.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
)

from config import RF_MODEL_PATH, TABLE_VOLS_VALIDES
from db import read_table
from features import FlightFeatureEncoder, normalize_vols_valides, get_fleet_family

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RF_N_FEATURES = 8   # FreqComboFamille (index 8) exclue du RF


def _normalize_anomalies(df, source):
    """Renomme colonnes AIMS/MRO et retire les 'Score IF anormal'."""
    df = df.copy()
    if "TypeAnomalie" in df.columns:
        avant = len(df)
        df = df[df["TypeAnomalie"] != "Score IF anormal"]
        retires = avant - len(df)
        if retires:
            logger.info(f"  {retires} lignes 'Score IF anormal' retirées ({source}).")
    rename = {
        "AIMS": {"MatriculeAIMS": "Matricule", "NumVolAIMS": "NumVol",
                 "AeroDepartAIMS": "AeroDepart", "AeroArrivAIMS": "AeroArriv",
                 "DateAIMS": "Date"},
        "MRO":  {"MatriculeMRO": "Matricule",  "NumVolMRO": "NumVol",
                 "AeroDepartMRO": "AeroDepart", "AeroArrivMRO": "AeroArriv",
                 "DateMRO": "Date"},
    }
    df = df.rename(columns=rename[source])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date", "Matricule", "NumVol", "AeroDepart", "AeroArriv"])


def _filtrer_remplacement_famille(df_anom, df_vv):
    """Retire les anomalies où un matricule de même famille est dans VolsValides."""
    idx_familles = (
        df_vv.groupby(["NumVol", "AeroDepart", "AeroArriv"])["Matricule"]
        .apply(lambda mats: {get_fleet_family(m) for m in mats} - {0})
        .to_dict()
    )
    def est_remplacement(row):
        fam = get_fleet_family(str(row["Matricule"]))
        if fam == 0:
            return False
        cle = (str(row["NumVol"]), str(row["AeroDepart"]), str(row["AeroArriv"]))
        return fam in idx_familles.get(cle, set())
    avant  = len(df_anom)
    masque = df_anom.apply(est_remplacement, axis=1)
    df_ok  = df_anom[~masque].reset_index(drop=True)
    retires = avant - len(df_ok)
    if retires:
        logger.info(f"  {retires} remplacement(s) intra-famille retirés.")
    return df_ok


def train_rf():

    # ── 1. VolsValides EtapeValidation='1' (label=0) ─────────────────────────
    logger.info("Chargement VolsValidesEtape1 EtapeValidation='1' (label=0)...")
    raw_vv = read_table(TABLE_VOLS_VALIDES)
    if "EtapeValidation" in raw_vv.columns:
        avant  = len(raw_vv)
        raw_vv = raw_vv[raw_vv["EtapeValidation"].astype(str) == "1"]
        logger.info(f"  EtapeValidation='1' : {len(raw_vv)} / {avant} lignes conservées.")
    else:
        logger.warning("  Colonne EtapeValidation absente — toutes les lignes utilisées.")
    df_vv = normalize_vols_valides(raw_vv)
    df_vv = df_vv.dropna(subset=["Date", "Matricule", "NumVol", "AeroDepart", "AeroArriv"])
    logger.info(f"  {len(df_vv)} vols valides après nettoyage.")

    # ── 2. Anomalies IF (label=1) ─────────────────────────────────────────────
    logger.info("Chargement AnomaliesAIMS + AnomaliesMRO (label=1)...")
    try:
        df_an_aims = _normalize_anomalies(read_table("AnomaliesAIMS"), "AIMS")
    except Exception:
        df_an_aims = pd.DataFrame()
        logger.warning("  AnomaliesAIMS introuvable.")
    try:
        df_an_mro = _normalize_anomalies(read_table("AnomaliesMRO"), "MRO")
    except Exception:
        df_an_mro = pd.DataFrame()
        logger.warning("  AnomaliesMRO introuvable.")

    df_anom = pd.concat([df_an_aims, df_an_mro], ignore_index=True)
    if df_anom.empty:
        logger.error("Aucune anomalie disponible. Lancez d'abord le pipeline IF.")
        return
    logger.info(
        f"  {len(df_anom)} anomalies après filtre 'Score IF anormal' "
        f"({len(df_an_aims)} AIMS + {len(df_an_mro)} MRO)."
    )
    logger.info("Filtre remplacements intra-famille...")
    df_anom = _filtrer_remplacement_famille(df_anom, df_vv)
    logger.info(f"  {len(df_anom)} anomalies retenues pour entraînement RF.")

    # ── 3. Cross-validation 5-fold — évaluation sans leakage ─────────────────
    #
    # Problème structurel : les tables de fréquence sont construites depuis
    # VolsValides. Si on évalue sur les mêmes données, les normaux ont TOUJOURS
    # freq > 0 et les anomalies TOUJOURS freq = -1.0 → séparation triviale.
    #
    # Fix : pour chaque fold, on refit l'encodeur uniquement sur les normaux
    # du train. Les normaux du test peuvent alors avoir freq = -1.0 si leur
    # combinaison n'apparaît pas dans le train → évaluation réaliste.
    #
    logger.info("Cross-validation 5-fold (encodeur refit par fold)...")

    ECOLS = ["Matricule", "NumVol", "AeroDepart", "AeroArriv", "Date"]
    df_cv = pd.concat([
        df_vv[ECOLS].assign(_lbl=0),
        df_anom[ECOLS].assign(_lbl=1),
    ], ignore_index=True)
    y_cv = df_cv.pop("_lbl").values

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_preds  = np.zeros(len(df_cv), dtype=int)
    cv_probas = np.zeros(len(df_cv))

    for fold_i, (tr_idx, te_idx) in enumerate(kf.split(df_cv, y_cv), 1):
        df_tr = df_cv.iloc[tr_idx].reset_index(drop=True)
        df_te = df_cv.iloc[te_idx].reset_index(drop=True)
        y_tr  = y_cv[tr_idx]

        # Encodeur refit sur les normaux du train uniquement
        enc_fold = FlightFeatureEncoder()
        enc_fold.fit(df_tr[y_tr == 0].reset_index(drop=True))

        X_tr = enc_fold.transform(df_tr)[:, :RF_N_FEATURES]
        X_te = enc_fold.transform(df_te)[:, :RF_N_FEATURES]

        rf_fold = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        )
        rf_fold.fit(X_tr, y_tr)
        cv_preds[te_idx]  = rf_fold.predict(X_te)
        cv_probas[te_idx] = rf_fold.predict_proba(X_te)[:, 1]
        logger.info(f"  Fold {fold_i}/5 terminé")

    # ── 4. Modèle final — entraîné sur TOUTES les données ────────────────────
    # Le modèle de production utilise l'encodeur complet (fitté sur 100% des données)
    # pour maximiser la précision des fréquences en production.
    logger.info("Entraînement modèle final (100% des données, encodeur production)...")
    encoder = FlightFeatureEncoder.load()
    X_final = np.vstack([
        encoder.transform(df_vv)[:, :RF_N_FEATURES],
        encoder.transform(df_anom)[:, :RF_N_FEATURES],
    ])
    y_final = np.array([0] * len(df_vv) + [1] * len(df_anom))

    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_final, y_final)
    logger.info("  Modèle final entraîné.")

    # ── 5. Affichage ──────────────────────────────────────────────────────────
    cm = confusion_matrix(y_cv, cv_preds)
    tn, fp, fn, tp = cm.ravel()
    auc = roc_auc_score(y_cv, cv_probas)

    print("\n" + "=" * 60)
    print("  RÉSULTAT ENTRAÎNEMENT RANDOM FOREST")
    print("=" * 60)
    print(f"  Dataset  — normaux : {(y_cv==0).sum()}  |  anomalies : {(y_cv==1).sum()}")
    print(f"  Évaluation : cross-validation 5-fold (encodeur refit par fold)")

    print("\n  ── Métriques cross-validation (sans leakage) ────────────")
    print(classification_report(y_cv, cv_preds, target_names=["Normal", "Anomalie"],
                                 digits=4))

    print(f"  AUC-ROC (CV) : {auc:.4f}  (1.0 = parfait, 0.5 = aléatoire)")

    print("\n  ── Matrice de confusion (CV cumulée) ────────────────────")
    print(f"                    Prédit Normal   Prédit Anomalie")
    print(f"  Réel Normal     {tn:>14}   {fp:>15}")
    print(f"  Réel Anomalie   {fn:>14}   {tp:>15}")
    print(f"\n  TN={tn} | TP={tp} | FP={fp} | FN={fn}")

    feature_names = [
        "NumVolNum", "MatriculeCode", "FleetFamilyCode",
        "FreqVolJour", "FreqVolFamille", "FreqRouteJour",
        "FreqVolRoute", "FreqVolRouteJour",
        # FreqComboFamille exclue — utilisée par l'IF uniquement
    ]
    importances = sorted(
        zip(feature_names, rf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("\n  Importance des features (modèle final) :")
    for name, imp in importances:
        bar = "█" * int(imp * 40)
        print(f"    {name:<22} {imp:.4f}  {bar}")

    # ── 6. Sauvegarder ────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    with open(RF_MODEL_PATH, "wb") as f:
        pickle.dump(rf, f)
    print(f"\n  Modèle RF sauvegardé -> {RF_MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train_rf()
