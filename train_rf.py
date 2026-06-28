"""
train_rf.py — Entraînement Random Forest supervisé (étape 2).

Labels :
  0 = Normal   ← VolsValidesEtape1 filtrée sur EtapeValidation='1'
  1 = Anomalie ← AnomaliesAIMS + AnomaliesMRO (toutes, y compris TypeAnomalie
                 == "Score IF anormal"). Le RF apprend à distinguer
                 les vrais positifs des faux positifs IF — un recall/precision
                 < 100% est attendu et acceptable.

Features : 8 (FreqComboFamille exclue — même logique que l'IF l'utilise déjà).
Évaluation : cross-validation 5-fold avec encodeur refit par fold.
  → Chaque fold refit les tables de fréquence sans les données test,
    ce qui évite que les normaux aient systématiquement freq > 0.
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve,
)

from config import RF_MODEL_PATH, TABLE_VOLS_VALIDES
from db import read_table
from features import FlightFeatureEncoder, normalize_vols_valides

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RF_N_FEATURES = 8   # toutes les features, FreqComboFamille incluse


def _normalize_anomalies(df, source):
    """Renomme colonnes AIMS/MRO (toutes les anomalies sont conservées)."""
    df = df.copy()
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
        f"  {len(df_anom)} anomalies retenues pour entraînement RF "
        f"({len(df_an_aims)} AIMS + {len(df_an_mro)} MRO)."
    )

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

    # ── 5. Evaluation (CV = metriques honnetes, prod = verification) ────────
    cv_auc = roc_auc_score(y_cv, cv_probas)
    cv_cm  = confusion_matrix(y_cv, cv_preds)
    tn, fp, fn, tp = cv_cm.ravel()

    print("\n" + "=" * 60)
    print("  RESULTAT ENTRAINEMENT RANDOM FOREST")
    print("=" * 60)
    print(f"  Dataset  -- normaux : {len(df_vv)}  |  anomalies : {len(df_anom)}")

    print("\n  -- Cross-validation 5-fold (evaluation sans leakage) -----")
    print(classification_report(y_cv, cv_preds,
          target_names=["Normal", "Anomalie"], digits=4))
    print(f"  AUC-ROC : {cv_auc:.4f}")
    print(f"  TN={tn} | TP={tp} | FP={fp} | FN={fn}")

    feature_names = [
        "NumVolNum", "FleetFamilyCode",
        "FreqVolJour", "FreqVolFamille", "FreqRouteJour",
        "FreqVolRoute", "FreqVolRouteJour", "FreqComboFamille",
    ]
    importances = sorted(
        zip(feature_names, rf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("\n  Importance des features :")
    for name, imp in importances:
        bar = "#" * int(imp * 40)
        print(f"    {name:<22} {imp:.4f}  {bar}")

    # ── 6. Graphiques pour le memoire (metriques CV) ──────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    os.makedirs("figures", exist_ok=True)

    fig_style = dict(figsize=(7, 5), dpi=300)
    colors = {"green": "#54942C", "blue": "#3b6fe0", "red": "#D40C1C",
              "amber": "#e8920c", "teal": "#10a59b", "violet": "#7c5cff"}

    # 6a. Courbe ROC (CV)
    fpr, tpr, _ = roc_curve(y_cv, cv_probas)
    fig, ax = plt.subplots(**fig_style)
    ax.plot(fpr, tpr, color=colors["blue"], lw=2.5, label=f"RF (AUC = {cv_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Aleatoire (AUC = 0.5)")
    ax.set_xlabel("Taux de Faux Positifs (FPR)")
    ax.set_ylabel("Taux de Vrais Positifs (TPR)")
    ax.set_title("Courbe ROC -- Random Forest (CV 5-fold)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig("figures/roc_curve_rf.png")
    plt.close(fig)

    # 6b. Courbe Precision-Recall (CV)
    prec_arr, rec_arr, _ = precision_recall_curve(y_cv, cv_probas)
    fig, ax = plt.subplots(**fig_style)
    ax.plot(rec_arr, prec_arr, color=colors["green"], lw=2.5)
    ax.set_xlabel("Recall (classe Anomalie)")
    ax.set_ylabel("Precision (classe Anomalie)")
    ax.set_title("Courbe Precision-Recall -- Random Forest (CV 5-fold)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig("figures/precision_recall_rf.png")
    plt.close(fig)

    # 6c. Matrice de confusion (CV)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    ax.imshow(cv_cm, cmap="Greens", aspect="auto")
    labels_rc = ["Normal", "Anomalie"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels_rc)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels_rc)
    ax.set_xlabel("Predit"); ax.set_ylabel("Reel")
    ax.set_title("Matrice de confusion -- Random Forest (CV 5-fold)")
    for i in range(2):
        for j in range(2):
            val = cv_cm[i, j]
            ax.text(j, i, f"{val:,}", ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    color="white" if val > cv_cm.max()/2 else "black")
    fig.tight_layout()
    fig.savefig("figures/confusion_matrix_rf.png")
    plt.close(fig)

    # 6d. Importance des features
    imp_names = [n for n, _ in importances]
    imp_vals  = [v for _, v in importances]
    fig, ax = plt.subplots(**fig_style)
    bars = ax.barh(imp_names[::-1], imp_vals[::-1],
                   color=[colors["green"] if v > 0.1 else colors["teal"] for v in imp_vals[::-1]])
    ax.set_xlabel("Importance (Gini)")
    ax.set_title("Importance des features -- Random Forest")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(axis="x", alpha=0.2)
    for bar, val in zip(bars, imp_vals[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.1%}", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig("figures/feature_importance_rf.png")
    plt.close(fig)

    # 6e. Distribution des probabilites RF (CV)
    fig, ax = plt.subplots(**fig_style)
    ax.hist(cv_probas[y_cv == 0], bins=50, alpha=0.7, color=colors["green"],
            label="Normaux", density=True)
    ax.hist(cv_probas[y_cv == 1], bins=50, alpha=0.7, color=colors["red"],
            label="Anomalies", density=True)
    ax.axvline(0.5, color="black", ls="--", lw=1.5, label="Seuil = 0.50")
    ax.set_xlabel("P(Anomalie)")
    ax.set_ylabel("Densite")
    ax.set_title("Distribution des probabilites RF (CV 5-fold)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig("figures/distribution_probas_rf.png")
    plt.close(fig)

    print(f"\n  5 graphiques sauvegardes dans figures/")

    # ── 7. Sauvegarder modele + metriques ─────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    with open(RF_MODEL_PATH, "wb") as f:
        pickle.dump(rf, f)

    metrics = {
        "auc": round(cv_auc, 4),
        "n_normal": int(len(df_vv)),
        "n_anomalie": int(len(df_anom)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "feature_importances": {n: round(float(v), 4) for n, v in
                                zip(feature_names, rf.feature_importances_)},
    }
    with open("models/rf_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Modele RF sauvegarde -> {RF_MODEL_PATH}")
    print(f"  Metriques  -> models/rf_metrics.json")
    print("=" * 60)


if __name__ == "__main__":
    train_rf()
