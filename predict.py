"""
predict.py — Détection d'anomalies sur les vols non matchés AIMS/MRO.

L'Isolation Forest est appliqué uniquement aux vols qui n'ont pas trouvé
de correspondance dans VolsValidesEtape1 (via IDAIMS / IDMRO).

Tables produites :
  - AnomaliesAIMS   : vols AIMS non matchés + ScoreAnomalie + Statut + RaisonAnomalie
  - AnomaliesMRO    : vols MRO  non matchés + ScoreAnomalie + Statut + RaisonAnomalie
  - VolsManquantsIF : vols manquants détectés par analyse de rotation (tous vols)
"""

import logging
import numpy as np
import pandas as pd

from config import MODEL_PATH, TABLE_VOLS_VALIDES
from db import read_table, read_query, write_table
from features import (
    FlightFeatureEncoder,
    normalize_aims, normalize_mro, normalize_vols_valides,
)

import pickle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT MODELE
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Modele charge.")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DONNEES
# ══════════════════════════════════════════════════════════════════════════════

def charger_ref_historique():
    """Charge VolsValidesEtape1 normalisé — référence pour analyser_raison_anomalie."""
    raw = read_table(TABLE_VOLS_VALIDES)
    df  = normalize_vols_valides(raw)
    return df.dropna(subset=["Date", "Matricule", "NumVol", "AeroDepart", "AeroArriv"])


def charger_vols_non_matches():
    """
    Charge les vols AIMS et MRO qui n'ont pas trouvé de match ETL,
    identifiés via IDAIMS / IDMRO absents de VolsValidesEtape1.

    Colonnes retournées (uniformisées AIMS + MRO) :
        IDAIMS, source_table, jour_semaine,
        MatriculeAIMS, NumVolAIMS, AeroDepartAIMS, AeroArrivAIMS, DateAIMS
    """

    # ── AIMS non matchés ──────────────────────────────────────────────────────
    df_aims = read_query(f"""
        SELECT
            a.IDAIMS,
            'AIMS'                          AS source_table,
            DATEPART(DW, a.Date)            AS jour_semaine,
            LTRIM(RTRIM(a.Matricule))       AS MatriculeAIMS,
            LTRIM(RTRIM(a.NumVol))          AS NumVolAIMS,
            LTRIM(RTRIM(a.AeroDepart))      AS AeroDepartAIMS,
            LTRIM(RTRIM(a.AeroArriv))       AS AeroArrivAIMS,
            CAST(a.Date AS DATE)            AS DateAIMS
        FROM AIMS a
        WHERE a.IDAIMS NOT IN (
            SELECT IDAIMS
            FROM {TABLE_VOLS_VALIDES}
            WHERE IDAIMS IS NOT NULL
        )
    """)

    # ── MRO non matchés ───────────────────────────────────────────────────────
    df_mro = read_query(f"""
        SELECT
            m.IDMRO                         AS IDAIMS,
            'MRO'                           AS source_table,
            DATEPART(DW, m.Date_Vol)        AS jour_semaine,
            LTRIM(RTRIM(m.registr))         AS MatriculeAIMS,
            REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
                UPPER(m.[flight no.]),
                'SF',''),'AH',''),'SE',''),'SD',''),'XF','')
                                            AS NumVolAIMS,
            LTRIM(RTRIM(m.[from]))          AS AeroDepartAIMS,
            LTRIM(RTRIM(m.[to]))            AS AeroArrivAIMS,
            CAST(m.Date_Vol AS DATE)        AS DateAIMS
        FROM MRO m
        WHERE m.IDMRO NOT IN (
            SELECT IDMRO
            FROM {TABLE_VOLS_VALIDES}
            WHERE IDMRO IS NOT NULL
        )
    """)

    df_tous = pd.concat([df_aims, df_mro], ignore_index=True)
    logger.info(
        f"Vols non matches : {len(df_tous)} "
        f"(AIMS={len(df_aims)}, MRO={len(df_mro)})"
    )
    return df_tous


# ══════════════════════════════════════════════════════════════════════════════
# ATTRIBUTION DES STATUTS
# ══════════════════════════════════════════════════════════════════════════════

def attribuer_statut(scores, predictions):
    """
    Isolation Forest :
        predict == 1  → Normal
        predict == -1 → Anomalie

    Classification :
        - Très Suspect : top 6% anomalies les plus extrêmes
        - Suspect      : autres anomalies
    """
    scores_anormaux = scores[predictions == -1]

    if len(scores_anormaux) == 0:
        logger.warning("Aucune anomalie detectee par Isolation Forest.")
        return ["Normal"] * len(scores), None

    seuil_tres = np.percentile(scores_anormaux, 6)
    logger.info(f"  Seuil Très Suspect : {seuil_tres:.4f}")

    statuts = []
    for score, pred in zip(scores, predictions):
        if pred == 1:
            statuts.append("Normal")
        elif score < seuil_tres:
            statuts.append("Très Suspect")
        else:
            statuts.append("Suspect")

    return statuts, seuil_tres


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSE RAISON ANOMALIE
# ══════════════════════════════════════════════════════════════════════════════

def analyser_raison_anomalie(df_volsvalides, row, score, seuil_tres):
    """
    Compare le vol suspect avec VolsValides pour expliquer l'anomalie.
    df_volsvalides : DataFrame normalisé de VolsValidesEtape1
    """
    raisons = []

    nv  = row["NumVol"]
    dep = row["AeroDepart"]
    arr = row["AeroArriv"]
    mat = row["Matricule"]
    dat = row["Date"]

    # ── 1. NumVol jamais vu dans VolsValides ──────────────────────────────────
    numvols_connus = set(df_volsvalides["NumVol"].unique())
    if nv not in numvols_connus:
        raisons.append(f"NumVol {nv} jamais vu dans vols valides")
        return " | ".join(raisons)

    # ── 2. Route (NumVol + Depart + Arrivee) jamais vue ──────────────────────
    match_route = df_volsvalides[
        (df_volsvalides["NumVol"]     == nv) &
        (df_volsvalides["AeroDepart"] == dep) &
        (df_volsvalides["AeroArriv"]  == arr)
    ]
    if match_route.empty:
        departs_connus = df_volsvalides[
            df_volsvalides["NumVol"] == nv
        ]["AeroDepart"].unique()
        if dep not in departs_connus:
            raisons.append(
                f"AeroDepart '{dep}' inhabituel pour vol {nv} "
                f"(attendu : {', '.join(departs_connus)})"
            )

        arrivees_connues = df_volsvalides[
            df_volsvalides["NumVol"] == nv
        ]["AeroArriv"].unique()
        if arr not in arrivees_connues:
            raisons.append(
                f"AeroArriv '{arr}' inhabituelle pour vol {nv} "
                f"(attendu : {', '.join(arrivees_connues)})"
            )

        if not raisons:
            raisons.append(f"Route {dep}→{arr} jamais associée au vol {nv}")

    # ── 3. Matricule inhabituel pour cette route ──────────────────────────────
    if not match_route.empty:
        matricules_connus = match_route["Matricule"].unique()
        if mat not in matricules_connus:
            raisons.append(
                f"Matricule '{mat}' inhabituel sur route {dep}→{arr} vol {nv} "
                f"(attendu : {', '.join(matricules_connus)})"
            )

    # ── 4. Date inhabituelle (jour de semaine jamais opéré) ───────────────────
    if pd.notna(dat):
        jours_noms = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]
        jour = pd.to_datetime(dat).dayofweek
        match_jour = df_volsvalides[
            (df_volsvalides["NumVol"]     == nv) &
            (df_volsvalides["AeroDepart"] == dep) &
            (df_volsvalides["AeroArriv"]  == arr)
        ]
        if not match_jour.empty:
            jours_connus = set(
                pd.to_datetime(match_jour["Date"]).dt.dayofweek.unique()
            )
            if jour not in jours_connus:
                jours_connus_noms = [jours_noms[j] for j in sorted(jours_connus)]
                raisons.append(
                    f"Vol {nv} jamais opéré le {jours_noms[jour]} "
                    f"(jours habituels : {', '.join(jours_connus_noms)})"
                )

    # ── 5. Score extrêmement bas ──────────────────────────────────────────────
    if seuil_tres is not None and score < seuil_tres:
        raisons.append(f"Score IF très bas ({score:.4f}) — combinaison très atypique")

    if not raisons:
        freq = len(df_volsvalides[
            (df_volsvalides["NumVol"]     == nv) &
            (df_volsvalides["AeroDepart"] == dep) &
            (df_volsvalides["AeroArriv"]  == arr) &
            (df_volsvalides["Matricule"]  == mat)
        ])
        if freq > 0:
            raisons.append(
                f"Score IF légèrement anormal ({score:.4f}) — "
                f"combinaison connue ({freq} fois dans vols valides) — possible faux positif"
            )
        else:
            raisons.append(
                f"Combinaison (Vol:{nv} | {dep}→{arr} | Mat:{mat}) "
                f"absente des vols valides (score:{score:.4f})"
            )

    return " | ".join(raisons)


# ══════════════════════════════════════════════════════════════════════════════
# DETECTION DES VOLS MANQUANTS (rotation — tous vols)
# ══════════════════════════════════════════════════════════════════════════════

def detecter_vols_manquants(df, source, raw):
    """
    Tri : Matricule, Date, ETD (ordre réel des opérations).
    Rupture : AeroArriv vol N != AeroDepart vol N+1
    """
    manquants = []
    df = df.copy().reset_index(drop=True)

    df_raw = raw.loc[df["_raw_index"]].copy()
    etd_col = "ETD" if "ETD" in df_raw.columns else "bloc departure"
    df["_ETD"] = pd.to_datetime(df_raw[etd_col], format="mixed", errors="coerce").values

    df = df.sort_values(["Matricule", "Date", "_ETD"]).reset_index(drop=True)

    for i in range(len(df) - 1):
        vol_act  = df.iloc[i]
        vol_suiv = df.iloc[i + 1]

        if vol_act["Matricule"] != vol_suiv["Matricule"]:
            continue

        arriv_act   = vol_act["AeroArriv"]
        depart_suiv = vol_suiv["AeroDepart"]

        if arriv_act == depart_suiv:
            continue

        manquants.append({
            "Source":            source,
            "Matricule":         vol_act["Matricule"],
            "DateVolPrecedent":  vol_act["Date"],
            "DateVolSuivant":    vol_suiv["Date"],
            "AeroDepartEstime":  arriv_act,
            "AeroArrivEstime":   depart_suiv,
            "NumVolEstime":      "?",
            "NumVolPrecedent":   vol_act["NumVol"],
            "NumVolSuivant":     vol_suiv["NumVol"],
            "ETDPrecedent":      vol_act["_ETD"],
            "ETDSuivant":        vol_suiv["_ETD"],
            "RaisonDetection":   f"RotationRompue: attendu {arriv_act} trouvé {depart_suiv}",
        })

    df_manquants = pd.DataFrame(manquants)
    logger.info(f"  [{source}] {len(df_manquants)} vols manquants detectes.")
    return df_manquants


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run():

    model   = load_model()
    encoder = FlightFeatureEncoder.load()
    df_ref  = charger_ref_historique()

    # ── 1. Vols non matchés (filtrés côté SQL) ────────────────────────────────
    logger.info("─" * 50)
    logger.info("Chargement des vols non matches...")
    df_nm = charger_vols_non_matches()

    if df_nm.empty:
        logger.warning("Aucun vol non matche — rien a scorer.")
        return

    # ── 2. Renommage pour l'encodeur ──────────────────────────────────────────
    df_enc = df_nm.rename(columns={
        "MatriculeAIMS":  "Matricule",
        "NumVolAIMS":     "NumVol",
        "AeroDepartAIMS": "AeroDepart",
        "AeroArrivAIMS":  "AeroArriv",
        "DateAIMS":       "Date",
    }).copy()
    df_enc["Date"] = pd.to_datetime(df_enc["Date"], errors="coerce")
    df_enc = df_enc.dropna(
        subset=["Date", "Matricule", "NumVol", "AeroDepart", "AeroArriv"]
    )
    df_nm = df_nm.loc[df_enc.index]  # alignement

    # ── 3. Isolation Forest ───────────────────────────────────────────────────
    logger.info("─" * 50)
    logger.info("Scoring Isolation Forest sur vols non matches...")
    X           = encoder.transform(df_enc)
    scores      = model.score_samples(X)
    predictions = model.predict(X)
    statuts, seuil_tres = attribuer_statut(scores, predictions)

    # ── 4. Raisons anomalie ───────────────────────────────────────────────────
    raisons = []
    for i, (idx, row) in enumerate(df_enc.iterrows()):
        if statuts[i] == "Normal":
            raisons.append("Aucune")
        else:
            raisons.append(
                analyser_raison_anomalie(df_ref, row, scores[i], seuil_tres)
            )

    # ── 5. Construction résultats ─────────────────────────────────────────────
    df_result = df_nm.copy().reset_index(drop=True)
    df_result["ScoreAnomalie"]  = np.round(scores, 4)
    df_result["Statut"]         = statuts
    df_result["RaisonAnomalie"] = raisons

    anom = df_result[df_result["Statut"] != "Normal"].copy()

    anom_aims = anom[anom["source_table"] == "AIMS"].drop(columns=["source_table"])
    anom_mro  = anom[anom["source_table"] == "MRO"].drop(columns=["source_table"])

    n_aims_total = (df_result["source_table"] == "AIMS").sum()
    n_mro_total  = (df_result["source_table"] == "MRO").sum()

    logger.info(f"  AIMS non matches scorés : {n_aims_total}  →  suspects : {len(anom_aims)}")
    logger.info(f"  MRO  non matches scorés : {n_mro_total}   →  suspects : {len(anom_mro)}")

    # ── 6. Vols manquants (rotation sur TOUS les vols) ────────────────────────
    logger.info("─" * 50)
    logger.info("Analyse rotation (tous vols)...")

    raw_aims    = read_table("AIMS")
    df_aims_all = normalize_aims(raw_aims)
    df_aims_all["_raw_index"] = raw_aims.index
    df_aims_all = df_aims_all.dropna(
        subset=["Date", "Matricule", "NumVol", "AeroDepart", "AeroArriv"]
    )

    raw_mro    = read_table("MRO")
    df_mro_all = normalize_mro(raw_mro)
    df_mro_all["_raw_index"] = raw_mro.index
    df_mro_all = df_mro_all.dropna(
        subset=["Date", "Matricule", "NumVol", "AeroDepart", "AeroArriv"]
    )

    manquants_aims = detecter_vols_manquants(df_aims_all, "AIMS", raw_aims)
    manquants_mro  = detecter_vols_manquants(df_mro_all, "MRO",  raw_mro)

    # ── 7. Sauvegarde SQL Server ──────────────────────────────────────────────
    logger.info("─" * 50)
    logger.info("Sauvegarde dans SQL Server...")

    write_table(anom_aims, "AnomaliesAIMS")

    anom_mro = anom_mro.rename(columns={
        "IDAIMS":        "IDMRO",
        "MatriculeAIMS": "MatriculeMRO",
        "NumVolAIMS":    "NumVolMRO",
        "AeroDepartAIMS":"AeroDepartMRO",
        "AeroArrivAIMS": "AeroArrivMRO",
        "DateAIMS":      "DateMRO",
    })
    write_table(anom_mro, "AnomaliesMRO")

    vols_manquants = pd.concat([manquants_aims, manquants_mro], ignore_index=True)
    write_table(vols_manquants, "VolsManquantsIF")

    # ── 8. Résumé ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  RÉSULTATS DÉTECTION ANOMALIES")
    print("=" * 55)

    print(f"\n  AIMS ({n_aims_total} vols non matchés) :")
    print(f"    ⚠ Suspects       : {(anom_aims['Statut'] == 'Suspect').sum()}")
    print(f"    ✗ Très Suspects  : {(anom_aims['Statut'] == 'Très Suspect').sum()}")
    print(f"    ✈ Vols manquants : {len(manquants_aims)}")

    print(f"\n  MRO ({n_mro_total} vols non matchés) :")
    print(f"    ⚠ Suspects       : {(anom_mro['Statut'] == 'Suspect').sum()}")
    print(f"    ✗ Très Suspects  : {(anom_mro['Statut'] == 'Très Suspect').sum()}")
    print(f"    ✈ Vols manquants : {len(manquants_mro)}")

    print("\n  Tables créées dans SQL Server :")
    print(f"    → AnomaliesAIMS   ({len(anom_aims)} lignes)")
    print(f"    → AnomaliesMRO    ({len(anom_mro)} lignes)")
    print(f"    → VolsManquantsIF ({len(vols_manquants)} lignes)")
    print("=" * 55)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run()
