"""
predict.py — Détection d'anomalies + vols manquants.

Tables produites :
  - AnomaliesAIMS   : photocopie AIMS + ScoreAnomalie + Statut + RaisonAnomalie
  - AnomaliesMRO    : photocopie MRO  + ScoreAnomalie + Statut + RaisonAnomalie
  - VolsManquantsIF : vols manquants détectés par analyse de rotation
"""

import pickle
import logging
import pandas as pd
import numpy as np

from config import MODEL_PATH
from db import read_table, write_table
from features import FlightFeatureEncoder, normalize_aims, normalize_mro

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
# ATTRIBUTION DES STATUTS
# Basé uniquement sur les anomalies détectées par Isolation Forest
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

    # Scores uniquement des anomalies
    scores_anormaux = scores[predictions == -1]

    # Sécurité si aucune anomalie détectée
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

    # ── 1. NumVol jamais vu dans VolsValides ──────────────────────────────
    numvols_connus = set(df_volsvalides["NumVol"].unique())
    if nv not in numvols_connus:
        raisons.append(f"NumVol {nv} jamais vu dans vols valides")
        return " | ".join(raisons)  # inutile de continuer

    # ── 2. Route (NumVol + Depart + Arrivee) jamais vue ───────────────────
    match_route = df_volsvalides[
        (df_volsvalides["NumVol"]     == nv) &
        (df_volsvalides["AeroDepart"] == dep) &
        (df_volsvalides["AeroArriv"]  == arr)
    ]
    if match_route.empty:

        # ── 2a. Départ inhabituel pour ce NumVol ──────────────────────────
        departs_connus = df_volsvalides[
            df_volsvalides["NumVol"] == nv
        ]["AeroDepart"].unique()

        if dep not in departs_connus:
            raisons.append(
                f"AeroDepart '{dep}' inhabituel pour vol {nv} "
                f"(attendu : {', '.join(departs_connus)})"
            )

        # ── 2b. Arrivée inhabituelle pour ce NumVol ───────────────────────
        arrivees_connues = df_volsvalides[
            df_volsvalides["NumVol"] == nv
        ]["AeroArriv"].unique()

        if arr not in arrivees_connues:
            raisons.append(
                f"AeroArriv '{arr}' inhabituelle pour vol {nv} "
                f"(attendu : {', '.join(arrivees_connues)})"
            )

        if not raisons:
            raisons.append(
                f"Route {dep}→{arr} jamais associée au vol {nv}"
            )

    # ── 3. Matricule inhabituel pour cette route ───────────────────────────
    if not match_route.empty:
        matricules_connus = match_route["Matricule"].unique()
        if mat not in matricules_connus:
            raisons.append(
                f"Matricule '{mat}' inhabituel sur route {dep}→{arr} vol {nv} "
                f"(attendu : {', '.join(matricules_connus)})"
            )

    # ── 4. Date inhabituelle (jour de semaine jamais opéré) ───────────────
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

    # ── 5. Score extrêmement bas ───────────────────────────────────────────
    if seuil_tres is not None and score < seuil_tres:
        raisons.append(f"Score IF très bas ({score:.4f}) — combinaison très atypique")

    # ── Fallback uniquement si vraiment rien trouvé ───────────────────────
    if not raisons:
        raisons.append(
            f"Combinaison (Vol:{nv} | {dep}→{arr} | Mat:{mat}) "
            f"statistiquement rare dans les vols valides (score:{score:.4f})"
        )

    return " | ".join(raisons)


# ══════════════════════════════════════════════════════════════════════════════
# DETECTION DES VOLS MANQUANTS
# ══════════════════════════════════════════════════════════════════════════════

def detecter_vols_manquants(df, source, raw):
    """
    Tri : Matricule, Date, ETD (ordre réel des opérations).

    Rupture :
        AeroArriv vol N != AeroDepart vol N+1

    Exemple :
        Vol 1 : ALG → ORN
        Vol 2 : CZL → ALG

        => il manque probablement un vol ORN → CZL

    NumVol estimé :
        impossible à prédire → '?'
    """

    manquants = []

    df = df.copy().reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Récupération ETD depuis la table brute
    # ──────────────────────────────────────────────────────────────────────────

    df_raw = raw.loc[df["_raw_index"]].copy()

    if "ETD" in df_raw.columns:
        etd_col = "ETD"
    else:
        etd_col = "bloc departure"

    # Conversion datetime sécurisée
    df["_ETD"] = pd.to_datetime(
        df_raw[etd_col],
        errors="coerce"
    ).values

    # ──────────────────────────────────────────────────────────────────────────
    # Tri chronologique réel
    # ──────────────────────────────────────────────────────────────────────────

    df = df.sort_values(
        ["Matricule", "Date", "_ETD"]
    ).reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Analyse rotation
    # ──────────────────────────────────────────────────────────────────────────

    for i in range(len(df) - 1):

        vol_act = df.iloc[i]
        vol_suiv = df.iloc[i + 1]

        # Même avion uniquement
        if vol_act["Matricule"] != vol_suiv["Matricule"]:
            continue

        arriv_act = vol_act["AeroArriv"]
        depart_suiv = vol_suiv["AeroDepart"]

        # Rotation correcte
        if arriv_act == depart_suiv:
            continue

        # Vol manquant détecté
        manquants.append({
            "Source": source,

            "Matricule": vol_act["Matricule"],

            "DateVolPrecedent": vol_act["Date"],
            "DateVolSuivant": vol_suiv["Date"],

            "AeroDepartEstime": arriv_act,
            "AeroArrivEstime": depart_suiv,

            "NumVolEstime": "?",

            "NumVolPrecedent": vol_act["NumVol"],
            "NumVolSuivant": vol_suiv["NumVol"],

            "ETDPrecedent": vol_act["_ETD"],
            "ETDSuivant": vol_suiv["_ETD"],

            "RaisonDetection":
                f"RotationRompue: attendu {arriv_act} trouvé {depart_suiv}",
        })

    df_manquants = pd.DataFrame(manquants)

    logger.info(
        f"  [{source}] {len(df_manquants)} vols manquants detectes."
    )

    return df_manquants


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run():

    model = load_model()
    encoder = FlightFeatureEncoder.load()

    # ══════════════════════════════════════════════════════════════════════════
    # AIMS
    # ══════════════════════════════════════════════════════════════════════════

    logger.info("─" * 50)
    logger.info("Traitement AIMS...")

    raw_aims = read_table("AIMS")

    df_aims = normalize_aims(raw_aims)

    # Sauvegarde index original SQL
    df_aims["_raw_index"] = raw_aims.index

    # Suppression lignes inexploitables
    df_aims = df_aims.dropna(
        subset=[
            "Date",
            "Matricule",
            "NumVol",
            "AeroDepart",
            "AeroArriv"
        ]
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Encodage
    # ──────────────────────────────────────────────────────────────────────────

    X_aims = encoder.transform(df_aims)

    # ──────────────────────────────────────────────────────────────────────────
    # Isolation Forest
    # ──────────────────────────────────────────────────────────────────────────

    scores_aims = model.score_samples(X_aims)
    predict_aims = model.predict(X_aims)

    statuts_aims, seuil_tres_aims = attribuer_statut(
        scores_aims,
        predict_aims
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Construction résultat final
    # ──────────────────────────────────────────────────────────────────────────

    result_aims = raw_aims.loc[df_aims["_raw_index"]].copy()

    result_aims["ScoreAnomalie"] = scores_aims.round(4)

    result_aims["Statut"] = statuts_aims

    raisons_aims = []

    for idx, row in df_aims.iterrows():

        statut = statuts_aims[df_aims.index.get_loc(idx)]

        if statut == "Normal":

            raisons_aims.append("Aucune")

        else:

            raison = analyser_raison_anomalie(
                df_aims,
                row,
                scores_aims[df_aims.index.get_loc(idx)],
                seuil_tres_aims
            )

            raisons_aims.append(raison)

    result_aims["RaisonAnomalie"] = raisons_aims

    # ──────────────────────────────────────────────────────────────────────────
    # Séparation
    # ──────────────────────────────────────────────────────────────────────────

    anom_aims = result_aims[
        result_aims["Statut"] != "Normal"
    ].copy()

    norm_aims = result_aims[
        result_aims["Statut"] == "Normal"
    ].copy()

    logger.info(f"  AIMS total      : {len(result_aims)}")
    logger.info(f"  AIMS normaux    : {len(norm_aims)}")
    logger.info(
        f"  AIMS suspects   : "
        f"{(result_aims['Statut'] == 'Suspect').sum()}"
    )
    logger.info(
        f"  AIMS très susp. : "
        f"{(result_aims['Statut'] == 'Très Suspect').sum()}"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Détection vols manquants
    # ──────────────────────────────────────────────────────────────────────────

    manquants_aims = detecter_vols_manquants(
        df_aims,
        "AIMS",
        raw_aims
    )

    # ══════════════════════════════════════════════════════════════════════════
    # MRO
    # ══════════════════════════════════════════════════════════════════════════

    logger.info("─" * 50)
    logger.info("Traitement MRO...")

    raw_mro = read_table("MRO")

    df_mro = normalize_mro(raw_mro)

    # Sauvegarde index original SQL
    df_mro["_raw_index"] = raw_mro.index

    # Suppression lignes inexploitables
    df_mro = df_mro.dropna(
        subset=[
            "Date",
            "Matricule",
            "NumVol",
            "AeroDepart",
            "AeroArriv"
        ]
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Encodage
    # ──────────────────────────────────────────────────────────────────────────

    X_mro = encoder.transform(df_mro)

    # ──────────────────────────────────────────────────────────────────────────
    # Isolation Forest
    # ──────────────────────────────────────────────────────────────────────────

    scores_mro = model.score_samples(X_mro)
    predict_mro = model.predict(X_mro)

    statuts_mro, seuil_tres_mro = attribuer_statut(
        scores_mro,
        predict_mro
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Construction résultat final
    # ──────────────────────────────────────────────────────────────────────────

    result_mro = raw_mro.loc[df_mro["_raw_index"]].copy()

    result_mro["ScoreAnomalie"] = scores_mro.round(4)

    result_mro["Statut"] = statuts_mro

    raisons_mro = []

    for idx, row in df_mro.iterrows():

        statut = statuts_mro[df_mro.index.get_loc(idx)]

        if statut == "Normal":

            raisons_mro.append("Aucune")

        else:

            raison = analyser_raison_anomalie(
                df_mro,
                row,
                scores_mro[df_mro.index.get_loc(idx)],
                seuil_tres_mro
            )

            raisons_mro.append(raison)

    result_mro["RaisonAnomalie"] = raisons_mro

    # ──────────────────────────────────────────────────────────────────────────
    # Séparation
    # ──────────────────────────────────────────────────────────────────────────

    anom_mro = result_mro[
        result_mro["Statut"] != "Normal"
    ].copy()

    norm_mro = result_mro[
        result_mro["Statut"] == "Normal"
    ].copy()

    logger.info(f"  MRO total      : {len(result_mro)}")
    logger.info(f"  MRO normaux    : {len(norm_mro)}")
    logger.info(
        f"  MRO suspects   : "
        f"{(result_mro['Statut'] == 'Suspect').sum()}"
    )
    logger.info(
        f"  MRO très susp. : "
        f"{(result_mro['Statut'] == 'Très Suspect').sum()}"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Détection vols manquants
    # ──────────────────────────────────────────────────────────────────────────

    manquants_mro = detecter_vols_manquants(
        df_mro,
        "MRO",
        raw_mro
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SAUVEGARDE SQL SERVER
    # ══════════════════════════════════════════════════════════════════════════

    logger.info("─" * 50)
    logger.info("Sauvegarde dans SQL Server...")

    write_table(anom_aims, "AnomaliesAIMS")

    write_table(anom_mro, "AnomaliesMRO")

    # Union vols manquants
    vols_manquants = pd.concat(
        [manquants_aims, manquants_mro],
        ignore_index=True
    )

    write_table(
        vols_manquants,
        "VolsManquantsIF"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # RÉSUMÉ FINAL
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 55)
    print("  RÉSULTATS DÉTECTION ANOMALIES")
    print("=" * 55)

    print(f"\n  AIMS ({len(raw_aims)} vols total) :")
    print(f"    ✓ Normaux        : {len(norm_aims)}")
    print(
        f"    ⚠ Suspects       : "
        f"{(result_aims['Statut'] == 'Suspect').sum()}"
    )
    print(
        f"    ✗ Très Suspects  : "
        f"{(result_aims['Statut'] == 'Très Suspect').sum()}"
    )
    print(f"    ✈ Vols manquants : {len(manquants_aims)}")

    print(f"\n  MRO ({len(raw_mro)} vols total) :")
    print(f"    ✓ Normaux        : {len(norm_mro)}")
    print(
        f"    ⚠ Suspects       : "
        f"{(result_mro['Statut'] == 'Suspect').sum()}"
    )
    print(
        f"    ✗ Très Suspects  : "
        f"{(result_mro['Statut'] == 'Très Suspect').sum()}"
    )
    print(f"    ✈ Vols manquants : {len(manquants_mro)}")

    print("\n  Tables créées dans SQL Server :")
    print(f"    → AnomaliesAIMS    ({len(anom_aims)} lignes)")
    print(f"    → AnomaliesMRO     ({len(anom_mro)} lignes)")
    print(f"    → VolsManquantsIF ({len(vols_manquants)} lignes)")

    print("=" * 55)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run()