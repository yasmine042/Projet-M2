"""
predict.py — Détection d'anomalies sur les vols non matchés AIMS/MRO.
"""

import logging
import numpy as np
import pandas as pd

from config import MODEL_PATH, TABLE_VOLS_VALIDES
from db import read_table, read_query, write_table
from features import (
    FlightFeatureEncoder,
    normalize_aims, normalize_mro, normalize_vols_valides,
    get_fleet_family, same_fleet_family,
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
    raw = read_table(TABLE_VOLS_VALIDES)
    df  = normalize_vols_valides(raw)
    return df.dropna(subset=["Date", "Matricule", "NumVol", "AeroDepart", "AeroArriv"])


def charger_vols_non_matches():
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
            SELECT IDAIMS FROM {TABLE_VOLS_VALIDES} WHERE IDAIMS IS NOT NULL
        )
    """)

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
            SELECT IDMRO FROM {TABLE_VOLS_VALIDES} WHERE IDMRO IS NOT NULL
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

    Règle famille :
      Un matricule est considéré acceptable si un matricule de la MÊME famille
      a déjà opéré ce vol sur cette route. Seul un matricule d'une famille
      différente (ou famille inconnue = 0) déclenche le motif d'anomalie.
    """
    raisons = []

    nv       = row["NumVol"]
    dep      = row["AeroDepart"]
    arr      = row["AeroArriv"]
    mat      = row["Matricule"]
    dat      = row["Date"]
    famille  = get_fleet_family(mat)   # 0 si matricule inconnu

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

    # ── 3. Matricule hors famille pour cette route ────────────────────────────
    # Un matricule de la MÊME famille qu'un matricule historique est acceptable.
    # On ne signale une anomalie que si AUCUN matricule de la même famille
    # n'a jamais opéré ce vol sur cette route.
    if not match_route.empty:
        if famille == 0:
            # Matricule non référencé dans aucune famille connue → anomalie
            matricules_connus = match_route["Matricule"].unique()
            raisons.append(
                f"Matricule '{mat}' n'appartient à aucune famille connue "
                f"(matricules historiques sur route {dep}→{arr} vol {nv} : "
                f"{', '.join(matricules_connus)})"
            )
        else:
            # Vérifier si la famille est représentée dans l'historique de cette route
            familles_historiques = set(
                match_route["Matricule"].apply(get_fleet_family).unique()
            )
            familles_historiques.discard(0)  # ignorer les matricules non référencés

            if famille not in familles_historiques:
                # La famille du matricule suspect n'a jamais opéré cette route
                # → vrai changement de famille, anomalie légitime
                matricules_connus = match_route["Matricule"].unique()
                familles_connues_str = ", ".join(
                    f"famille {f}" for f in sorted(familles_historiques)
                )
                raisons.append(
                    f"Matricule '{mat}'  inhabituel sur route "
                    f"{dep}→{arr} vol {nv} — boeing historique : "
                   
                    f"(matricules : {', '.join(matricules_connus)})"
                )
            # else : même famille → pas d'anomalie sur ce critère

    # ── 4. Date inhabituelle (jour de semaine jamais opéré) ───────────────────
    if pd.notna(dat):
        jours_noms = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]
        jour = pd.to_datetime(dat).dayofweek
        if not match_route.empty:
            jours_connus = set(
                pd.to_datetime(match_route["Date"]).dt.dayofweek.unique()
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
        jour = pd.to_datetime(dat).dayofweek if pd.notna(dat) else None

        # Recherche dans l'historique en tenant compte de la famille
        if famille != 0:
            masque_famille = match_route["Matricule"].apply(
                lambda m: get_fleet_family(m) == famille
            )
            freq_famille_sans_jour = masque_famille.sum()
            freq_famille_avec_jour = (
                masque_famille & (
                    pd.to_datetime(match_route["Date"]).dt.dayofweek == jour
                )
            ).sum() if jour is not None else freq_famille_sans_jour
        else:
            freq_famille_sans_jour = 0
            freq_famille_avec_jour = 0

        if freq_famille_avec_jour > 0:
            raisons.append(
                f"Score IF légèrement anormal ({score:.4f}) — "
                f"combinaison connue ce jour ({freq_famille_avec_jour} fois, même famille) "
                f"— possible faux positif"
            )
        elif freq_famille_sans_jour > 0:
            jours_noms = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]
            raisons.append(
                f"Vol {nv} {dep}→{arr} famille {famille} connu ({freq_famille_sans_jour} fois) "
                f"mais jamais opéré le {jours_noms[jour]} (score:{score:.4f})"
            )
        else:
            raisons.append(
                f"Combinaison (Vol:{nv} | {dep}→{arr} | famille:{famille}) "
                f"absente des vols valides (score:{score:.4f})"
            )

    return " | ".join(raisons)


# ══════════════════════════════════════════════════════════════════════════════
# TYPE ANOMALIE (colonnes erronées)
# ══════════════════════════════════════════════════════════════════════════════

def determiner_type_anomalie(df_volsvalides, row):
    """
    Retourne les colonnes erronées sous forme compacte, ex: "NumVol | Matricule".
    Miroir de analyser_raison_anomalie, mais produit un label structuré.
    """
    types = []

    nv      = row["NumVol"]
    dep     = row["AeroDepart"]
    arr     = row["AeroArriv"]
    mat     = row["Matricule"]
    dat     = row["Date"]
    famille = get_fleet_family(mat)

    # 1. NumVol inconnu
    numvols_connus = set(df_volsvalides["NumVol"].unique())
    if nv not in numvols_connus:
        return "NumVol"

    # 2. Route inconnue → AeroDepart et/ou AeroArriv
    match_route = df_volsvalides[
        (df_volsvalides["NumVol"]     == nv) &
        (df_volsvalides["AeroDepart"] == dep) &
        (df_volsvalides["AeroArriv"]  == arr)
    ]
    if match_route.empty:
        departs_connus  = df_volsvalides[df_volsvalides["NumVol"] == nv]["AeroDepart"].unique()
        arrivees_connues = df_volsvalides[df_volsvalides["NumVol"] == nv]["AeroArriv"].unique()
        if dep not in departs_connus:
            types.append("AeroDepart")
        if arr not in arrivees_connues:
            types.append("AeroArriv")
        if not types:
            types.append("Route")

    # 3. Matricule hors famille
    if not match_route.empty:
        if famille == 0:
            types.append("Matricule")
        else:
            familles_historiques = set(match_route["Matricule"].apply(get_fleet_family).unique())
            familles_historiques.discard(0)
            if famille not in familles_historiques:
                types.append("Matricule")

    # 4. Jour de semaine inhabituel
    if pd.notna(dat) and not match_route.empty:
        jour = pd.to_datetime(dat).dayofweek
        jours_connus = set(pd.to_datetime(match_route["Date"]).dt.dayofweek.unique())
        if jour not in jours_connus:
            types.append("Date")

    # 5. Score IF très bas sans autre cause identifiée
    if not types:
        types.append("Score IF anormal")

    return " | ".join(types)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run():

    model   = load_model()
    encoder = FlightFeatureEncoder.load()
    df_ref  = charger_ref_historique()

    logger.info("─" * 50)
    logger.info("Chargement des vols non matches...")
    df_nm = charger_vols_non_matches()

    if df_nm.empty:
        logger.warning("Aucun vol non matche — rien a scorer.")
        return

    df_enc = df_nm.rename(columns={
        "MatriculeAIMS":  "Matricule",
        "NumVolAIMS":     "NumVol",
        "AeroDepartAIMS": "AeroDepart",
        "AeroArrivAIMS":  "AeroArriv",
        "DateAIMS":       "Date",
    }).copy()
    df_enc["Date"] = pd.to_datetime(df_enc["Date"], errors="coerce")

    def est_vide(s):
        return s.isna() | (s.astype(str).str.strip() == "")

    masque_incomplet = (
        df_enc["Date"].isna()            |
        est_vide(df_enc["Matricule"])    |
        est_vide(df_enc["NumVol"])       |
        est_vide(df_enc["AeroDepart"])   |
        est_vide(df_enc["AeroArriv"])
    )

    idx_incomplets = df_enc[masque_incomplet].index
    idx_complets   = df_enc[~masque_incomplet].index

    champs_verif = [
        ("MatriculeAIMS",  "Matricule"),
        ("NumVolAIMS",     "NumVol"),
        ("AeroDepartAIMS", "AéroDepart"),
        ("AeroArrivAIMS",  "AéroArriv"),
        ("DateAIMS",       "Date"),
    ]
    df_incomplets = df_nm.loc[idx_incomplets].copy()
    df_incomplets["ScoreAnomalie"] = -1.0
    df_incomplets["Statut"]        = "Suspect"
    df_incomplets["RaisonAnomalie"] = df_incomplets.apply(
        lambda r: "Données incomplètes : " + ", ".join([
            label + " vide"
            for orig, label in champs_verif
            if pd.isna(r.get(orig)) or str(r.get(orig, "")).strip() == ""
        ]),
        axis=1,
    )
    df_incomplets["TypeAnomalie"] = df_incomplets.apply(
        lambda r: " | ".join([
            orig
            for orig, _ in champs_verif
            if pd.isna(r.get(orig)) or str(r.get(orig, "")).strip() == ""
        ]),
        axis=1,
    )
    n_incomplets = len(df_incomplets)
    if n_incomplets:
        logger.warning(
            f"  {n_incomplets} vols avec champs vides "
            f"→ flaggés Suspect directement (non scorés par IF)."
        )

    df_enc = df_enc.loc[idx_complets].reset_index(drop=True)
    df_nm  = df_nm.loc[idx_complets].reset_index(drop=True)

    logger.info("─" * 50)
    logger.info("Scoring Isolation Forest sur vols non matches...")
    X      = encoder.transform(df_enc)
    scores = model.score_samples(X)

    # Score weighting sur FreqComboFamille (index 8 — dernière colonne)
    freq_combo_col = X[:, 8]
    novel_mask = freq_combo_col < 0
    if novel_mask.any():
        AMPLIFICATION = 1.25
        scores = scores.copy()
        scores[novel_mask] *= AMPLIFICATION
        logger.info(
            f"  Score weighting x{AMPLIFICATION} : {novel_mask.sum()} vols "
            f"a combinaison inconnue (FreqComboFamille=-1.0)."
        )

    predictions = np.where(scores >= model.offset_, 1, -1)
    statuts, seuil_tres = attribuer_statut(scores, predictions)

    raisons = []
    types   = []
    for i, (_, row) in enumerate(df_enc.iterrows()):
        if statuts[i] == "Normal":
            raisons.append("Aucune")
            types.append("Aucune")
        else:
            raisons.append(analyser_raison_anomalie(df_ref, row, scores[i], seuil_tres))
            types.append(determiner_type_anomalie(df_ref, row))

    df_result = df_nm.copy().reset_index(drop=True)
    df_result["ScoreAnomalie"]  = np.round(scores, 4)
    df_result["Statut"]         = statuts
    df_result["RaisonAnomalie"] = raisons
    df_result["TypeAnomalie"]   = types

    anom    = df_result[df_result["Statut"] != "Normal"].copy()
    normaux = df_result[df_result["Statut"] == "Normal"].copy()

    if not df_incomplets.empty:
        anom = pd.concat([anom, df_incomplets], ignore_index=True)

    anom_aims = anom[anom["source_table"] == "AIMS"].drop(columns=["source_table"])
    anom_mro  = anom[anom["source_table"] == "MRO"].drop(columns=["source_table"])
    norm_aims = normaux[normaux["source_table"] == "AIMS"].drop(columns=["source_table"])
    norm_mro  = normaux[normaux["source_table"] == "MRO"].drop(columns=["source_table"])

    n_aims_total = (df_result["source_table"] == "AIMS").sum() + \
                   (df_incomplets["source_table"] == "AIMS").sum()
    n_mro_total  = (df_result["source_table"] == "MRO").sum() + \
                   (df_incomplets["source_table"] == "MRO").sum()

    logger.info(f"  AIMS non matches scorés : {n_aims_total}  →  suspects : {len(anom_aims)}")
    logger.info(f"  MRO  non matches scorés : {n_mro_total}   →  suspects : {len(anom_mro)}")

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
    write_table(norm_aims, "VoisNormalesAIMS")

    norm_mro = norm_mro.rename(columns={
        "IDAIMS":        "IDMRO",
        "MatriculeAIMS": "MatriculeMRO",
        "NumVolAIMS":    "NumVolMRO",
        "AeroDepartAIMS":"AeroDepartMRO",
        "AeroArrivAIMS": "AeroArrivMRO",
        "DateAIMS":      "DateMRO",
    })
    write_table(norm_mro, "VoisNormalesMRO")

    print("\n" + "=" * 55)
    print("  RESULTATS DETECTION ANOMALIES")
    print("=" * 55)

    n_incomp_aims = (df_incomplets["source_table"] == "AIMS").sum()
    n_incomp_mro  = (df_incomplets["source_table"] == "MRO").sum()

    print(f"\n  AIMS ({n_aims_total} vols non matches) :")
    print(f"    [OK] Normaux (IF)         : {len(norm_aims)}")
    print(f"    [!]  Suspects (IF)        : {(anom_aims['Statut'] == 'Suspect').sum() - n_incomp_aims}")
    print(f"    [X]  Tres Suspects (IF)   : {(anom_aims['Statut'] == 'Tres Suspect').sum()}")
    print(f"    [X]  Donnees incompletes  : {n_incomp_aims}")

    print(f"\n  MRO ({n_mro_total} vols non matches) :")
    print(f"    [OK] Normaux (IF)         : {len(norm_mro)}")
    print(f"    [!]  Suspects (IF)        : {(anom_mro['Statut'] == 'Suspect').sum() - n_incomp_mro}")
    print(f"    [X]  Tres Suspects (IF)   : {(anom_mro['Statut'] == 'Tres Suspect').sum()}")
    print(f"    [X]  Donnees incompletes  : {n_incomp_mro}")

    print("\n  Tables creees dans SQL Server :")
    print(f"    -> AnomaliesAIMS     ({len(anom_aims)} lignes)")
    print(f"    -> AnomaliesMRO      ({len(anom_mro)} lignes)")
    print(f"    -> VoisNormalesAIMS  ({len(norm_aims)} lignes)")
    print(f"    -> VoisNormalesMRO   ({len(norm_mro)} lignes)")
    print("=" * 55)


if __name__ == "__main__":
    run()