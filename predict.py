"""
predict.py — Détection d'anomalies sur les vols non matchés AIMS/MRO.
"""

import logging
import numpy as np
import pandas as pd

from config import MODEL_PATH, RF_MODEL_PATH, TABLE_VOLS_VALIDES
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


def _fmt(val):
    """Représentation lisible d'une valeur pour les messages — '(vide)' si manquante."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "(vide)"
    s = str(val).strip()
    return s if s and s.lower() != "nan" else "(vide)"


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
        raisons.append(f"NumVol {_fmt(nv)} jamais vu dans vols valides")
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
                f"AeroDepart '{_fmt(dep)}' inhabituel pour vol {nv} "
                f"(attendu : {', '.join(departs_connus)})"
            )

        arrivees_connues = df_volsvalides[
            df_volsvalides["NumVol"] == nv
        ]["AeroArriv"].unique()
        if arr not in arrivees_connues:
            raisons.append(
                f"AeroArriv '{_fmt(arr)}' inhabituelle pour vol {nv} "
                f"(attendu : {', '.join(arrivees_connues)})"
            )

        if not raisons:
            raisons.append(f"Route {_fmt(dep)}→{_fmt(arr)} jamais associée au vol {nv}")

    # ── 3. Matricule hors famille pour cette route ────────────────────────────
    # Un matricule de la MÊME famille qu'un matricule historique est acceptable.
    # On ne signale une anomalie que si AUCUN matricule de la même famille
    # n'a jamais opéré ce vol sur cette route.
    if not match_route.empty:
        if famille == 0:
            # Matricule non référencé dans aucune famille connue → anomalie
            matricules_connus = match_route["Matricule"].unique()
            raisons.append(
                f"Matricule '{_fmt(mat)}' n'appartient à aucune famille connue "
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
                    f"Matricule '{_fmt(mat)}'  inhabituel sur route "
                    f"{dep}→{arr} vol {nv} — boeing historique : "

                    f"(matricules : {', '.join(matricules_connus)})"
                )
            # else : même famille → pas d'anomalie sur ce critère

    # ── 4. Date manquante ou inhabituelle (jour de semaine jamais opéré) ──────
    if pd.isna(dat):
        raisons.append("Date manquante")
    else:
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

    # 4. Date manquante ou jour de semaine inhabituel
    if pd.isna(dat):
        types.append("Date")
    elif not match_route.empty:
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

    logger.info("─" * 50)
    logger.info("Scoring Isolation Forest sur vols non matches...")
    X      = encoder.transform(df_enc)
    scores = model.score_samples(X)

    predictions = np.where(scores >= model.offset_, 1, -1)
    statuts, seuil_tres = attribuer_statut(scores, predictions)

    raisons = []
    types   = []
    n_reclasses = 0
    for i, (_, row) in enumerate(df_enc.iterrows()):
        if statuts[i] == "Normal":
            raisons.append("Aucune")
            types.append("Aucune")
            continue

        type_anomalie = determiner_type_anomalie(df_ref, row)
        if type_anomalie == "Score IF anormal":
            # NumVol, route, famille du matricule et jour sont tous cohérents
            # avec l'historique : pas de raison métier d'anomalie -> Normal.
            statuts[i] = "Normal"
            raisons.append("Aucune")
            types.append("Aucune")
            n_reclasses += 1
        else:
            raisons.append(analyser_raison_anomalie(df_ref, row, scores[i], seuil_tres))
            types.append(type_anomalie)

    if n_reclasses:
        logger.info(
            f"  {n_reclasses} vol(s) reclassés Normal "
            f"(score IF bas mais NumVol/Route/Famille/Date cohérents)."
        )

    df_result = df_nm.copy().reset_index(drop=True)
    df_result["ScoreAnomalie"]  = np.round(scores, 4)
    df_result["Statut"]         = statuts
    df_result["RaisonAnomalie"] = raisons
    df_result["TypeAnomalie"]   = types

    anom    = df_result[df_result["Statut"] != "Normal"].copy()
    normaux = df_result[df_result["Statut"] == "Normal"].copy()

    anom_aims = anom[anom["source_table"] == "AIMS"].drop(columns=["source_table"])
    anom_mro  = anom[anom["source_table"] == "MRO"].drop(columns=["source_table"])
    norm_aims = normaux[normaux["source_table"] == "AIMS"].drop(columns=["source_table"])
    norm_mro  = normaux[normaux["source_table"] == "MRO"].drop(columns=["source_table"])

    n_aims_total = (df_result["source_table"] == "AIMS").sum()
    n_mro_total  = (df_result["source_table"] == "MRO").sum()

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

    print(f"\n  AIMS ({n_aims_total} vols non matches) :")
    print(f"    [OK] Normaux (IF)         : {len(norm_aims)}")
    print(f"    [!]  Suspects (IF)        : {(anom_aims['Statut'] == 'Suspect').sum()}")
    print(f"    [X]  Tres Suspects (IF)   : {(anom_aims['Statut'] == 'Tres Suspect').sum()}")

    print(f"\n  MRO ({n_mro_total} vols non matches) :")
    print(f"    [OK] Normaux (IF)         : {len(norm_mro)}")
    print(f"    [!]  Suspects (IF)        : {(anom_mro['Statut'] == 'Suspect').sum()}")
    print(f"    [X]  Tres Suspects (IF)   : {(anom_mro['Statut'] == 'Tres Suspect').sum()}")

    print("\n  Tables creees dans SQL Server :")
    print(f"    -> AnomaliesAIMS     ({len(anom_aims)} lignes)")
    print(f"    -> AnomaliesMRO      ({len(anom_mro)} lignes)")
    print(f"    -> VoisNormalesAIMS  ({len(norm_aims)} lignes)")
    print(f"    -> VoisNormalesMRO   ({len(norm_mro)} lignes)")
    print("=" * 55)


# ══════════════════════════════════════════════════════════════════════════════
# SCAN RF — détecte les faux négatifs dans VoisNormalesAIMS / VoisNormalesMRO
# ══════════════════════════════════════════════════════════════════════════════

def run_rf_scan():
    """
    Applique le Random Forest sur les vols que l'IF a classés normaux.
    Les vols que le RF reclasse en anomalie sont les faux négatifs de l'IF.
    Résultat sauvegardé dans FauxNegatifsAIMS et FauxNegatifsMRO,
    avec RaisonAnomalie et TypeAnomalie calculés comme pour les anomalies IF.
    """
    logger.info("─" * 50)
    logger.info("Chargement du modèle Random Forest...")
    try:
        with open(RF_MODEL_PATH, "rb") as f:
            rf = pickle.load(f)
    except FileNotFoundError:
        logger.error(f"Modèle RF introuvable : {RF_MODEL_PATH}. Lancez train_rf.py d'abord.")
        return

    encoder = FlightFeatureEncoder.load()
    df_ref  = charger_ref_historique()   # VolsValides — nécessaire pour analyser_raison_anomalie

    specs = [
        ("VoisNormalesAIMS", "MatriculeAIMS", "NumVolAIMS",
         "AeroDepartAIMS", "AeroArrivAIMS", "DateAIMS", "FauxNegatifsAIMS"),
        ("VoisNormalesMRO",  "MatriculeMRO",  "NumVolMRO",
         "AeroDepartMRO",  "AeroArrivMRO",  "DateMRO",  "FauxNegatifsMRO"),
    ]

    total_fn = 0
    for table_src, mat_c, nv_c, dep_c, arr_c, dat_c, table_dst in specs:

        logger.info(f"  Scan RF sur {table_src}...")
        try:
            df_norm = read_table(table_src)
        except Exception:
            logger.warning(f"  {table_src} introuvable — ignoré.")
            continue

        if df_norm.empty:
            logger.info(f"  {table_src} vide — rien à scanner.")
            continue

        # Renommer vers format commun pour l'encodeur et l'analyse
        df_enc = df_norm.rename(columns={
            mat_c: "Matricule", nv_c: "NumVol",
            dep_c: "AeroDepart", arr_c: "AeroArriv", dat_c: "Date",
        }).copy()
        df_enc["Date"] = pd.to_datetime(df_enc["Date"], errors="coerce")

        masque_ok = (
            df_enc["Date"].notna() &
            df_enc["Matricule"].astype(str).str.strip().ne("") &
            df_enc["NumVol"].astype(str).str.strip().ne("") &
            df_enc["AeroDepart"].astype(str).str.strip().ne("") &
            df_enc["AeroArriv"].astype(str).str.strip().ne("")
        )
        df_enc  = df_enc[masque_ok].reset_index(drop=True)
        df_norm = df_norm[masque_ok].reset_index(drop=True)

        if df_enc.empty:
            continue

        X      = encoder.transform(df_enc)
        X_rf   = X[:, :8]                    # FreqComboFamille exclue — même que l'entraînement
        preds  = rf.predict(X_rf)
        probas = rf.predict_proba(X_rf)[:, 1]   # probabilité d'être anomalie (0→1)

        fn_mask   = preds == 1
        n_fn      = fn_mask.sum()
        total_fn += n_fn
        logger.info(f"  {table_src} : {len(df_norm)} vols normaux (IF) → {n_fn} faux négatifs (RF)")

        df_fn     = df_norm[fn_mask].copy().reset_index(drop=True)
        df_enc_fn = df_enc[fn_mask].copy().reset_index(drop=True)
        scores_fn = probas[fn_mask]

        # Analyser la raison métier pour chaque faux négatif (même logique que IF)
        raisons, types = [], []
        for i, (_, row) in enumerate(df_enc_fn.iterrows()):
            raisons.append(analyser_raison_anomalie(df_ref, row, scores_fn[i], seuil_tres=None))
            types.append(determiner_type_anomalie(df_ref, row))

        df_fn["ScoreAnomalie"]  = np.round(scores_fn, 4)
        df_fn["Statut"]         = "Faux Négatif IF"
        df_fn["RaisonAnomalie"] = raisons
        df_fn["TypeAnomalie"]   = types

        write_table(df_fn, table_dst)
        logger.info(f"  → {table_dst} sauvegardé ({len(df_fn)} lignes)")

    print("\n" + "=" * 55)
    print("  RÉSULTATS SCAN RF (faux négatifs IF)")
    print("=" * 55)
    print(f"  Total faux négatifs détectés : {total_fn}")
    print(f"    -> FauxNegatifsAIMS")
    print(f"    -> FauxNegatifsMRO")
    print("=" * 55)


if __name__ == "__main__":
    run()