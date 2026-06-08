"""
predict.py — Détection d'anomalies sur les vols non matchés AIMS/MRO.

L'Isolation Forest est appliqué uniquement aux vols qui n'ont pas trouvé
de correspondance dans VolsValidesEtape1 (via IDAIMS / IDMRO).

Logique famille d'appareils :
    - Route connue avec matricule de la MÊME famille  → pas d'anomalie matricule
    - Route connue avec matricule d'une AUTRE famille → anomalie (matricule trouvé / attendu)

Colonne TypeAnomalie (valeur courte) :
    NUMVOL | AERO_DEPART | AERO_ARRIV | ROUTE | MATRICULE | FAMILLE_APPAREIL
    DATE | SCORE | MULTIPLE | FAUX_POSITIF | INCONNU

Tables produites :
  - AnomaliesAIMS            : vols AIMS non matchés suspects
  - AnomaliesMRO             : vols MRO  non matchés suspects
  - AnomaliesFamilleAppareil : vols matchés ETL avec famille d'appareil croisée
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

def _matricule_anomalie(df_ref, nv, dep, arr, mat):
    """
    Vérifie si le matricule `mat` est cohérent avec la route (nv, dep, arr)
    en tenant compte des familles d'appareils.

    Règles :
        - Exact match                        → OK
        - Même famille qu'un connu           → OK (même type d'appareil)
        - Famille différente sur route connue → anomalie
            → message : matricule trouvé / matricule(s) attendu(s)

    Retourne (bool anomalie, str explication | None).
    """
    match_route = df_ref[
        (df_ref["NumVol"]     == nv) &
        (df_ref["AeroDepart"] == dep) &
        (df_ref["AeroArriv"]  == arr)
    ]

    if match_route.empty:
        return False, None  # anomalie route, gérée ailleurs

    matricules_connus = match_route["Matricule"].unique()
    famille_mat = get_fleet_family(mat)

    # Exact match → OK
    if mat in matricules_connus:
        return False, None

    # Même famille qu'au moins un connu → OK
    if famille_mat != 0:
        for m_connu in matricules_connus:
            if same_fleet_family(mat, m_connu):
                return False, None

    # Famille différente ou inconnue sur route connue → anomalie
    familles_connues = set(
        get_fleet_family(m) for m in matricules_connus if get_fleet_family(m) != 0
    )
    famille_mat_str = f"Boeing {famille_mat}" if famille_mat != 0 else "type inconnu"
    familles_str    = ", ".join(f"Boeing {f}" for f in sorted(familles_connues)) \
                      if familles_connues else "type(s) inconnu(s)"

    explication = (
        f"Matricule trouvé : '{mat}' ({famille_mat_str}) | "
        f"Matricule(s) attendu(s) : {', '.join(matricules_connus)} ({familles_str}) "
        f"sur route {dep}→{arr} vol {nv}"
    )
    return True, explication


def analyser_raison_anomalie(df_volsvalides, row, score, seuil_tres):
    """
    Compare le vol suspect avec VolsValides pour expliquer l'anomalie.
    Retourne (raison: str, type_anomalie: str).
    """
    raisons = []
    types   = []

    nv  = row["NumVol"]
    dep = row["AeroDepart"]
    arr = row["AeroArriv"]
    mat = row["Matricule"]
    dat = row["Date"]

    # ── 1. NumVol jamais vu ───────────────────────────────────────────────────
    numvols_connus = set(df_volsvalides["NumVol"].unique())
    if nv not in numvols_connus:
        raisons.append(f"NumVol {nv} jamais vu dans les vols valides")
        return " | ".join(raisons), "NUMVOL"

    # ── 2. Route jamais vue ───────────────────────────────────────────────────
    match_route = df_volsvalides[
        (df_volsvalides["NumVol"]     == nv) &
        (df_volsvalides["AeroDepart"] == dep) &
        (df_volsvalides["AeroArriv"]  == arr)
    ]
    if match_route.empty:
        departs_connus  = df_volsvalides[df_volsvalides["NumVol"] == nv]["AeroDepart"].unique()
        arrivees_connues = df_volsvalides[df_volsvalides["NumVol"] == nv]["AeroArriv"].unique()

        if dep not in departs_connus:
            raisons.append(
                f"AeroDepart trouvé : '{dep}' | "
                f"AeroDepart(s) attendu(s) : {', '.join(departs_connus)} "
                f"pour vol {nv}"
            )
            types.append("AERO_DEPART")

        if arr not in arrivees_connues:
            raisons.append(
                f"AeroArrivée trouvée : '{arr}' | "
                f"AeroArrivée(s) attendue(s) : {', '.join(arrivees_connues)} "
                f"pour vol {nv}"
            )
            types.append("AERO_ARRIV")

        if not raisons:
            raisons.append(f"Route {dep}→{arr} jamais associée au vol {nv}")
            types.append("ROUTE")

    # ── 3. Matricule / famille d'appareils ────────────────────────────────────
    anom_mat, expl_mat = _matricule_anomalie(df_volsvalides, nv, dep, arr, mat)
    if anom_mat:
        raisons.append(expl_mat)
        types.append("FAMILLE_APPAREIL")

    # ── 4. Date inhabituelle (jour jamais opéré) ──────────────────────────────
    if pd.notna(dat):
        jours_noms = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]
        jour = pd.to_datetime(dat).dayofweek
        match_jour = df_volsvalides[
            (df_volsvalides["NumVol"]     == nv) &
            (df_volsvalides["AeroDepart"] == dep) &
            (df_volsvalides["AeroArriv"]  == arr)
        ]
        if not match_jour.empty:
            jours_connus = set(pd.to_datetime(match_jour["Date"]).dt.dayofweek.unique())
            if jour not in jours_connus:
                jours_connus_noms = [jours_noms[j] for j in sorted(jours_connus)]
                raisons.append(
                    f"Date trouvée : {jours_noms[jour]} {str(dat)[:10]} | "
                    f"Jours habituels : {', '.join(jours_connus_noms)} "
                    f"pour vol {nv} route {dep}→{arr}"
                )
                types.append("DATE")

    # ── 5. Score extrêmement bas ──────────────────────────────────────────────
    if seuil_tres is not None and score < seuil_tres:
        raisons.append(f"Score IF très bas ({score:.4f}) — combinaison très atypique")
        types.append("SCORE")

    # ── 6. Fallback ───────────────────────────────────────────────────────────
    if not raisons:
        jour = pd.to_datetime(dat).dayofweek if pd.notna(dat) else None

        freq_sans_jour = len(df_volsvalides[
            (df_volsvalides["NumVol"]     == nv) &
            (df_volsvalides["AeroDepart"] == dep) &
            (df_volsvalides["AeroArriv"]  == arr) &
            (df_volsvalides["Matricule"]  == mat)
        ])

        freq_avec_jour = len(df_volsvalides[
            (df_volsvalides["NumVol"]     == nv) &
            (df_volsvalides["AeroDepart"] == dep) &
            (df_volsvalides["AeroArriv"]  == arr) &
            (df_volsvalides["Matricule"]  == mat) &
            (pd.to_datetime(df_volsvalides["Date"]).dt.dayofweek == jour)
        ]) if jour is not None else freq_sans_jour

        if freq_avec_jour > 0:
            raisons.append(
                f"Score IF légèrement anormal ({score:.4f}) — "
                f"combinaison connue ce jour ({freq_avec_jour} fois dans vols valides) — possible faux positif"
            )
        elif freq_sans_jour > 0:
            jours_noms = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]
            raisons.append(
                f"Vol {nv} {dep}→{arr} mat:{mat} connu ({freq_sans_jour} fois) "
                f"mais jamais opéré le {jours_noms[jour]} (score:{score:.4f})"
            )
            types.append("FAUX_POSITIF")
        else:
            match_route_ref = df_volsvalides[
                (df_volsvalides["NumVol"]     == nv) &
                (df_volsvalides["AeroDepart"] == dep) &
                (df_volsvalides["AeroArriv"]  == arr)
            ]
            famille_mat = get_fleet_family(mat)
            if not match_route_ref.empty:
                mat_connus = match_route_ref["Matricule"].unique()
                meme_famille = any(same_fleet_family(mat, m) for m in mat_connus)
                if meme_famille:
                    # ✅ Ex: 7T-VCB à la place de 7T-VCA → même Boeing → pas d'anomalie réelle
                    boeing_mat = f"Boeing {famille_mat}" if famille_mat != 0 else "type inconnu"
                    raisons.append(
                        f"Combinaison (Vol:{nv} | {dep}→{arr} | Mat:{mat}) absente des vols valides "
                        f"— mais matricule de même famille ({boeing_mat}) que les appareils habituels : "
                        f"{', '.join(mat_connus)} (score:{score:.4f}) — possible faux positif"
                    )
                    types.append("FAUX_POSITIF")  # ← était INCONNU, maintenant FAUX_POSITIF
                else:
                    # ❌ Ex: 7T-VCN (famille 3) à la place de 7T-VCA (famille 1) → vraie anomalie
                    familles_connues = set(
                        get_fleet_family(m) for m in mat_connus if get_fleet_family(m) != 0
                    )
                    famille_mat_str = f"Boeing {famille_mat}" if famille_mat != 0 else "type inconnu"
                    familles_str    = ", ".join(f"Boeing {f}" for f in sorted(familles_connues)) \
                                      if familles_connues else "type(s) inconnu(s)"
                    raisons.append(
                        f"Matricule trouvé : '{mat}' ({famille_mat_str}) | "
                        f"Matricule(s) attendu(s) : {', '.join(mat_connus)} ({familles_str}) "
                        f"sur route {dep}→{arr} vol {nv}"
                    )
                    types.append("MATRICULE")  # ← était INCONNU, maintenant MATRICULE
            else:
                raisons.append(
                    f"Combinaison (Vol:{nv} | {dep}→{arr} | Mat:{mat}) "
                    f"absente des vols valides (score:{score:.4f})"
                )
                types.append("INCONNU")

    # ── Type final ────────────────────────────────────────────────────────────
    if len(types) == 0:
        type_final = "INCONNU"
    elif len(types) == 1:
        type_final = types[0]
    else:
        type_final = "MULTIPLE"

    return " | ".join(raisons), type_final



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

    # ── 2b. Isoler les vols avec champs vides → anomalie directe, pas IF ─────
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

    # Construire la raison pour chaque vol incomplet
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
    n_incomplets = len(df_incomplets)
    if n_incomplets:
        logger.warning(
            f"  {n_incomplets} vols avec champs vides "
            f"→ flaggés Suspect directement (non scorés par IF)."
        )

    # Passer uniquement les vols complets dans l'IF
    df_enc = df_enc.loc[idx_complets].reset_index(drop=True)
    df_nm  = df_nm.loc[idx_complets].reset_index(drop=True)

    # ── 3. Isolation Forest ───────────────────────────────────────────────────
    logger.info("─" * 50)
    logger.info("Scoring Isolation Forest sur vols non matches...")
    X           = encoder.transform(df_enc)
    scores      = model.score_samples(X)
    predictions = model.predict(X)
    statuts, seuil_tres = attribuer_statut(scores, predictions)

    # ── 4. Raisons + types d'anomalie ─────────────────────────────────────────
    raisons = []
    types   = []
    for i, (idx, row) in enumerate(df_enc.iterrows()):
        if statuts[i] == "Normal":
            raisons.append("Aucune")
            types.append("—")
        else:
            raison, type_anom = analyser_raison_anomalie(df_ref, row, scores[i], seuil_tres)
            raisons.append(raison)
            types.append(type_anom)

    # ── 5. Construction résultats ─────────────────────────────────────────────
    df_result = df_nm.copy().reset_index(drop=True)
    df_result["ScoreAnomalie"]  = np.round(scores, 4)
    df_result["Statut"]         = statuts
    df_result["TypeAnomalie"]   = types
    df_result["RaisonAnomalie"] = raisons

    anom    = df_result[df_result["Statut"] != "Normal"].copy()
    normaux = df_result[df_result["Statut"] == "Normal"].copy()

    # Fusionner les anomalies IF avec les vols incomplets détectés avant IF
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

# ── 5b. Vols validés par Isolation Forest (Normal) ────────────────────────
    valides = df_result[df_result["Statut"] == "Normal"].copy()

    valides_aims = valides[valides["source_table"] == "AIMS"].drop(columns=["source_table"])
    valides_mro  = valides[valides["source_table"] == "MRO"].drop(columns=["source_table"])

    n_valides_aims = len(valides_aims)
    n_valides_mro  = len(valides_mro)

    logger.info(f"  AIMS validés par IF : {n_valides_aims}")
    logger.info(f"  MRO  validés par IF : {n_valides_mro}")

    # ── 6. Scan famille croisée sur vols matchés (VolsValidesEtape1) ─────────
    logger.info("─" * 50)
    logger.info("Scan anomalies famille appareil sur vols valides...")


    # ── 7. Sauvegarde SQL Server ──────────────────────────────────────────────
    logger.info("─" * 50)
    logger.info("Sauvegarde dans SQL Server...")

    write_table(anom_aims, "AnomaliesAIMS")

    anom_mro = anom_mro.rename(columns={
        "IDAIMS":         "IDMRO",
        "MatriculeAIMS":  "MatriculeMRO",
        "NumVolAIMS":     "NumVolMRO",
        "AeroDepartAIMS": "AeroDepartMRO",
        "AeroArrivAIMS":  "AeroArrivMRO",
        "DateAIMS":       "DateMRO",
    })
    write_table(anom_mro, "AnomaliesMRO")

    # Vols non matchés classés NORMAUX par IF (predict == 1)
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

    # ── 8. Résumé ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  RÉSULTATS DÉTECTION ANOMALIES")
    print("=" * 55)

    n_incomp_aims = (df_incomplets["source_table"] == "AIMS").sum()
    n_incomp_mro  = (df_incomplets["source_table"] == "MRO").sum()

    print(f"\n  AIMS ({n_aims_total} vols non matchés) :")
    print(f"    ✓ Normaux (IF)         : {len(norm_aims)}")
    print(f"    ⚠ Suspects (IF)        : {(anom_aims['Statut'] == 'Suspect').sum() - n_incomp_aims}")
    print(f"    ✗ Très Suspects (IF)   : {(anom_aims['Statut'] == 'Très Suspect').sum()}")
    print(f"    ✗ Données incomplètes  : {n_incomp_aims}")


    print(f"\n  MRO ({n_mro_total} vols non matchés) :")
    print(f"    ✓ Normaux (IF)         : {len(norm_mro)}")
    print(f"    ⚠ Suspects (IF)        : {(anom_mro['Statut'] == 'Suspect').sum() - n_incomp_mro}")
    print(f"    ✗ Très Suspects (IF)   : {(anom_mro['Statut'] == 'Très Suspect').sum()}")
    print(f"    ✗ Données incomplètes  : {n_incomp_mro}")


    print("\n  Tables créées dans SQL Server :")
    print(f"    → AnomaliesAIMS     ({len(anom_aims)} lignes)")
    print(f"    → AnomaliesMRO      ({len(anom_mro)} lignes)")
    print(f"    → VoisNormalesAIMS  ({len(norm_aims)} lignes)")
    print(f"    → VoisNormalesMRO   ({len(norm_mro)} lignes)")
    print("=" * 55)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run()