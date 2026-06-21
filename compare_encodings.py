"""
compare_encodings.py -- Comparaison des 4 methodes d'encodage pour le memoire.

Lancement : python compare_encodings.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── Connexion donnees ─────────────────────────────────────────────────────────
try:
    from db import read_table
    from features import normalize_vols_valides
    raw = read_table("VolsValidesEtape1")
    df  = normalize_vols_valides(raw).dropna(
        subset=["Date", "Matricule", "NumVol", "AeroDepart", "AeroArriv"]
    ).reset_index(drop=True)
    SOURCE = "SQL Server -- VolsValidesEtape1"
except Exception as e:
    print(f"[AVERTISSEMENT] Base indisponible ({e}) -- donnees de demonstration utilisees.")
    np.random.seed(42)
    n = 300
    mats  = ["7T-VCA","7T-VCB","7T-VCC","7T-VCD","7T-VCE","7T-VCF",
             "7T-VCT","7T-VCP","7T-VCQ","7T-VCR","7T-VCS","7T-VCL",
             "7T-VCM","7T-VCN","7T-VCO"]
    apts  = ["ALG","ORN","CZL","HME","IAM","TMR","AAE","GJL","TLM","BJA"]
    dates = pd.to_datetime("2019-01-01") + pd.to_timedelta(
        np.random.randint(0, 365, n), unit="D")
    df = pd.DataFrame({
        "Matricule":  np.random.choice(mats, n),
        "NumVol":     np.random.randint(1000, 6500, n).astype(str),
        "AeroDepart": np.random.choice(apts, n),
        "AeroArriv":  np.random.choice(apts, n),
        "Date":       dates,
    })
    SOURCE = "Donnees de demonstration"

SEP  = "=" * 72
SEP2 = "-" * 72
N    = 8   # lignes affichees par tableau

print(f"\n{SEP}")
print("  COMPARAISON DES METHODES D'ENCODAGE -- MEMOIRE PFE")
print(f"  Source : {SOURCE}  ({len(df)} vols)")
print(SEP)

# ── Echantillon brut ──────────────────────────────────────────────────────────
sample = df[["Matricule","NumVol","AeroDepart","AeroArriv"]].head(N).copy()
print(f"\n{SEP2}")
print("  DONNEES BRUTES (extrait)")
print(SEP2)
print(sample.to_string(index=False))

# =============================================================================
# 1. ONE-HOT ENCODING
# =============================================================================
print(f"\n{SEP}")
print("  1. ONE-HOT ENCODING   (variable : Matricule)")
print(SEP)

ohe = pd.get_dummies(df["Matricule"], prefix="MAT").astype(int)
print(f"\n  -> {ohe.shape[1]} nouvelles colonnes creees (une par matricule unique).")
print(f"  -> Chaque ligne contient exactement un 1 et ({ohe.shape[1]-1}) zeros.\n")
print(pd.concat([df["Matricule"].head(N).reset_index(drop=True),
                 ohe.head(N).reset_index(drop=True)], axis=1).to_string(index=False))

total_ohe = df["Matricule"].nunique() + df["NumVol"].nunique() \
          + df["AeroDepart"].nunique() + df["AeroArriv"].nunique()
print(f"\n  INCONVENIENT : {ohe.shape[1]} colonnes pour Matricule seul.")
print(f"  Estimation totale avec One-Hot sur toutes les variables : ~{total_ohe} colonnes.")
print(f"  => Explosion dimensionnelle -- inadapte pour nos algorithmes.")

# =============================================================================
# 2. LABEL ENCODING
# =============================================================================
print(f"\n{SEP}")
print("  2. LABEL ENCODING   (variable : Matricule)")
print(SEP)

le = LabelEncoder()
le.fit(df["Matricule"].astype(str))
label_col = pd.Series(le.transform(df["Matricule"].astype(str)), name="MatriculeCode")

mapping = pd.DataFrame({
    "Matricule (brut)": le.classes_,
    "Code (Label)":     range(len(le.classes_))
})
print(f"\n  -> 1 seule colonne entiere (0 a {len(le.classes_)-1}).")
print(f"  -> {len(le.classes_)} matricules uniques encodes.\n")
print("  Table de correspondance :")
print(mapping.to_string(index=False))

print(f"\n  Extrait resultat :")
print(pd.concat([df["Matricule"].head(N).reset_index(drop=True),
                 label_col.head(N).reset_index(drop=True)], axis=1).to_string(index=False))

print(f"\n  AVANTAGE : compact (1 colonne), sans explosion dimensionnelle.")
print(f"  LIMITE   : introduit un ordre artificiel (7T-VCA=0 < 7T-VCB=1 ...).")
print(f"  => Acceptable ici : IF et RF (arbres) ne supposent pas de relation")
print(f"     d'ordre entre les codes -- chaque seuil est appris independamment.")
print(f"  => Les matricules inconnus recoivent le code -1 (valeur sentinelle).")

# =============================================================================
# 3. TARGET ENCODING
# =============================================================================
print(f"\n{SEP}")
print("  3. TARGET ENCODING   (variable : Matricule, cible binaire simulee)")
print(SEP)

np.random.seed(0)
df["_anom"] = np.random.choice([0,0,0,0,1], len(df))

target_map = df.groupby("Matricule")["_anom"].mean()
target_col = df["Matricule"].map(target_map).rename("MatriculeTargetEnc").round(4)

tbl = target_map.reset_index()
tbl.columns = ["Matricule", "P(anomalie)"]
tbl["P(anomalie)"] = tbl["P(anomalie)"].round(4)
print(f"\n  -> 1 colonne : probabilite moyenne d'anomalie pour ce matricule.\n")
print("  Table de correspondance (P(anomalie | matricule)) :")
print(tbl.sort_values("Matricule").to_string(index=False))

print(f"\n  Extrait resultat :")
print(pd.concat([df["Matricule"].head(N).reset_index(drop=True),
                 target_col.head(N).reset_index(drop=True)], axis=1).to_string(index=False))

print(f"\n  LIMITE CRITIQUE : necessite une variable cible (labels 0/1 connus).")
print(f"  => Notre contexte est NON SUPERVISE (Isolation Forest sans labels).")
print(f"  => NON APPLICABLE dans notre pipeline d'entrainement IF.")
df.drop(columns=["_anom"], inplace=True)

# =============================================================================
# 4. FREQUENCY ENCODING -- univarie
# =============================================================================
print(f"\n{SEP}")
print("  4. FREQUENCY ENCODING   (variable : Matricule -- frequence univariee)")
print(SEP)

freq_map = df["Matricule"].value_counts(normalize=True)
freq_col  = df["Matricule"].map(freq_map).rename("MatriculeFreq").round(6)

tbl_f = freq_map.reset_index()
tbl_f.columns = ["Matricule", "P(Matricule)"]
tbl_f["P(Matricule)"] = tbl_f["P(Matricule)"].round(4)
print(f"\n  -> 1 colonne : proportion relative du matricule dans les donnees.\n")
print("  Table de correspondance P(Matricule) :")
print(tbl_f.sort_values("Matricule").to_string(index=False))

print(f"\n  Extrait resultat :")
print(pd.concat([df["Matricule"].head(N).reset_index(drop=True),
                 freq_col.head(N).reset_index(drop=True)], axis=1).to_string(index=False))

print(f"\n  AVANTAGE : aucune variable cible requise => adapte au non supervise.")
print(f"  Un matricule inconnu recoit -1.0 (jamais vu dans l'entrainement).")

# =============================================================================
# 4b. FREQUENCY ENCODING MULTIDIMENSIONNEL (tel qu'implemente dans notre modele)
# =============================================================================
print(f"\n{SEP}")
print("  4b. FREQUENCY ENCODING MULTIDIMENSIONNEL (implementation reelle)")
print(SEP)

tmp = df.copy()
tmp["_jour"]  = tmp["Date"].dt.dayofweek
n_total = len(tmp)

freq_vol_route      = tmp.groupby(["NumVol","AeroDepart","AeroArriv"]).size() / n_total
freq_vol_route_jour = tmp.groupby(["NumVol","AeroDepart","AeroArriv","_jour"]).size() / n_total
freq_vol_jour_ser   = (tmp.groupby(["NumVol","_jour"]).size() /
                       tmp.groupby("NumVol").size())

s8 = df.head(N).copy().reset_index(drop=True)
s8["_jour"] = s8["Date"].dt.dayofweek

s8["FreqVolJour"] = [
    round(freq_vol_jour_ser.get((r["NumVol"], r["_jour"]), -1.0), 6)
    for _, r in s8.iterrows()
]
s8["FreqVolRoute"] = [
    round(freq_vol_route.get((r["NumVol"], r["AeroDepart"], r["AeroArriv"]), -1.0), 6)
    for _, r in s8.iterrows()
]
s8["FreqVolRouteJour"] = [
    round(freq_vol_route_jour.get(
        (r["NumVol"], r["AeroDepart"], r["AeroArriv"], r["_jour"]), -1.0), 6)
    for _, r in s8.iterrows()
]

print(f"\n  Principe : frequences calculees sur des COMBINAISONS de variables.")
print(f"  Une frequence = -1.0 => la combinaison n'a jamais ete observee.\n")
print(s8[["NumVol","AeroDepart","AeroArriv","_jour",
          "FreqVolJour","FreqVolRoute","FreqVolRouteJour"]].to_string(index=False))

print(f"""
  Interpretation :
    FreqVolJour       = P(ce numero de vol opere ce jour de semaine)
    FreqVolRoute      = P(NumVol + AeroDepart + AeroArriv ensemble)
    FreqVolRouteJour  = P(NumVol + Route + Jour de semaine ensemble)
    Valeur -1.0       = combinaison absente des vols valides => signal d'anomalie
""")

# =============================================================================
# TABLEAU COMPARATIF FINAL
# =============================================================================
print(SEP)
print("  TABLEAU COMPARATIF -- CHOIX FINAL")
print(SEP)

tableau = pd.DataFrame([
    ["One-Hot",   "N colonnes (1 par modalite)", "Non",  "Non",  "=> Explosion dimensionnelle -- ECARTE"],
    ["Label",     "1 colonne entiere",            "Non",  "Oui",  "=> CHOISI pour Matricule"],
    ["Target",    "1 colonne float (P cible)",    "Oui",  "Non",  "=> Requiert labels -- IMPOSSIBLE non supervise"],
    ["Frequency", "1 colonne float (proportion)", "Non",  "Oui",  "=> CHOISI pour les 6 features de frequence"],
], columns=["Methode", "Dimensionnalite", "Cible requise", "Non supervise", "Decision"])

print(tableau.to_string(index=False))

print(f"""
{SEP}
  JUSTIFICATION DU CHOIX DANS NOTRE MODELE
{SEP}

  LABEL ENCODING  =>  Matricule
  -----------------------------------------------------------------------
  Le LabelEncoder de Scikit-learn encode chaque matricule par un entier
  unique (ex : 7T-VCA -> 42). L'Isolation Forest et le Random Forest
  etant bases sur des arbres de decision, ils n'imposent aucune relation
  d'ordre entre les codes : le code 42 n'est pas "superieur" a 0, chaque
  seuil de branchement est appris independamment.
  L'inconvenient classique du Label Encoding (ordre artificiel) est donc
  neutralise par le choix d'algorithmes a base d'arbres.
  Les matricules inconnus (absents de l'entrainement) recoivent -1.

  FREQUENCY ENCODING MULTIDIMENSIONNEL  =>  6 features de frequence
  -----------------------------------------------------------------------
  Plutot qu'une frequence univariee P(Matricule), notre encodeur calcule
  des probabilites sur des COMBINAISONS de variables, capturant ainsi
  les dependances entre dimensions :

    FreqVolJour       = P(vol opere ce jour de semaine)
    FreqVolFamille    = P(vol assure par cette famille d'appareils)
    FreqRouteJour     = P(route opere ce jour de semaine)
    FreqVolRoute      = P(NumVol + AeroDepart + AeroArriv ensemble)
    FreqVolRouteJour  = P(NumVol + Route + Jour ensemble)
    FreqComboFamille  = P(NumVol + Route + Famille + Jour ensemble)

  Trois avantages cles :
    (1) Aucune variable cible => compatible avec Isolation Forest (non supervise)
    (2) La valeur -1.0 signale une combinaison inconnue, signal d'anomalie
        directement interpreatable par l'IF
    (3) Les combinaisons multi-variables capturent des patterns impossibles
        a detecter avec une frequence univariee
        (ex : vol habituel opere un jour inhabituel)
{SEP}
""")
