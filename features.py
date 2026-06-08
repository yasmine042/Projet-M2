import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from config import ENCODER_PATH

logger = logging.getLogger(__name__)

# ─── Normalisation ────────────────────────────────────────────────────────────

def normalize_aims(df):
    df = df.copy()
    df["Matricule"]  = df["Matricule"].astype(str).str.strip().str.upper()
    df["NumVol"]     = df["NumVol"].astype(str).str.strip().str.upper()
    df["AeroDepart"] = df["AeroDepart"].astype(str).str.strip().str.upper()
    df["AeroArriv"]  = df["AeroArriv"].astype(str).str.strip().str.upper()
    df["Date"]       = pd.to_datetime(df["Date"], errors="coerce")
    return df

def normalize_mro(df):
    df = df.copy()
    numvol = df["flight no."].astype(str).str.upper()
    for prefix in ["SF", "AH", "SE", "SD", "XF"]:
        numvol = numvol.str.replace(prefix, "", regex=False)
    df["NumVol"]     = numvol.str.strip()
    df["Matricule"]  = df["registr"].astype(str).str.strip().str.upper()
    df["AeroDepart"] = df["from"].astype(str).str.strip().str.upper()
    df["AeroArriv"]  = df["to"].astype(str).str.strip().str.upper()
    df["Date"]       = pd.to_datetime(df["Date_Vol"], errors="coerce")
    return df

def normalize_vols_valides(df):
    df = df.copy()
    df["Matricule"]  = df["MatriculeAIMS"].astype(str).str.strip().str.upper()
    df = df[df["Matricule"].str.len() == 6]
    df["NumVol"]     = df["NumVolAIMS"].astype(str).str.strip().str.upper()
    df["AeroDepart"] = df["AeroDepartAIMS"].astype(str).str.strip().str.upper()
    df["AeroArriv"]  = df["AeroArrivAIMS"].astype(str).str.strip().str.upper()
    df["Date"]       = pd.to_datetime(df["DateAIMS"], errors="coerce")
    return df

# ─── Encodeur ─────────────────────────────────────────────────────────────────

class FlightFeatureEncoder:
    """
    Transforme les colonnes en nombres pour l'Isolation Forest.

    Features de base (5) :
      NumVol     → extraction int   "1010"      → 1010
      Matricule  → LabelEncoder     "7T-VCA"    → 42
      Date       → dayofweek        21/01/2019  → 0 (Lundi)
      AeroDepart → LabelEncoder     "ALG"       → 0
      AeroArriv  → LabelEncoder     "HME"       → 7

    Features de fréquence historique (3) — calculées sur VolsValidesEtape1 :
      FreqVolJour   → proportion de fois où ce vol opère ce jour (0.0–1.0)
      FreqVolMat    → proportion de fois où ce vol est assuré par cette immat.
      FreqRouteJour → proportion de fois où cette route opère ce jour

    Un vol avec FreqVolJour=0.0 n'a JAMAIS opéré ce jour dans l'historique.
    L'IF détecte 0.0 comme isolé → classé Suspect sans règle externe.
    """

    def __init__(self):
        self.encoders = {}
        self._fitted  = False
        # Tables de fréquence (tuple-key → float)
        self._freq_vol_jour   = {}
        self._freq_vol_mat    = {}
        self._freq_route_jour = {}
        # Set complet des combinaisons valides (NumVol, Dep, Arr, Mat, dayofweek)
        # Equivalent exact de la jointure SQL sur 5 champs + DATEPART(dw)
        self._known_combos: set = set()

    def _numvol_to_int(self, series):
        extracted = series.astype(str).str.extract(r"(\d+)")[0]
        return pd.to_numeric(extracted, errors="coerce").fillna(-1).astype(int)

    def _label_encode(self, col, series):
        le    = self.encoders[col]
        known = set(le.classes_)
        return series.astype(str).apply(
            lambda x: int(le.transform([x])[0]) if x in known else -1
        )

    def _build_freq_tables(self, df):
        """Calcule les 3 tables de fréquence à partir du jeu d'entraînement."""
        tmp = df.copy()
        tmp["_jour"]  = tmp["Date"].dt.dayofweek
        tmp["_route"] = tmp["AeroDepart"].astype(str) + "_" + tmp["AeroArriv"].astype(str)
        tmp["_nv"]    = tmp["NumVol"].astype(str)
        tmp["_mat"]   = tmp["Matricule"].astype(str)

        # freq_vol_jour : parmi tous les vols de ce numéro, quelle fraction est ce jour ?
        vol_total      = tmp.groupby("_nv").size()
        vol_jour_cnt   = tmp.groupby(["_nv", "_jour"]).size()
        self._freq_vol_jour = (vol_jour_cnt / vol_total).to_dict()

        # freq_vol_mat : parmi tous les vols de ce numéro, quelle fraction avec cette immat ?
        vol_mat_cnt    = tmp.groupby(["_nv", "_mat"]).size()
        self._freq_vol_mat = (vol_mat_cnt / vol_total).to_dict()

        # freq_route_jour : parmi tous les vols de cette route, quelle fraction est ce jour ?
        route_total    = tmp.groupby("_route").size()
        route_jour_cnt = tmp.groupby(["_route", "_jour"]).size()
        self._freq_route_jour = (route_jour_cnt / route_total).to_dict()

        # Fréquence de la combinaison complète (5 champs + jour)
        # C'est la feature la plus discriminante : jamais vue → -1.0
        n_total = len(tmp)
        combo_cnt = tmp.groupby(["_nv", "AeroDepart", "AeroArriv", "_mat", "_jour"]).size()
        self._freq_combo = (combo_cnt / n_total).to_dict()

        # Set pour lookup rapide O(1) — utilisé aussi par le post-filtre si besoin
        self._known_combos = set(self._freq_combo.keys())

        logger.info(
            f"  Tables freq : {len(self._freq_vol_jour)} vol-jours | "
            f"{len(self._freq_vol_mat)} vol-mats | "
            f"{len(self._freq_route_jour)} route-jours | "
            f"{len(self._freq_combo)} combos complets"
        )

    def fit(self, df):
        for col in ["Matricule", "AeroDepart", "AeroArriv"]:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.encoders[col] = le
            logger.info(f"  '{col}' : {len(le.classes_)} valeurs uniques")

        self._build_freq_tables(df)
        self._fitted = True
        logger.info(f"Encodeur entraine sur {len(df)} vols valides (9 features).")
        return self

    def transform(self, df):
        if not self._fitted:
            raise RuntimeError("Appelez .fit() avant .transform()")

        result = pd.DataFrame(index=df.index)

        # ── 5 features de base ──────────────────────────────────────────────
        result["NumVolNum"]      = self._numvol_to_int(df["NumVol"])
        result["MatriculeCode"]  = self._label_encode("Matricule",  df["Matricule"])
        result["DateJour"]       = df["Date"].apply(
            lambda d: d.dayofweek if pd.notna(d) else -1
        )
        result["AeroDepartCode"] = self._label_encode("AeroDepart", df["AeroDepart"])
        result["AeroArrivCode"]  = self._label_encode("AeroArriv",  df["AeroArriv"])

        # ── 3 features de fréquence historique ─────────────────────────────
        nv    = df["NumVol"].astype(str)
        mat   = df["Matricule"].astype(str)
        dep   = df["AeroDepart"].astype(str)
        arr   = df["AeroArriv"].astype(str)
        route = dep + "_" + arr
        jour  = df["Date"].apply(lambda d: d.dayofweek if pd.notna(d) else -1)

        # Combinaison jamais vue dans l'historique → -1.0 (hors plage [0,1])
        # L'IF est entraîné sur des vols valides où ces fréquences sont >0
        # Encoder 0 comme -1 place le point clairement hors de la distribution apprise
        result["FreqVolJour"]   = [
            self._freq_vol_jour.get((v, j), -1.0)
            for v, j in zip(nv, jour)
        ]
        result["FreqVolMat"]    = [
            self._freq_vol_mat.get((v, m), -1.0)
            for v, m in zip(nv, mat)
        ]
        result["FreqRouteJour"] = [
            self._freq_route_jour.get((r, j), -1.0)
            for r, j in zip(route, jour)
        ]

        # ── Feature 9 : fréquence de la combinaison complète ────────────────
        # Valeur dans [0.00003, ~0.15] pour les vols connus de l'historique.
        # -1.0 si la combinaison (NumVol+Route+Mat+Jour) n'a jamais existé.
        # C'est la feature la plus discriminante pour les anomalies sémantiques.
        result["FreqComboComplete"] = [
            self._freq_combo.get((v, d, a, m, j), -1.0)
            for v, d, a, m, j in zip(nv, dep, arr, mat, jour)
        ]

        return result.values.astype(float)

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def save(self):
        with open(ENCODER_PATH, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Encodeur sauvegarde -> {ENCODER_PATH}")

    @classmethod
    def load(cls):
        with open(ENCODER_PATH, "rb") as f:
            enc = pickle.load(f)
        logger.info(f"Encodeur charge <- {ENCODER_PATH}")
        return enc

# ─── Test rapide ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from db import read_table

    print("Test features.py...")
    raw = read_table("VolsValidesEtape1")
    df  = normalize_vols_valides(raw)
    df  = df.dropna(subset=["Date", "Matricule", "NumVol", "AeroDepart", "AeroArriv"])

    enc = FlightFeatureEncoder()
    X   = enc.fit_transform(df)

    print(f"  Lignes encodees  : {X.shape[0]}")
    print(f"  Nombre features  : {X.shape[1]}")
    print(f"  Exemple ligne 0  : {X[0]}")
    print("  OK - features.py fonctionne correctement")