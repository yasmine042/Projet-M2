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
    df["NumVol"]     = df["NumVolAIMS"].astype(str).str.strip().str.upper()
    df["AeroDepart"] = df["AeroDepartAIMS"].astype(str).str.strip().str.upper()
    df["AeroArriv"]  = df["AeroArrivAIMS"].astype(str).str.strip().str.upper()
    df["Date"]       = pd.to_datetime(df["DateAIMS"], errors="coerce")
    return df

# ─── Encodeur ─────────────────────────────────────────────────────────────────

class FlightFeatureEncoder:
    """
    Transforme les 5 colonnes en nombres pour l'Isolation Forest.

    Colonne      Méthode           Exemple
    NumVol     → extraction int   "1010"      → 1010
    Matricule  → LabelEncoder     "7T-VCA"    → 42
    Date       → ordinal          21/01/2019  → 736715
    AeroDepart → LabelEncoder     "ALG"       → 0
    AeroArriv  → LabelEncoder     "HME"       → 7
    """

    def __init__(self):
        self.encoders = {}
        self._fitted  = False

    def _numvol_to_int(self, series):
        extracted = series.astype(str).str.extract(r"(\d+)")[0]
        return pd.to_numeric(extracted, errors="coerce").fillna(-1).astype(int)

    def _label_encode(self, col, series):
        le    = self.encoders[col]
        known = set(le.classes_)
        return series.astype(str).apply(
            lambda x: int(le.transform([x])[0]) if x in known else -1
        )

    def fit(self, df):
        for col in ["Matricule", "AeroDepart", "AeroArriv"]:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.encoders[col] = le
            logger.info(f"  '{col}' : {len(le.classes_)} valeurs uniques")
        self._fitted = True
        logger.info(f"Encodeur entraine sur {len(df)} vols valides.")
        return self

    def transform(self, df):
        if not self._fitted:
            raise RuntimeError("Appelez .fit() avant .transform()")
        result = pd.DataFrame(index=df.index)
        result["NumVolNum"]      = self._numvol_to_int(df["NumVol"])
        result["MatriculeCode"]  = self._label_encode("Matricule",  df["Matricule"])
        result["DateOrdinal"]    = df["Date"].apply(
            lambda d: d.toordinal() if pd.notna(d) else -1
        )
        result["AeroDepartCode"] = self._label_encode("AeroDepart", df["AeroDepart"])
        result["AeroArrivCode"]  = self._label_encode("AeroArriv",  df["AeroArriv"])
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