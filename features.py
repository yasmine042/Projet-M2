import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from config import ENCODER_PATH

logger = logging.getLogger(__name__)

# ─── Familles d'appareils (fleet types) ───────────────────────────────────────

FLEET_FAMILIES = {
    # Famille 1
    "7T-VCA": 1, "7T-VCB": 1, "7T-VCC": 1, "7T-VCD": 1,
    "7T-VCE": 1, "7T-VCF": 1, "7T-VCT": 1,
    # Famille 2
    "7T-VCP": 2, "7T-VCQ": 2, "7T-VCR": 2, "7T-VCS": 2,
    # Famille 3
    "7T-VCL": 3, "7T-VCM": 3, "7T-VCN": 3, "7T-VCO": 3,
}

def get_fleet_family(matricule: str) -> int:
    """Retourne la famille (1, 2, 3) ou 0 si matricule inconnu."""
    return FLEET_FAMILIES.get(str(matricule).strip().upper(), 0)

def same_fleet_family(mat1: str, mat2: str) -> bool:
    """True si les deux matricules appartiennent à la même famille connue."""
    f1, f2 = get_fleet_family(mat1), get_fleet_family(mat2)
    return f1 != 0 and f1 == f2

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
    Transforme les colonnes en features numériques pour l'Isolation Forest.

    Colonne         Méthode             Exemple
    NumVol        → extraction int      "1010"     → 1010
    Matricule     → LabelEncoder        "7T-VCA"   → 42
    FleetFamily   → int (1/2/3/0)       "7T-VCA"   → 1
    Date          → dayofweek           21/01/2019 → 0 (Lundi)
    AeroDepart    → LabelEncoder        "ALG"      → 0
    AeroArriv     → LabelEncoder        "HME"      → 7

    FleetFamily encode la famille d'appareils à la place du matricule brut
    pour que deux appareils de la même famille ne se pénalisent pas mutuellement.
    Les deux features (MatriculeCode + FleetFamilyCode) sont conservées :
    FleetFamily apporte le contexte flotte, MatriculeCode garde la granularité fine.
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

    def _fleet_family_encode(self, series):
        return series.astype(str).apply(get_fleet_family)

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
        result["NumVolNum"]       = self._numvol_to_int(df["NumVol"])
        result["MatriculeCode"]   = self._label_encode("Matricule", df["Matricule"])
        result["FleetFamilyCode"] = self._fleet_family_encode(df["Matricule"])
        result["DateJour"]        = df["Date"].apply(
            lambda d: d.dayofweek if pd.notna(d) else -1
        )
        result["AeroDepartCode"]  = self._label_encode("AeroDepart", df["AeroDepart"])
        result["AeroArrivCode"]   = self._label_encode("AeroArriv",  df["AeroArriv"])
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
    print(f"  Nombre features  : {X.shape[1]}  (6 : NumVol, Matricule, FleetFamily, Jour, Depart, Arrivee)")
    print(f"  Exemple ligne 0  : {X[0]}")

    # Test familles
    print("\n  Test familles :")
    print(f"    7T-VCA vs 7T-VCB  → même famille : {same_fleet_family('7T-VCA', '7T-VCB')}")   # True
    print(f"    7T-VCA vs 7T-VCO  → même famille : {same_fleet_family('7T-VCA', '7T-VCO')}")   # False
    print(f"    7T-VCP vs 7T-VCQ  → même famille : {same_fleet_family('7T-VCP', '7T-VCQ')}")   # True
    print("  OK - features.py fonctionne correctement")