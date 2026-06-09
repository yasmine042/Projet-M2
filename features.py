# features.py  — version corrigée complète
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from config import ENCODER_PATH

logger = logging.getLogger(__name__)

# ─── Familles d'appareils (fleet types) ───────────────────────────────────────

FLEET_FAMILIES = {
    "7T-VCA": 1, "7T-VCB": 1, "7T-VCC": 1, "7T-VCD": 1,
    "7T-VCE": 1, "7T-VCF": 1, "7T-VCT": 1,
    "7T-VCP": 2, "7T-VCQ": 2, "7T-VCR": 2, "7T-VCS": 2,
    "7T-VCL": 3, "7T-VCM": 3, "7T-VCN": 3, "7T-VCO": 3,
}

def get_fleet_family(matricule: str) -> int:
    return FLEET_FAMILIES.get(str(matricule).strip().upper(), 0)

def same_fleet_family(mat1: str, mat2: str) -> bool:
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
    Features (9) :
      NumVolNum         → int extrait du numéro de vol
      MatriculeCode     → LabelEncoder
      FleetFamilyCode   → famille (1/2/3) ou 0
      FreqVolJour       → P(vol opère ce jour)
      FreqVolFamille    → P(vol opéré par cette FAMILLE)
      FreqRouteJour     → P(route opère ce jour)
      FreqVolRoute      → P(NumVol, Dep, Arr) sans jour ni immat
      FreqVolRouteJour  → P(NumVol, Dep, Arr, Jour) sans immat
      FreqComboFamille  → P(NumVol+Route+Famille+Jour)

    Raisonement clé :
      Deux matricules de la même famille sont considérés équivalents.
      FreqVolFamille et FreqComboFamille agrègent sur la famille, pas sur le matricule.
      Un vol fait habituellement par la famille 1, s'il arrive avec un autre matricule
      de la famille 1 → fréquence positive → pas d'anomalie.
      Un matricule hors famille (ou famille inconnue = 0) → fréquence potentiellement
      nulle → -1.0 → détecté comme anomalie par l'Isolation Forest.
    """

    def __init__(self):
        self.encoders     = {}
        self._fitted      = False
        self._freq_vol_jour      = {}
        self._freq_vol_famille   = {}   # (NumVol, FleetFamily) → float
        self._freq_route_jour    = {}
        self._freq_vol_route     = {}   # (NumVol, Dep, Arr) → float
        self._freq_vol_route_jour = {}  # (NumVol, Dep, Arr, jour) → float
        self._freq_combo_fam     = {}   # (NumVol, Dep, Arr, FleetFamily, jour) → float
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
        tmp = df.copy()
        tmp["_jour"]    = tmp["Date"].dt.dayofweek
        tmp["_route"]   = tmp["AeroDepart"].astype(str) + "_" + tmp["AeroArriv"].astype(str)
        tmp["_nv"]      = tmp["NumVol"].astype(str)
        tmp["_famille"] = tmp["Matricule"].apply(get_fleet_family)

        # freq_vol_jour : fraction du vol ce jour
        vol_total    = tmp.groupby("_nv").size()
        vol_jour_cnt = tmp.groupby(["_nv", "_jour"]).size()
        self._freq_vol_jour = (vol_jour_cnt / vol_total).to_dict()

        # freq_vol_famille : fraction du vol assurée par cette FAMILLE (pas le matricule)
        # Permet l'interchangeabilité intra-famille
        vol_fam_cnt  = tmp.groupby(["_nv", "_famille"]).size()
        self._freq_vol_famille = (vol_fam_cnt / vol_total).to_dict()

        # freq_route_jour
        route_total    = tmp.groupby("_route").size()
        route_jour_cnt = tmp.groupby(["_route", "_jour"]).size()
        self._freq_route_jour = (route_jour_cnt / route_total).to_dict()

        # freq_vol_route : fréquence de (NumVol, Dep, Arr) – sans jour ni immat
        n_total          = len(tmp)
        vol_route_cnt    = tmp.groupby(["_nv", "AeroDepart", "AeroArriv"]).size()
        self._freq_vol_route = (vol_route_cnt / n_total).to_dict()

        # freq_vol_route_jour : fréquence de (NumVol, Dep, Arr, Jour) – sans immat
        vol_route_jour_cnt = tmp.groupby(["_nv", "AeroDepart", "AeroArriv", "_jour"]).size()
        self._freq_vol_route_jour = (vol_route_jour_cnt / n_total).to_dict()

        # freq_combo_famille : combinaison complète avec FAMILLE (pas matricule)
        combo_cnt = tmp.groupby(
            ["_nv", "AeroDepart", "AeroArriv", "_famille", "_jour"]
        ).size()
        self._freq_combo_fam = (combo_cnt / n_total).to_dict()
        self._known_combos   = set(self._freq_combo_fam.keys())

        logger.info(
            f"  Tables freq : {len(self._freq_vol_jour)} vol-jours | "
            f"{len(self._freq_vol_famille)} vol-familles | "
            f"{len(self._freq_route_jour)} route-jours | "
            f"{len(self._freq_vol_route)} vol-routes | "
            f"{len(self._freq_vol_route_jour)} vol-route-jours | "
            f"{len(self._freq_combo_fam)} combos (famille)"
        )

    def fit(self, df):
        le = LabelEncoder()
        le.fit(df["Matricule"].astype(str))
        self.encoders["Matricule"] = le
        logger.info(f"  'Matricule' : {len(le.classes_)} valeurs uniques")

        self._build_freq_tables(df)
        self._fitted = True
        logger.info(f"Encodeur entraine sur {len(df)} vols valides (9 features).")
        return self

    def transform(self, df):
        if not self._fitted:
            raise RuntimeError("Appelez .fit() avant .transform()")

        result = pd.DataFrame(index=df.index)

        # ── features de base ────────────────────────────────────────────────
        result["NumVolNum"]       = self._numvol_to_int(df["NumVol"])
        result["MatriculeCode"]   = self._label_encode("Matricule",  df["Matricule"])
        result["FleetFamilyCode"] = df["Matricule"].apply(
            lambda m: get_fleet_family(str(m))
        )

        # ── features de fréquence ────────────────────────────────────────────
        nv      = df["NumVol"].astype(str)
        dep     = df["AeroDepart"].astype(str)
        arr     = df["AeroArriv"].astype(str)
        route   = dep + "_" + arr
        jour    = df["Date"].apply(lambda d: d.dayofweek if pd.notna(d) else -1)
        famille = df["Matricule"].apply(get_fleet_family)  # int 0/1/2/3

        result["FreqVolJour"] = [
            self._freq_vol_jour.get((v, j), -1.0)
            for v, j in zip(nv, jour)
        ]

        # Lookup par FAMILLE (interchangeabilité intra-famille garantie)
        result["FreqVolFamille"] = [
            self._freq_vol_famille.get((v, f), -1.0)
            for v, f in zip(nv, famille)
        ]

        result["FreqRouteJour"] = [
            self._freq_route_jour.get((r, j), -1.0)
            for r, j in zip(route, jour)
        ]
        result["FreqVolRoute"] = [
            self._freq_vol_route.get((v, d, a), -1.0)
            for v, d, a in zip(nv, dep, arr)
        ]
        result["FreqVolRouteJour"] = [
            self._freq_vol_route_jour.get((v, d, a, j), -1.0)
            for v, d, a, j in zip(nv, dep, arr, jour)
        ]

        # Combo complet avec famille (pas matricule)
        result["FreqComboFamille"] = [
            self._freq_combo_fam.get((v, d, a, f, j), -1.0)
            for v, d, a, f, j in zip(nv, dep, arr, famille, jour)
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