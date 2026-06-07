"""
Configuration centrale du projet — Détection d'anomalies de vols.
Modifiez uniquement les valeurs marquées 'à adapter'.
"""

# ─── Connexion SQL Server ──────────────────────────────────────────────────────
DB_CONFIG = {
    "server":             "DESKTOP-GH5U4JQ",
    "database":           "TALEXPDWH",
    "driver":             "ODBC Driver 17 for SQL Server",
    "trusted_connection": "yes",
}

# ─── Tables source (ne pas modifier) ──────────────────────────────────────────
TABLE_AIMS         = "AIMS"
TABLE_MRO          = "MRO"
TABLE_VOLS_VALIDES = "VolsValidesEtape1"   # données d'entraînement

# ─── Tables résultantes créées automatiquement ────────────────────────────────
TABLE_ANOMALIES_AIMS = "AnomaliesAIMS"
TABLE_ANOMALIES_MRO  = "AnomaliesMRO"

# ─── Isolation Forest ─────────────────────────────────────────────────────────
IF_CONFIG = {
    "n_estimators": 200,    # nombre d'arbres — augmenter si résultats instables  , lorsqu on augmente la vitesse diminue, lorsqu on diminue la vitesse auugmente mais n'est pas bien entrainé 
    "contamination": 0.05,  # 5% de vols attendus comme anomalies — d apres calcul de vols invalides depuis aims 
    "random_state":  42,    # ne pas changer — garantit la reproductibilité
    "max_samples":  "auto", # ne pas changer
}

# ─── Les 5 features de validation (définies par le promoteur) ─────────────────
FEATURES = [
    "NumVolNum",       # NumVol     → extraction entière   ex: "1010"     → 1010
    "MatriculeCode",   # Matricule  → LabelEncoder         ex: "7T-VCA"   → 42
    "DateJour",        # Date       → dayofweek             ex: 21/01/2019 → 0 (Lundi)
    "AeroDepartCode",  # AeroDepart → LabelEncoder         ex: "ALG"      → 0
    "AeroArrivCode",   # AeroArriv  → LabelEncoder         ex: "HME"      → 7
]
# ─── Features d'anomalie métier (invisibles au SSIS) ─────────────────────────
FEATURES_ANOMALIES = [
    "DureeVolMin",     # ATA - ATD en minutes  → durée de vol anormale ?
    "RetardDepartMin", # ATD - ETD en minutes  → retard départ anormal ?
    "RetardArrivMin",  # ATA - ETA en minutes  → retard arrivée anormal ?
    "BlockMin",        # bh_h*60 + bh_m        → temps bloc anormal ?
]
# ─── Chemins de sauvegarde ────────────────────────────────────────────────────
MODEL_PATH   = "models/isolation_forest.pkl"
ENCODER_PATH = "models/label_encoders.pkl"
LOG_PATH     = "logs/pipeline.log"