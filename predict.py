"prédiction sur données brutes"
import pickle
import logging
import pandas as pd
from config import MODEL_PATH
from db import read_table, write_table
from features import FlightFeatureEncoder, normalize_aims, normalize_mro

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Modele charge.")
    return model

def predict(df, model, encoder, source):
    X           = encoder.transform(df)
    scores      = model.score_samples(X)
    predictions = model.predict(X)

    df = df.copy()
    df["AnomalyScore"] = scores
    df["IsAnomaly"]    = predictions == -1
    df["Source"]       = source

    n = df["IsAnomaly"].sum()
    logger.info(f"  [{source}] {n} anomalies sur {len(df)} vols ({n/len(df)*100:.1f}%)")
    return df

def run():
    # ── Charger modele et encodeur ─────────────────────────────────────────
    model   = load_model()
    encoder = FlightFeatureEncoder.load()

    # ── AIMS ───────────────────────────────────────────────────────────────
    logger.info("Prediction sur AIMS...")
    raw_aims = read_table("AIMS")
    df_aims  = normalize_aims(raw_aims)
    df_aims  = df_aims.dropna(subset=["Date", "Matricule", "NumVol", "AeroDepart", "AeroArriv"])
    res_aims = predict(df_aims, model, encoder, "AIMS")

    # ── MRO ────────────────────────────────────────────────────────────────
    logger.info("Prediction sur MRO...")
    raw_mro = read_table("MRO")
    df_mro  = normalize_mro(raw_mro)
    df_mro  = df_mro.dropna(subset=["Date", "Matricule", "NumVol", "AeroDepart", "AeroArriv"])
    res_mro = predict(df_mro, model, encoder, "MRO")

    # ── Sauvegarder uniquement les anomalies dans SQL Server ───────────────
    anomalies_aims = res_aims[res_aims["IsAnomaly"]][
        ["Matricule", "NumVol", "Date", "AeroDepart", "AeroArriv", "AnomalyScore", "Source"]
    ]
    anomalies_mro = res_mro[res_mro["IsAnomaly"]][
        ["Matricule", "NumVol", "Date", "AeroDepart", "AeroArriv", "AnomalyScore", "Source"]
    ]

    write_table(anomalies_aims, "AnomaliesAIMS")
    write_table(anomalies_mro,  "AnomaliesMRO")

    # ── Résumé ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 45)
    print("RESULTAT PREDICTION")
    print("=" * 45)
    print(f"  AIMS  total    : {len(res_aims)} vols")
    print(f"  AIMS  anomalies: {res_aims['IsAnomaly'].sum()} vols suspects")
    print(f"  MRO   total    : {len(res_mro)} vols")
    print(f"  MRO   anomalies: {res_mro['IsAnomaly'].sum()} vols suspects")
    print("=" * 45)
    print("\nTables creees dans SQL Server :")
    print("  -> AnomaliesAIMS")
    print("  -> AnomaliesMRO")

if __name__ == "__main__":
    run()