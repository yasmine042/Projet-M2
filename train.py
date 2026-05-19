"entraînement Isolation Forest"
import os
import pickle
import logging
from sklearn.ensemble import IsolationForest
from config import IF_CONFIG, MODEL_PATH, ENCODER_PATH
from db import read_table
from features import FlightFeatureEncoder, normalize_vols_valides

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def train():
    # ── 1. Charger VolsValides ─────────────────────────────────────────────
    logger.info("Chargement de VolsValidesEtape1...")
    raw = read_table("VolsValidesEtape1")
    df  = normalize_vols_valides(raw)
    df  = df.dropna(subset=["Date", "Matricule", "NumVol", "AeroDepart", "AeroArriv"])
    logger.info(f"  {len(df)} vols valides apres nettoyage.")

    # ── 2. Encoder les 5 features ──────────────────────────────────────────
    logger.info("Encodage des 5 features...")
    encoder = FlightFeatureEncoder()
    X       = encoder.fit_transform(df)
    logger.info(f"  Matrice : {X.shape[0]} lignes x {X.shape[1]} features.")

    # ── 3. Entrainer l'Isolation Forest ───────────────────────────────────
    logger.info("Entrainement Isolation Forest...")
    model = IsolationForest(**IF_CONFIG)
    model.fit(X)
    logger.info("  Modele entraine.")

    # ── 4. Sauvegarder ────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"  Modele sauvegarde -> {MODEL_PATH}")

    encoder.save()

    # ── 5. Apercu des resultats sur les donnees d'entrainement ────────────
    predictions = model.predict(X)
    scores      = model.score_samples(X)
    n_anomalies = (predictions == -1).sum()

    print("\n" + "=" * 45)
    print("RESULTAT ENTRAINEMENT")
    print("=" * 45)
    print(f"  Vols utilises pour entrainement : {len(X)}")
    print(f"  Anomalies detectees sur train   : {n_anomalies} ({n_anomalies/len(X)*100:.1f}%)")
    print(f"  Score moyen                     : {scores.mean():.4f}")
    print(f"  Score minimum (plus suspect)    : {scores.min():.4f}")
    print(f"  Score maximum (plus normal)     : {scores.max():.4f}")
    print("=" * 45)
    print(f"\nModele pret : {MODEL_PATH}")
    print(f"Encodeur pret : {ENCODER_PATH}")

if __name__ == "__main__":
    train()