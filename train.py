"entraînement Isolation Forest"
import os
import pickle
import logging
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
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

    # ── 3. Split 80/20 ────────────────────────────────────────────────────
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    logger.info(f"  Train : {len(X_train)} vols  |  Test : {len(X_test)} vols")

    # ── 4. Entrainer l'Isolation Forest ───────────────────────────────────
    logger.info("Entrainement Isolation Forest...")
    model = IsolationForest(**IF_CONFIG)
    model.fit(X_train)
    logger.info("  Modele entraine.")

    # ── 5. Sauvegarder ────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"  Modele sauvegarde -> {MODEL_PATH}")

    encoder.save()

    # ── 6. Evaluation sur le test set (vols valides non vus) ──────────────
    pred_train  = model.predict(X_train)
    pred_test   = model.predict(X_test)
    scores_test = model.score_samples(X_test)

    fp_train = (pred_train == -1).sum()
    fp_test  = (pred_test  == -1).sum()

    print("\n" + "=" * 50)
    print("RESULTAT ENTRAINEMENT")
    print("=" * 50)
    print(f"  Vols train                        : {len(X_train)}")
    print(f"  Vols test  (non vus)              : {len(X_test)}")
    print(f"  Faux positifs sur train           : {fp_train} ({fp_train/len(X_train)*100:.1f}%)")
    print(f"  Faux positifs sur test            : {fp_test}  ({fp_test/len(X_test)*100:.1f}%)")
    print(f"    -> attendu ~{IF_CONFIG['contamination']*100:.0f}%  (= contamination)")
    print(f"  Score moyen test                  : {scores_test.mean():.4f}")
    print(f"  Score min  test (plus suspect)    : {scores_test.min():.4f}")
    print(f"  Score max  test (plus normal)     : {scores_test.max():.4f}")
    print("=" * 50)
    print(f"\nModele pret : {MODEL_PATH}")
    print(f"Encodeur pret : {ENCODER_PATH}")

if __name__ == "__main__":
    train()