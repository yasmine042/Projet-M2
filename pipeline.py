" lance tout de A à Z"
import logging
from train import train
from train_rf import train_rf
from predict import run, run_rf_scan

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/pipeline.log", encoding="utf-8")
    ]
)

import os
os.makedirs("logs", exist_ok=True)

print("\n" + "=" * 50)
print("  PIPELINE DETECTION ANOMALIES VOLS")
print("=" * 50)

print("\n[ETAPE 1/4] Entraînement Isolation Forest...")
train()

print("\n[ETAPE 2/4] Prédiction IF sur vols non matchés...")
run()

print("\n[ETAPE 3/4] Entraînement Random Forest supervisé...")
train_rf()

print("\n[ETAPE 4/4] Scan RF — détection des faux négatifs IF...")
run_rf_scan()

print("\n" + "=" * 50)
print("  PIPELINE TERMINÉ")
print("=" * 50)
print("Tables disponibles dans SQL Server :")
print("  AnomaliesAIMS / AnomaliesMRO      (détectées par IF)")
print("  FauxNegatifsAIMS / FauxNegatifsMRO (rattrapées par RF)")
print("  VoisNormalesAIMS / VoisNormalesMRO  (normaux confirmés)")
