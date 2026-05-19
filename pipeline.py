" lance tout de A à Z"
import logging
from train import train
from predict import run

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

print("\n" + "=" * 45)
print("  PIPELINE DETECTION ANOMALIES VOLS")
print("=" * 45)

print("\n[ETAPE 1/2] Entrainement du modele...")
train()

print("\n[ETAPE 2/2] Prediction sur AIMS et MRO...")
run()

print("\n" + "=" * 45)
print("  PIPELINE TERMINE")
print("=" * 45)
print("Les tables AnomaliesAIMS et AnomaliesMRO")
print("sont disponibles dans SQL Server.")