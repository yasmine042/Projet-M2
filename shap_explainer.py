"""
shap_explainer.py — Explication SHAP pour l'Isolation Forest.
Pour chaque vol anomalie : quelle feature a le plus contribué au score IF.
"""

import pickle
import numpy as np
import pandas as pd
import shap

from config import MODEL_PATH, ENCODER_PATH
from features import FlightFeatureEncoder

# Noms lisibles des 9 features (5 de base + 3 fréquences partielles + 1 fréquence complète)
FEATURE_LABELS = [
    "Numéro Vol",
    "Matricule",
    "Jour semaine",
    "Aéroport Départ",
    "Aéroport Arrivée",
    "Fréq. Vol×Jour",
    "Fréq. Vol×Matricule",
    "Fréq. Route×Jour",
    "Fréq. Combo Complète",
]


def load_model_and_encoder():
    """Charge le modèle IF et l'encodeur depuis les fichiers pickle."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    encoder = FlightFeatureEncoder.load()
    return model, encoder


def get_shap_values(row: dict, encoder: FlightFeatureEncoder, explainer) -> dict:
    """
    Calcule les valeurs SHAP pour un vol.

    row : dict avec clés "Num Vol", "Matricule", "Date", "Départ", "Arrivée"
    Retourne : {label_feature: valeur_shap}
    """
    df_row = pd.DataFrame([{
        "NumVol":     str(row.get("Num Vol",   "")),
        "Matricule":  str(row.get("Matricule", "")),
        "Date":       pd.to_datetime(row.get("Date", ""), errors="coerce"),
        "AeroDepart": str(row.get("Départ",   "")),
        "AeroArriv":  str(row.get("Arrivée",  "")),
    }])

    X = encoder.transform(df_row)

    raw = explainer.shap_values(X)
    # TreeExplainer peut retourner liste ou array selon version shap
    if isinstance(raw, list):
        vals = np.array(raw[0]).flatten()
    else:
        vals = np.array(raw).flatten()

    return {label: float(v) for label, v in zip(FEATURE_LABELS, vals)}


def interpret_shap(shap_dict: dict) -> str:
    """Génère une phrase d'interprétation en français."""
    # Feature qui a le plus poussé vers anomalie (valeur la plus négative)
    most_neg = min(shap_dict, key=lambda k: shap_dict[k])
    val_neg  = shap_dict[most_neg]

    # Feature qui a le plus poussé vers normal (valeur la plus positive)
    most_pos = max(shap_dict, key=lambda k: shap_dict[k])
    val_pos  = shap_dict[most_pos]

    lines = []
    if val_neg < -0.001:
        lines.append(
            f"**{most_neg}** est la feature qui a le plus contribué "
            f"à classer ce vol comme anomalie (contribution : {val_neg:+.4f})."
        )
    if val_pos > 0.001:
        lines.append(
            f"**{most_pos}** a au contraire poussé le modèle vers "
            f"une classification normale (contribution : {val_pos:+.4f})."
        )
    if not lines:
        lines.append("Les contributions SHAP sont toutes proches de zéro — vol à la limite du seuil.")

    return " ".join(lines)
