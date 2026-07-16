# FlightAnomaly

Plateforme de détection d'anomalies de vols pour les données AIMS et MRO.
Ce projet fournit :
- un pipeline de détection basé sur Isolation Forest + Random Forest,
- une connexion à SQL Server via ODBC,
- une interface Streamlit pour explorer les résultats et diagnostiquer les anomalies.

## Fonctionnalités principales

- `pipeline.py` : exécute l'ensemble du workflow de bout en bout
  1. entraînement de l'Isolation Forest (`train.py`)
  2. détection d'anomalies IF sur les vols non matchés (`predict.py`)
  3. entraînement du Random Forest supervisé (`train_rf.py`)
  4. scan RF pour détecter les faux négatifs IF
- `app.py` : application Streamlit principale avec navigation multi-pages
- `config.py` : configuration SQL Server, chemins de modèles et features
- `db.py` : accès SQL Server avec SQLAlchemy + pyodbc
- `features.py` : normalisation des données de vol et encodage métier

## Architecture du projet

- `app.py`
- `pipeline.py`
- `train.py`
- `train_rf.py`
- `predict.py`
- `config.py`
- `db.py`
- `features.py`
- `views/` : pages Streamlit (`home`, `overview`, `dashboard`, `tables`, `detection`)
- `models/` : stockage des modèles et encodeurs
- `logs/` : journaux d'exécution
- `figures/` : graphiques générés par `train_rf.py`

## Prérequis

- Python 3.9+ (ou compatible)
- SQL Server avec accès aux tables suivantes :
  - `AIMS`
  - `MRO`
  - `VolsValidesEtape1`
- ODBC Driver pour SQL Server (`ODBC Driver 17 for SQL Server` ou similaire)

## Dépendances Python

- pandas
- numpy
- scikit-learn
- sqlalchemy
- pyodbc
- streamlit
- plotly
- matplotlib

## Configuration

1. Ouvrez `config.py`.
2. Modifiez `DB_CONFIG` avec le serveur, la base de données et le driver SQL Server.
3. Vérifiez les noms des tables sources si votre schéma diffère.

## Exécution

### 1. Préparer l'environnement

```bash
python -m pip install pandas numpy scikit-learn sqlalchemy pyodbc streamlit plotly matplotlib
```

### 2. Lancer le pipeline complet

```bash
python pipeline.py
```

Ce script enchaîne :
- `train.py`
- `predict.py`
- `train_rf.py`
- la détection RF finale

### 3. Lancer l'interface principale

```bash
streamlit run app.py
```


## Modèles et sorties

- `models/isolation_forest.pkl` : modèle Isolation Forest
- `models/random_forest.pkl` : modèle Random Forest
- `models/label_encoders.pkl` : encodeur des features métier
- `logs/pipeline.log` : journal du pipeline
- `figures/` : images générées par `train_rf.py`

## Notes importantes

- `train.py` utilise la table `VolsValidesEtape1`, puis filtre sur `EtapeValidation='1'` si la colonne existe.
- `train_rf.py` s'appuie sur les tables `AnomaliesAIMS` et `AnomaliesMRO` produites par le pipeline IF.
- Les features métier incluent la fréquence des routes, la famille de l'appareil, et la durée de vol.

## Bonnes pratiques

- Exécutez d'abord `python pipeline.py` pour générer les données d'anomalies.
- Vérifiez les permissions SQL et l'accessibilité du serveur avant de lancer Streamlit.
- Adaptez `config.py` et les noms des tables si votre environnement SQL diffère.

## Licence

Ce projet est distribué sous la licence MIT. Voir le fichier `LICENSE` pour les détails.

## Contacts

Ce README est généré automatiquement à partir de la structure du projet. Pour toute question, consultez les scripts principaux : `pipeline.py`, `train.py`, `train_rf.py`, `predict.py`, `app.py`.
