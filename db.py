import logging
import urllib
import pandas as pd
from sqlalchemy import create_engine
from config import DB_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def get_engine():
    params = urllib.parse.quote_plus(
        f"DRIVER={{{DB_CONFIG['driver']}}};"
        f"SERVER={DB_CONFIG['server']};"
        f"DATABASE={DB_CONFIG['database']};"
        f"Trusted_Connection={DB_CONFIG['trusted_connection']};"
    )
    return create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

def read_table(table):
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
    logger.info(f"Table '{table}' : {len(df)} lignes.")
    return df

def read_query(sql):
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    logger.info(f"Requete : {len(df)} lignes.")
    return df

def write_table(df, table, if_exists="replace"):
    engine = get_engine()
    df.to_sql(table, engine, if_exists=if_exists, index=False)
    logger.info(f"Table '{table}' ecrite : {len(df)} lignes.")

print("Test connexion SQL Server...")

try:
    df = read_table("AIMS")
    print(f"  AIMS        : {len(df)} lignes")
except Exception as e:
    print(f"  Erreur AIMS : {e}")

try:
    df = read_table("MRO")
    print(f"  MRO         : {len(df)} lignes")
except Exception as e:
    print(f"  Erreur MRO  : {e}")

try:
    df = read_table("VolsValidesEtape1")
    print(f"  VolsValides : {len(df)} lignes")
except Exception as e:
    print(f"  Erreur VolsValides : {e}")