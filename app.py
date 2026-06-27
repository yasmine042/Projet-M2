# ============================================================================
#  app.py — Domestic Airlines · Plateforme de fiabilité des données
#  PFE : réconciliation AIMS / MRO + détection d'anomalies (IF / RF)
#
#  5 pages :  Accueil · Overview · Dashboard · Tables · Détection
#  Accueil  : couverture immersive (dégradé vert de marque).
#  Pages    : thème clair professionnel, palette multi-couleurs par section.
#
#  Lancement :  streamlit run app.py
#  Place ce fichier à la racine du projet (config.py, db.py, features.py,
#  predict.py, dossier models/, et assets/logo.png).
#  Sans base / modèles → bascule automatique sur données de DÉMONSTRATION.
# ============================================================================

import base64
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ─── Intégration au projet ──────────────────────────────────────────────────
try:
    from config import DB_CONFIG
    SQL_SERVER = DB_CONFIG.get("server", "DESKTOP-JC4TRUJ")
    SQL_DB     = DB_CONFIG.get("database", "TALEXPDWH")
except Exception:
    SQL_SERVER, SQL_DB = "DESKTOP-JC4TRUJ", "TALEXPDWH"

try:
    from db import read_query, read_table
    _HAS_DB = True
except Exception:
    _HAS_DB = False

APP_DIR = Path(__file__).parent
LOGO    = APP_DIR / "assets" / "logo.png"


def _get_engine():
    try:
        from db import get_engine
        return get_engine()
    except Exception:
        return None


def _ensure_traitement_table():
    if not _HAS_DB:
        return
    try:
        existing = read_query(
            "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
            "WHERE TABLE_NAME='TraitementAnomalies'")
        if existing.empty:
            from sqlalchemy import text
            engine = _get_engine()
            if engine:
                with engine.connect() as conn:
                    conn.execute(text("""
                        CREATE TABLE TraitementAnomalies (
                            ID INT,
                            Source VARCHAR(10),
                            DateTraitement DATETIME DEFAULT GETDATE(),
                            CONSTRAINT PK_Traitement PRIMARY KEY (ID, Source)
                        )"""))
                    conn.commit()
    except Exception:
        pass
    try:
        from db import get_engine
        return get_engine()
    except Exception:
        return None


def _mark_traitee(id_val, source):
    engine = _get_engine()
    if engine is None:
        return False
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text(
                "IF NOT EXISTS (SELECT 1 FROM TraitementAnomalies WHERE ID=:id AND Source=:src) "
                "INSERT INTO TraitementAnomalies (ID, Source) VALUES (:id, :src)"),
                {"id": int(id_val), "src": str(source)})
            conn.commit()
        return True
    except Exception as e:
        st.error(f"Erreur écriture SQL : {e}")
        return False


def _unmark_traitee(id_val, source):
    engine = _get_engine()
    if engine is None:
        return False
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text(
                "DELETE FROM TraitementAnomalies WHERE ID=:id AND Source=:src"),
                {"id": int(id_val), "src": str(source)})
            conn.commit()
        return True
    except Exception as e:
        st.error(f"Erreur suppression SQL : {e}")
        return False

# ============================================================================
#  PALETTE  (marque + palette catégorielle multi-couleurs)
# ============================================================================
OLIVE     = "#54942C"   # vert de marque (primaire)
OLIVE_D   = "#3f7320"
OLIVE_L   = "#eaf3e1"
GREEN     = "#5aa831"   # vert vif pour data-viz
BLUE      = "#3b6fe0"
TEAL      = "#10a59b"
AMBER     = "#e8920c"
VIOLET    = "#7c5cff"
ROSE      = "#e8475f"
CRIMSON   = "#D40C1C"   # rouge de marque — réservé au danger fort
PALETTE   = [BLUE, TEAL, AMBER, VIOLET, GREEN, ROSE, "#d96cc0", "#3aa0c9"]

BG        = "#f4f6f9"
CARD      = "#ffffff"
INK       = "#1f2a33"
MUTED     = "#7b8794"
BORDER    = "#e6eaef"
GRID      = "#eef1f5"

PAGES = [("home", '<i class="bi bi-house"></i> Accueil'),
         ("overview", '<i class="bi bi-speedometer2"></i> Vue globale'),
         ("dashboard", '<i class="bi bi-bar-chart-line"></i> Tableau de bord'),
         ("tables", '<i class="bi bi-table"></i> Tables'),
         ("detection", '<i class="bi bi-cpu"></i> Détection')]


def _tint(hexc: str, a: float = 0.14) -> str:
    h = hexc.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


# ============================================================================
#  CONNEXION / DONNÉES
# ============================================================================
@st.cache_data(ttl=60, show_spinner=False)
def db_alive() -> bool:
    if not _HAS_DB:
        return False
    try:
        df = read_query("SELECT 1 AS n")
        return df is not None and not df.empty
    except Exception:
        return False


def _sql(name: str, top: int | None = None) -> pd.DataFrame | None:
    if not db_alive():
        return None
    try:
        head = f"SELECT TOP {top} * FROM {name}" if top else f"SELECT * FROM {name}"
        return read_query(head)
    except Exception:
        return None


# ─── Démonstration (mêmes colonnes que les vraies tables) ───────────────────
_FLEET = ["7T-VCA","7T-VCB","7T-VCC","7T-VCD","7T-VCE","7T-VCF","7T-VCT",
          "7T-VCP","7T-VCQ","7T-VCR","7T-VCS","7T-VCL","7T-VCM","7T-VCN","7T-VCO"]
_APTS  = ["ALG","ORN","CZL","HME","IAM","TMR","AAE","GJL","TLM","BJA",
          "AZR","TFT","VVZ","RDN","DJG"]
_TYPES_ANOM = ["NumVol","AeroDepart","AeroArriv","Matricule","Date",
               "AeroDepart | AeroArriv","NumVol | Matricule"]


def _rng():
    return np.random.default_rng(2024)


@st.cache_data(show_spinner=False)
def _demo_core():
    rng = _rng()
    n = 1600
    dates = pd.to_datetime("2019-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    nv  = rng.integers(1000, 6500, n).astype(str)
    mat = rng.choice(_FLEET, n)
    dep = rng.choice(_APTS, n)
    arr = np.array([rng.choice([a for a in _APTS if a != d]) for d in dep])
    eh  = rng.integers(5, 22, n)
    dur = rng.integers(45, 180, n)
    return pd.DataFrame({
        "Date": dates, "NumVol": nv, "TypeVol": rng.choice(["REG","SUP"], n),
        "Matricule": mat, "TypeAeronef": rng.choice(["B738","AT72","B763"], n),
        "AeroDepart": dep, "AeroArriv": arr, "ETD": eh, "ETA": eh + dur//60,
        "ATD": eh, "ATA": eh + dur//60, "Block": dur, "PAX": rng.integers(40, 180, n),
        "Valider": rng.choice([1,1,1,1,0], n), "Anomalie": rng.choice([0]*9+[1], n),
        "IDAIMS": np.arange(1, n+1)})


def _demo_table(name: str) -> pd.DataFrame:
    rng = _rng()
    aims = _demo_core()
    if name == "AIMS":
        return aims
    if name == "MRO":
        core  = aims.sample(frac=0.94, random_state=1)        # ~ vols matchés
        extra = aims.sample(n=360, random_state=9)            # MRO-only (non-matchs)
        m = pd.concat([core, extra], ignore_index=True)
        bh = m["Block"].values
        return pd.DataFrame({
            "registr": m["Matricule"], "msn": rng.integers(100, 900, len(m)),
            "Date_Vol": m["Date"], "status": "OK", "flight no.": "SF" + m["NumVol"],
            "from": m["AeroDepart"], "bloc departure": m["ETD"], "take off": m["ETD"],
            "to": m["AeroArriv"], "landing": m["ETA"], "bloc arrival": m["ETA"],
            "bh": (bh/60).round(2), "bh_h": bh//60, "bh_m": bh%60,
            "PAX_PTCRMV": m["PAX"], "IDMRO": np.arange(1, len(m)+1)})
    if name == "VolsValidesEtape1":
        v = aims.sample(frac=0.94, random_state=2).reset_index(drop=True)
        et = rng.choice([1]*9 + [2,2,3,4,5], len(v))
        return pd.DataFrame({
            "IDAIMS": v["IDAIMS"], "IDMRO": np.arange(1, len(v)+1),
            "MatriculeAIMS": v["Matricule"], "NumVolAIMS": v["NumVol"],
            "AeroDepartAIMS": v["AeroDepart"], "AeroArrivAIMS": v["AeroArriv"],
            "DateAIMS": v["Date"], "EtapeValidation": et})
    if name in ("AnomaliesAIMS","AnomaliesMRO","FauxNegatifsAIMS","FauxNegatifsMRO"):
        is_mro = name.endswith("MRO"); is_fn = name.startswith("Faux")
        k = {"AnomaliesAIMS":92,"AnomaliesMRO":110,"FauxNegatifsAIMS":18,"FauxNegatifsMRO":23}[name]
        s = aims.sample(k, random_state=3 if not is_mro else 4).reset_index(drop=True)
        suf = "MRO" if is_mro else "AIMS"
        statut = (["Écart détecté"]*k if is_fn
                  else rng.choice(["Suspect","Suspect","Très Suspect"], k).tolist())
        typ = rng.choice(_TYPES_ANOM, k)
        score = -np.round(rng.uniform(0.45, 0.72, k), 4)
        raison = [f"{t.split(' | ')[0]} signalé pour vol {n_}" for t, n_ in zip(typ, s["NumVol"])]
        return pd.DataFrame({
            f"ID{suf}": np.arange(1, k+1),
            "jour_semaine": pd.to_datetime(s["Date"]).dt.dayofweek + 1,
            f"Matricule{suf}": s["Matricule"], f"NumVol{suf}": s["NumVol"],
            f"AeroDepart{suf}": s["AeroDepart"], f"AeroArriv{suf}": s["AeroArriv"],
            f"Date{suf}": s["Date"], "ScoreAnomalie": score, "Statut": statut,
            "RaisonAnomalie": raison, "TypeAnomalie": typ})
    if name in ("VoisNormalesAIMS", "VoisNormalesMRO"):
        is_mro = name.endswith("MRO")
        k_n = 55 if not is_mro else 65
        s = aims.sample(k_n, random_state=7 if not is_mro else 8).reset_index(drop=True)
        suf = "MRO" if is_mro else "AIMS"
        id_col = "IDMRO" if is_mro else "IDAIMS"
        sc = -np.round(rng.uniform(0.01, 0.38, k_n), 4)
        return pd.DataFrame({
            id_col: np.arange(1, k_n + 1),
            "jour_semaine": pd.to_datetime(s["Date"]).dt.dayofweek + 1,
            f"Matricule{suf}": s["Matricule"], f"NumVol{suf}": s["NumVol"],
            f"AeroDepart{suf}": s["AeroDepart"], f"AeroArriv{suf}": s["AeroArriv"],
            f"Date{suf}": s["Date"], "ScoreAnomalie": sc,
            "Statut": "Normal", "RaisonAnomalie": "Aucune", "TypeAnomalie": "Aucune"})
    return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load(name: str, top: int | None = None) -> pd.DataFrame:
    df = _sql(name, top)
    return df if (df is not None and not df.empty) else _demo_table(name)


def harmonize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return df.assign(Source=source)
    idc = "IDMRO" if source == "MRO" else "IDAIMS"
    ren = {f"Matricule{source}": "Matricule", f"NumVol{source}": "NumVol",
           f"AeroDepart{source}": "AeroDepart", f"AeroArriv{source}": "AeroArriv",
           f"Date{source}": "Date", idc: "ID"}
    out = df.rename(columns={k: v for k, v in ren.items() if k in df.columns}).copy()
    out["Source"] = source
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    return out


@st.cache_data(ttl=300, show_spinner=False)
def load_anomalies() -> pd.DataFrame:
    a = harmonize(load("AnomaliesAIMS"), "AIMS")
    m = harmonize(load("AnomaliesMRO"),  "MRO")
    cols = ["ID","Source","Matricule","NumVol","AeroDepart","AeroArriv","Date",
            "ScoreAnomalie","Statut","RaisonAnomalie","TypeAnomalie"]
    return pd.concat([a, m], ignore_index=True).reindex(columns=cols)


@st.cache_data(ttl=300, show_spinner=False)
def load_faux_negatifs() -> pd.DataFrame:
    a = harmonize(load("FauxNegatifsAIMS"), "AIMS")
    m = harmonize(load("FauxNegatifsMRO"),  "MRO")
    return pd.concat([a, m], ignore_index=True)


@st.cache_data(ttl=300, show_spinner=False)
def load_all_anomalies() -> pd.DataFrame:
    anom = load_anomalies().copy()
    fn   = load_faux_negatifs().copy()
    if "Statut" in fn.columns:
        fn["Statut"] = "Écart détecté"
    cols = ["ID", "Source", "Matricule", "NumVol", "AeroDepart", "AeroArriv",
            "Date", "ScoreAnomalie", "Statut", "RaisonAnomalie", "TypeAnomalie"]
    combined = pd.concat([anom, fn], ignore_index=True)
    combined = combined.reindex(columns=[c for c in cols if c in combined.columns])
    if "Date" in combined.columns:
        combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
        combined["jour_semaine"] = combined["Date"].dt.dayofweek
        combined["mois_num"] = combined["Date"].dt.month
    if db_alive():
        try:
            tr = read_query("SELECT ID, Source, DateTraitement FROM TraitementAnomalies")
            if not tr.empty and "ID" in combined.columns and "Source" in combined.columns:
                tr["Traitee"] = 1
                combined = combined.merge(tr[["ID", "Source", "Traitee", "DateTraitement"]],
                                          on=["ID", "Source"], how="left")
                combined["Traitee"] = combined["Traitee"].fillna(0).astype(int)
                return combined
        except Exception:
            pass
    combined["Traitee"] = 0
    combined["DateTraitement"] = pd.NaT
    return combined


@st.cache_data(ttl=300, show_spinner=False)
def load_vrais_normaux() -> pd.DataFrame:
    na = harmonize(load("VoisNormalesAIMS"), "AIMS")
    nm = harmonize(load("VoisNormalesMRO"),  "MRO")
    normaux = pd.concat([na, nm], ignore_index=True)
    fn = load_faux_negatifs()
    if normaux.empty or fn.empty:
        return normaux
    keys = ["Source", "NumVol", "Matricule", "AeroDepart", "AeroArriv"]
    avail = [c for c in keys if c in normaux.columns and c in fn.columns]
    if avail:
        nk = normaux[avail].astype(str).apply(tuple, axis=1)
        fk = fn[avail].astype(str).apply(tuple, axis=1)
        return normaux[~nk.isin(set(fk))].reset_index(drop=True)
    return normaux


@st.cache_data(show_spinner=False)
def load_rf_metrics():
    try:
        import json
        with open(APP_DIR / "models" / "rf_metrics.json") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def global_kpis() -> dict:
    aims, mro, vv = load("AIMS"), load("MRO"), load("VolsValidesEtape1")
    all_anom = load_all_anomalies()
    fn = load_faux_negatifs()
    total_aims, total_mro, valides = len(aims), len(mro), len(vv)
    etapes = {}
    if "EtapeValidation" in vv.columns and not vv.empty:
        etapes = {int(k): int(v) for k, v in vv["EtapeValidation"].value_counts().items()
                  if str(k).strip().isdigit()}
    a_aims = int((all_anom["Source"] == "AIMS").sum()) if not all_anom.empty else 0
    a_mro  = int((all_anom["Source"] == "MRO").sum()) if not all_anom.empty else 0
    anom_if = int((all_anom["Détection"] == "Isolation Forest").sum()) if "Détection" in all_anom.columns else 0
    anom_rf = int((all_anom["Détection"] == "Random Forest").sum()) if "Détection" in all_anom.columns else 0
    taux   = 100 * valides / total_aims if total_aims else 0.0
    if "IDAIMS" in aims.columns and "IDAIMS" in vv.columns:
        nm_aims = int((~aims["IDAIMS"].isin(set(vv["IDAIMS"].dropna()))).sum())
    else:
        nm_aims = max(0, total_aims - valides)
    if "IDMRO" in mro.columns and "IDMRO" in vv.columns:
        nm_mro = int((~mro["IDMRO"].isin(set(vv["IDMRO"].dropna()))).sum())
    else:
        nm_mro = max(0, total_mro - valides)
    norm_aims = max(0, nm_aims - a_aims)
    norm_mro  = max(0, nm_mro  - a_mro)
    return {"total_aims": total_aims, "total_mro": total_mro, "valides": valides,
            "etapes": etapes, "anom_aims": a_aims, "anom_mro": a_mro,
            "anom_total": a_aims + a_mro, "anom_if": anom_if, "anom_rf": anom_rf,
            "faux_neg": len(fn), "taux": taux,
            "nm_aims": nm_aims, "nm_mro": nm_mro,
            "norm_aims": norm_aims, "norm_mro": norm_mro}


# ============================================================================
#  FIGURES PLOTLY  (thème clair, texte foncé lisible)
# ============================================================================
def _light(fig, h=260):
    fig.update_layout(
        height=h, margin=dict(t=18, b=8, l=8, r=14),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=12, family="Inter, Segoe UI, sans-serif"),
        showlegend=False, hoverlabel=dict(bgcolor="#fff", font_size=12))
    fig.update_xaxes(showgrid=False, color=MUTED)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, color=MUTED, zeroline=False)
    return fig


def ring(value, label, color=OLIVE, h=230):
    value = max(0, min(100, value))
    fig = go.Figure(go.Pie(
        values=[value, 100 - value], hole=0.76, sort=False, direction="clockwise",
        marker=dict(colors=[color, "#edf1f5"], line=dict(color=CARD, width=3)),
        textinfo="none", hoverinfo="skip"))
    fig.update_layout(
        height=h, margin=dict(t=4, b=4, l=4, r=4), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(text=f"{value:.1f}%", x=.5, y=.54, showarrow=False,
                          font=dict(size=33, color=INK, family="Arial Black")),
                     dict(text=label, x=.5, y=.36, showarrow=False,
                          font=dict(size=12, color=MUTED))])
    return fig


def vbar(labels, values, color=BLUE, h=260):
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker=dict(color=color), text=values,
        textposition="outside", textfont=dict(color=INK, size=11),
        hovertemplate="%{x}: %{y}<extra></extra>"))
    return _light(fig, h)


def cbar(labels, values, h=300, palette=None):
    palette = palette or PALETTE
    colors = [palette[i % len(palette)] for i in range(len(labels))]
    fig = go.Figure(go.Bar(
        x=values, y=[str(l) for l in labels], orientation="h",
        marker=dict(color=colors), text=values, textposition="outside",
        textfont=dict(color=INK, size=11), hovertemplate="%{y}: %{x}<extra></extra>"))
    fig.update_layout(yaxis=dict(autorange="reversed", type="category"))
    return _light(fig, h)


def area_trend(x, y, color=BLUE, h=300):
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="lines+markers", line=dict(color=color, width=3, shape="spline"),
        marker=dict(size=6, color=color), fill="tozeroy", fillcolor=_tint(color, 0.14),
        hovertemplate="%{x}: %{y}<extra></extra>"))
    return _light(fig, h)


def donut(labels, values, colors, h=240, center=None, sub=None):
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62, sort=False,
        marker=dict(colors=colors, line=dict(color=CARD, width=2)),
        textinfo="percent", textfont=dict(color="#fff", size=12),
        hovertemplate="%{label}: %{value}<extra></extra>"))
    ann = []
    if center is not None:
        ann.append(dict(text=center, x=.5, y=.54, showarrow=False,
                        font=dict(size=23, color=INK, family="Arial Black")))
        ann.append(dict(text=sub or "", x=.5, y=.40, showarrow=False,
                        font=dict(size=11, color=MUTED)))
    fig.update_layout(
        height=h, margin=dict(t=8, b=8, l=8, r=8), paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True, legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center",
                                     font=dict(color=INK, size=11)),
        annotations=ann)
    return fig


def split_trend(x, y_aims, y_mro, h=280):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=y_aims, name="AIMS", marker_color=BLUE))
    fig.add_trace(go.Bar(x=x, y=y_mro, name="MRO", marker_color=AMBER))
    fig.update_layout(barmode="group", showlegend=True,
                      legend=dict(orientation="h", y=1.16, x=0, font=dict(color=INK)))
    return _light(fig, h)


# ── Graphiques avancés ─────────────────────────────────────────────────────
def sankey_flow(k, anom_df):
    n_suspect = n_tres = n_fn = 0
    if not anom_df.empty:
        if "Statut" in anom_df.columns:
            n_suspect = int((anom_df["Statut"] == "Suspect").sum())
            n_tres    = int((anom_df["Statut"] == "Très Suspect").sum())
            n_fn      = int((anom_df["Statut"] == "Écart détecté").sum())
    labels = ["AIMS", "MRO", "Réconciliés", "Écarts AIMS", "Écarts MRO",
              "Normaux", "Anomalies", "Suspects", "Très Suspects",
              "Écarts détectés (RF)"]
    node_colors = [BLUE, TEAL, GREEN, AMBER, AMBER,
                   "#5aa831", ROSE, "#e8920c", CRIMSON, VIOLET]
    source = [0, 0, 1, 1, 3, 3, 4, 4, 6, 6, 5]
    target = [2, 3, 2, 4, 5, 6, 5, 6, 7, 8, 9]
    value = [k["valides"], k["nm_aims"], k["valides"], k["nm_mro"],
             k["norm_aims"], k.get("anom_if", 0) * k["anom_aims"] // max(k["anom_total"], 1),
             k["norm_mro"],  k.get("anom_if", 0) * k["anom_mro"]  // max(k["anom_total"], 1),
             n_suspect, n_tres, n_fn]
    link_colors = [_tint(node_colors[s], 0.35) for s in source]
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(pad=20, thickness=22, line=dict(color="#fff", width=1.5),
                  label=labels, color=node_colors),
        link=dict(source=source, target=target, value=value, color=link_colors)))
    fig.update_layout(
        height=400, margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=12, family="Inter, Segoe UI, sans-serif"))
    return fig


_JOURS_HM = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
_MOIS_HM  = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
             "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]


def heatmap_chart(df, h=340):
    d = df.dropna(subset=["Date"]).copy()
    if d.empty:
        return None, None
    d["mois"] = d["Date"].dt.month
    d["jour"] = d["Date"].dt.dayofweek
    pivot = d.groupby(["jour", "mois"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(index=range(7), columns=range(1, 13), fill_value=0)
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=_MOIS_HM, y=_JOURS_HM,
        colorscale=[[0, "#ffffff"], [0.5, "#ffd580"], [1, "#d62728"]],
        hovertemplate="%{y} %{x} : %{z} anomalies de saisie<extra></extra>",
        showscale=False))
    fig.update_layout(
        height=h, margin=dict(t=20, b=20, l=60, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=12, family="Inter, Segoe UI, sans-serif"),
        yaxis=dict(autorange="reversed", fixedrange=True),
        xaxis=dict(side="top", fixedrange=True))
    return fig, pivot


def score_histogram(df, h=310):
    fig = go.Figure()
    for status, color in [("Suspect", AMBER), ("Très Suspect", CRIMSON),
                          ("Écart détecté", VIOLET)]:
        sub = df[df["Statut"] == status]["ScoreAnomalie"].dropna() if "Statut" in df.columns else pd.Series(dtype=float)
        if not sub.empty:
            fig.add_trace(go.Histogram(
                x=sub, nbinsx=20, name=status, opacity=0.78,
                marker=dict(color=_tint(color, 0.65), line=dict(color=color, width=1)),
                hovertemplate=f"{status}<br>Score : %{{x:.4f}}<br>Nb : %{{y}}<extra></extra>"))
    fig.update_layout(barmode="overlay", showlegend=True,
                      legend=dict(orientation="h", y=1.12, x=0, font=dict(color=INK, size=11)))
    return _light(fig, h)


def radar_quality(metrics, h=340):
    cats = list(metrics.keys())
    vals = list(metrics.values()) + [list(metrics.values())[0]]
    cats_c = cats + [cats[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats_c, fill="toself",
        fillcolor=_tint(OLIVE, 0.18), line=dict(color=OLIVE, width=2.5),
        marker=dict(size=7, color=OLIVE),
        hovertemplate="%{theta} : %{r:.1f}%<extra></extra>"))
    fig.update_layout(
        height=h, margin=dict(t=40, b=40, l=70, r=70),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=11, family="Inter, Segoe UI, sans-serif"),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%",
                            gridcolor=GRID, linecolor=BORDER),
            angularaxis=dict(gridcolor=GRID, linecolor=BORDER)))
    return fig


def funnel_chart(labels, values, h=320):
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    fig = go.Figure(go.Funnel(
        y=labels, x=values, marker=dict(color=colors),
        textposition="inside", textinfo="value+percent initial",
        textfont=dict(color="#fff", size=13),
        hovertemplate="%{y} : %{x}<extra></extra>",
        connector=dict(line=dict(color=BORDER, width=1))))
    fig.update_layout(
        height=h, margin=dict(t=20, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=12, family="Inter, Segoe UI, sans-serif"))
    return fig


def gauge_chart(value, title, min_val=-0.7, max_val=0.3, threshold=0, h=210):
    bar_color = GREEN if value >= threshold else CRIMSON
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number=dict(font=dict(size=24, color=INK), valueformat=".4f"),
        title=dict(text=title, font=dict(size=13, color=MUTED)),
        gauge=dict(
            axis=dict(range=[min_val, max_val], tickcolor=MUTED),
            bar=dict(color=bar_color, thickness=0.65),
            bgcolor="#f8faf5", borderwidth=1, bordercolor=BORDER,
            steps=[dict(range=[min_val, threshold], color=_tint(CRIMSON, 0.08)),
                   dict(range=[threshold, max_val], color=_tint(GREEN, 0.08))],
            threshold=dict(line=dict(color=CRIMSON, width=3),
                           thickness=0.8, value=threshold))))
    fig.update_layout(
        height=h, margin=dict(t=50, b=10, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, family="Inter, Segoe UI, sans-serif"))
    return fig


_FEATURE_LABELS = {
    "NumVolNum": "N° de vol", "FleetFamilyCode": "Famille appareil",
    "FreqVolJour": "Fréq. vol/jour", "FreqVolFamille": "Fréq. vol/famille",
    "FreqRouteJour": "Fréq. route/jour", "FreqVolRoute": "Fréq. vol/route",
    "FreqVolRouteJour": "Fréq. vol/route/jour", "FreqComboFamille": "Combinaison complète",
}


def feature_importance_bar(importances, h=280):
    labs = [_FEATURE_LABELS.get(k, k) for k in importances]
    vals = list(importances.values())
    idx = sorted(range(len(vals)), key=lambda i: vals[i])
    fig = go.Figure(go.Bar(
        x=[vals[i] for i in idx], y=[labs[i] for i in idx], orientation="h",
        marker=dict(color=[PALETTE[i % len(PALETTE)] for i in range(len(idx))]),
        text=[f"{vals[i]:.1%}" for i in idx], textposition="outside",
        textfont=dict(color=INK, size=11),
        hovertemplate="%{y} : %{x:.3f}<extra></extra>"))
    return _light(fig, h)


# ============================================================================
#  STYLES GLOBAUX  (clair)
# ============================================================================
def css():
    st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css">',
                unsafe_allow_html=True)
    st.markdown(f"""
    <style>
      .stApp {{ background:{BG}; }}
      #MainMenu, footer, header {{ visibility:hidden; }}
      section[data-testid="stSidebar"] {{ display:none; }}
      .block-container {{ padding:1.2rem 1.6rem 2.4rem; max-width:1480px; }}
      html, body, [class*="css"] {{ font-family:'Inter','Segoe UI',sans-serif; }}

      /* barre d'application */
      .appbar {{ background:{CARD}; border:1px solid {BORDER}; border-radius:20px;
                 padding:13px 18px 11px; box-shadow:0 4px 18px rgba(20,40,25,.05);
                 margin-bottom:18px; }}
      .ab-top {{ display:flex; justify-content:space-between; align-items:center; gap:16px; }}
      .ab-brand {{ display:flex; align-items:center; gap:14px; }}
      .ab-logobox {{ background:#fff; border:1px solid {BORDER}; border-radius:12px;
                     padding:6px 12px; display:flex; align-items:center; }}
      .ab-logo {{ height:30px; display:block; }}
      .ab-title {{ color:{INK}; font-weight:700; font-size:16px; line-height:1.15; }}
      .ab-sub {{ color:{MUTED}; font-size:11px; }}
      .ab-actions {{ display:flex; align-items:center; gap:12px; }}
      .ab-status {{ font-size:11px; padding:6px 13px; border-radius:20px; font-weight:600;
                    white-space:nowrap; }}
      .ab-status.ok {{ background:{OLIVE_L}; color:{OLIVE_D}; }}
      .ab-status.demo {{ background:#fff3d6; color:#9a7400; }}
      .ab-refresh {{ text-decoration:none !important; color:{MUTED} !important;
                     border:1px solid {BORDER}; border-radius:50%; width:34px; height:34px;
                     display:inline-flex; align-items:center; justify-content:center;
                     font-size:16px; transition:all .15s; }}
      .ab-refresh:hover {{ color:{OLIVE} !important; border-color:{OLIVE}; }}
      .ab-nav {{ display:flex; gap:8px; margin-top:11px; }}
      .ab-pill {{ flex:1; text-align:center; text-decoration:none !important;
                  padding:10px 14px; border-radius:14px; font-size:13.5px; font-weight:600;
                  color:{MUTED} !important; border:1px solid transparent; transition:all .15s; }}
      .ab-pill:hover {{ background:{OLIVE_L}; color:{OLIVE_D} !important; }}
      .ab-pill.active {{ background:{OLIVE}; color:#fff !important;
                         box-shadow:0 5px 14px rgba(84,148,44,.30); }}

      /* bandeau de titre */
      .pagehead {{ display:flex; align-items:center; gap:14px; margin:2px 0 16px; }}
      .ph-ic {{ width:48px; height:48px; border-radius:14px; color:#fff;
                background:linear-gradient(135deg,{OLIVE},{OLIVE_D});
                display:flex; align-items:center; justify-content:center; font-size:22px;
                box-shadow:0 6px 16px rgba(84,148,44,.28); }}
      .ph-title {{ color:{INK}; font-size:23px; font-weight:800; line-height:1.1; }}
      .ph-desc {{ color:{MUTED}; font-size:13px; margin-top:2px; }}

      /* sous-section titrée */
      .section {{ display:flex; align-items:center; gap:11px; margin:20px 0 12px; }}
      .sec-ic {{ width:38px; height:38px; border-radius:11px; background:#fff;
                 border:1px solid {BORDER}; display:flex; align-items:center;
                 justify-content:center; box-shadow:0 2px 8px rgba(20,40,25,.05); }}
      .sec-t {{ color:{INK}; font-size:18px; font-weight:800; }}

      /* cartes blanches */
      [data-testid="stVerticalBlockBorderWrapper"] {{
          background:{CARD}; border:1px solid {BORDER} !important; border-radius:18px;
          padding:16px 18px; box-shadow:0 2px 12px rgba(20,40,25,.04); }}
      .ttl {{ color:{INK}; font-size:15px; font-weight:700; margin-bottom:6px; }}
      .sub {{ color:{MUTED}; font-size:12px; }}

      /* stat cards — accent couleur par carte */
      .stat {{ background:{CARD}; border:1px solid {BORDER}; border-left-width:4px;
               border-radius:14px; padding:15px 16px; box-shadow:0 2px 10px rgba(20,40,25,.04);
               display:flex; gap:13px; align-items:center; }}
      .stat .st-ic {{ width:46px; height:46px; flex:0 0 46px; border-radius:13px;
                      display:flex; align-items:center; justify-content:center; font-size:21px; }}
      .stat .lab {{ color:{MUTED}; font-size:12px; }}
      .stat .val {{ color:{INK}; font-size:25px; font-weight:800; line-height:1.12; margin-top:1px; }}
      .stat .dl  {{ font-size:11px; margin-top:2px; }}

      /* verdict */
      .verdict {{ border-radius:18px; padding:20px 22px; color:#fff; text-align:center;
                  box-shadow:0 8px 22px rgba(20,40,25,.10); }}
      .verdict .big {{ font-size:23px; font-weight:800; }}
      .verdict.ok  {{ background:linear-gradient(135deg,{GREEN},{OLIVE_D}); }}
      .verdict.bad {{ background:linear-gradient(135deg,{ROSE},{CRIMSON}); }}
      .verdict.na  {{ background:linear-gradient(135deg,#9aa6b2,#79858f); }}

      .pill {{ display:inline-block; padding:4px 11px; border-radius:12px;
               font-size:11px; font-weight:600; }}
      .pill.g {{ background:{OLIVE_L}; color:{OLIVE_D}; }}
      .pill.r {{ background:{_tint(ROSE,.16)}; color:{CRIMSON}; }}

      div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input,
      .stDateInput input {{ border-radius:12px !important; }}
      .stDataFrame {{ border-radius:14px; overflow:hidden; }}
      .stButton>button {{ border-radius:12px; font-weight:600; }}
      .stButton>button[kind="primary"] {{ background:{OLIVE}; border-color:{OLIVE}; }}
      .stDownloadButton>button {{ border-radius:12px; border:1px solid {BORDER};
                                  background:{CARD}; color:{OLIVE_D}; font-weight:600; }}
      .stDownloadButton>button:hover {{ border-color:{OLIVE}; }}

      /* ── animations & transitions ── */
      @keyframes fadeSlideUp {{
          from {{ opacity: 0; transform: translateY(16px); }}
          to   {{ opacity: 1; transform: translateY(0); }}
      }}
      .stat {{ animation: fadeSlideUp .45s ease-out both;
               transition: transform .2s ease, box-shadow .2s ease; }}
      .stat:hover {{ transform: translateY(-3px);
                     box-shadow: 0 8px 24px rgba(20,40,25,.12); }}
      [data-testid="stVerticalBlockBorderWrapper"] {{
          animation: fadeSlideUp .5s ease-out both;
          transition: transform .2s ease, box-shadow .2s ease; }}
      [data-testid="stVerticalBlockBorderWrapper"]:hover {{
          box-shadow: 0 6px 22px rgba(20,40,25,.08); }}
      .pagehead {{ animation: fadeSlideUp .35s ease-out; }}
      .section  {{ animation: fadeSlideUp .4s ease-out; }}
      .verdict  {{ transition: transform .2s ease, box-shadow .2s ease; }}
      .verdict:hover {{ transform: scale(1.02);
                        box-shadow: 0 12px 28px rgba(20,40,25,.15); }}
      .ph-ic {{ transition: transform .3s ease; }}
      .ph-ic:hover {{ transform: scale(1.08) rotate(-3deg); }}
      a, a:hover, a:visited {{ text-decoration:none !important; color:inherit !important; }}
      a .stat {{ cursor:pointer; }}
      a .stat, a .stat * {{ text-decoration:none !important; }}
      a:hover .stat {{ border-left-width:5px; transform:translateY(-3px);
                       box-shadow:0 8px 24px rgba(20,40,25,.14); }}
    </style>""", unsafe_allow_html=True)


def stat(label, value, delta="", color=BLUE, icon=None):
    if icon and icon.strip().startswith("<"):
        inner = icon                                    # image HTML (<img …>)
    elif icon:
        inner = f'<span style="font-size:21px;line-height:1;">{icon}</span>'
    else:
        inner = ""
    ic = (f'<div class="st-ic" style="background:{_tint(color)};color:{color};">{inner}</div>'
          if icon else "")
    return (f'<div class="stat" style="border-left-color:{color};">{ic}'
            f'<div><div class="lab">{label}</div>'
            f'<div class="val">{value}</div>'
            f'<div class="dl" style="color:{color};">{delta}</div></div></div>')


def stat_clickable(label, value, delta, color, icon, href):
    card = stat(label, value, delta, color, icon)
    return (f'<a href="{href}" target="_self" '
            f'style="text-decoration:none !important;color:inherit !important;display:block;">'
            f'{card}</a>')


def grid(cards, cols=4):
    return (f'<div style="display:grid;grid-template-columns:repeat({cols},1fr);'
            f'gap:13px;">{"".join(cards)}</div>')


def fnum(x):
    try:
        return f"{int(x):,}".replace(",", " ")
    except Exception:
        return str(x)


# ============================================================================
#  BARRE D'APPLICATION + BANDEAU
# ============================================================================
def _logo_b64() -> str | None:
    try:
        return base64.b64encode(LOGO.read_bytes()).decode()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _icon_b64(name: str) -> str | None:
    try:
        return base64.b64encode((APP_DIR / "assets" / "icons" / f"{name}.png").read_bytes()).decode()
    except Exception:
        return None


def icon_img(name: str, size: int = 26) -> str:
    b = _icon_b64(name)
    return (f'<img src="data:image/png;base64,{b}" style="width:{size}px;height:{size}px;'
            f'object-fit:contain;display:block;"/>') if b else ""


def section(title: str, icon_name: str | None = None):
    chip = f'<div class="sec-ic">{icon_img(icon_name, 22)}</div>' if icon_name else ""
    st.markdown(f'<div class="section">{chip}<div class="sec-t">{title}</div></div>',
                unsafe_allow_html=True)


def appbar(current: str):
    b64  = _logo_b64()
    logo = (f'<div class="ab-logobox"><img class="ab-logo" src="data:image/png;base64,{b64}"/></div>'
            if b64 else f'<span style="color:{OLIVE};font-weight:800;">DOMESTIC AIRLINES</span>')
    status = ('<span class="ab-status ok">● SQL Server connecté</span>' if db_alive()
              else '<span class="ab-status demo">● Données de démonstration</span>')
    pills = "".join(
        f'<a class="ab-pill {"active" if k == current else ""}" href="?nav={k}" target="_self">{lab}</a>'
        for k, lab in PAGES)
    st.markdown(f"""
      <div class="appbar">
        <div class="ab-top">
          <div class="ab-brand">{logo}
            <div><div class="ab-title">Plateforme de fiabilité des données</div>
            <div class="ab-sub">{SQL_SERVER} / {SQL_DB}</div></div>
          </div>
          <div class="ab-actions">{status}
            <a class="ab-refresh" href="?nav={current}&refresh=1" target="_self"
               title="Rafraîchir">↻</a></div>
        </div>
        <div class="ab-nav">{pills}</div>
      </div>""", unsafe_allow_html=True)


def pagehead(title, desc, icon=""):
    st.markdown(f'<div class="pagehead"><div class="ph-ic">{icon}</div>'
                f'<div><div class="ph-title">{title}</div>'
                f'<div class="ph-desc">{desc}</div></div></div>', unsafe_allow_html=True)


# ============================================================================
#  PAGE — ACCUEIL (couverture immersive verte — inchangée)
# ============================================================================
_NAV_ICONS = {
    "overview": ('<svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" '
                 'stroke-linecap="round"><line x1="6" y1="20" x2="6" y2="11"/>'
                 '<line x1="12" y1="20" x2="12" y2="5"/><line x1="18" y1="20" x2="18" y2="14"/></svg>'),
    "dashboard": ('<svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" '
                  'stroke-linecap="round" stroke-linejoin="round">'
                  '<polyline points="3,16 9,10 13,14 21,5"/>'
                  '<circle cx="21" cy="5" r="1.5" fill="#fff" stroke="none"/></svg>'),
    "tables": ('<svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2">'
               '<rect x="3" y="4" width="18" height="16" rx="2"/>'
               '<line x1="3" y1="10" x2="21" y2="10"/><line x1="9" y1="4" x2="9" y2="20"/></svg>'),
    "detection": ('<svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" '
                  'stroke-linecap="round"><circle cx="11" cy="11" r="7"/>'
                  '<line x1="21" y1="21" x2="16.5" y2="16.5"/></svg>'),
}

_HERO_SVG = """
<svg viewBox="0 0 460 300" width="100%" style="max-width:440px;" xmlns="http://www.w3.org/2000/svg">
  <defs><radialGradient id="glow" cx="52%" cy="42%" r="55%">
    <stop offset="0%" stop-color="#ffffff" stop-opacity="0.30"/>
    <stop offset="100%" stop-color="#ffffff" stop-opacity="0"/></radialGradient></defs>
  <circle cx="250" cy="150" r="128" fill="url(#glow)"/>
  <circle cx="250" cy="150" r="120" fill="none" stroke="#fff" stroke-opacity="0.22"/>
  <circle cx="250" cy="150" r="82"  fill="none" stroke="#fff" stroke-opacity="0.15"/>
  <path d="M70,232 Q250,38 432,150" fill="none" stroke="#fff" stroke-opacity="0.6"
        stroke-width="2" stroke-dasharray="2 7" stroke-linecap="round"/>
  <circle cx="70" cy="232" r="6" fill="#fff"/>
  <circle cx="432" cy="150" r="6" fill="#D40C1C"/>
  <text x="52" y="254" fill="#fff" font-size="12" font-family="sans-serif" opacity="0.9">AIMS</text>
  <text x="408" y="136" fill="#fff" font-size="12" font-family="sans-serif" opacity="0.9">MRO</text>
  <g transform="translate(232,58) rotate(78) scale(1.7)">
    <path d="M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z" fill="#fff"/>
  </g>
</svg>
"""

_HOME_CSS = """
<style>
  .stApp {
    background:
      radial-gradient(circle at 84% 6%, rgba(255,255,255,.20), transparent 40%),
      radial-gradient(circle at 4% 96%, rgba(63,115,32,.55), transparent 46%),
      linear-gradient(135deg, #6cb33f 0%, #54942C 46%, #3f7320 100%) !important;
  }
  .block-container { padding-top:2.4rem !important; max-width:1320px; }
  .cover { display:flex; gap:34px; flex-wrap:wrap; align-items:stretch; min-height:84vh; }
  .brand { flex:1 1 500px; color:#fff; display:flex; flex-direction:column; justify-content:center; }
  .logo-chip { background:#fff; border-radius:16px; padding:12px 18px; width:max-content;
               box-shadow:0 10px 26px rgba(0,0,0,.16); }
  .logo-chip img { height:40px; display:block; }
  .brand h1 { font-size:clamp(40px,5.2vw,66px); font-weight:800; margin:24px 0 0;
              letter-spacing:.5px; line-height:1.02; color:#fff; }
  .accent-bar { width:92px; height:5px; background:#D40C1C; border-radius:4px; margin:16px 0 16px; }
  .brand .lead { font-size:18px; opacity:.96; max-width:500px; font-weight:500; }
  .brand .desc { font-size:14px; opacity:.86; max-width:500px; margin-top:14px; line-height:1.65; }
  .brand .foot { margin-top:26px; opacity:.82; font-size:13px;
                 border-top:1px solid rgba(255,255,255,.22); padding-top:16px; width:max-content; }
  .illu { margin-top:8px; }
  .navcol { flex:1 1 370px; display:flex; flex-direction:column; gap:15px; justify-content:center; }
  .navhead { color:#fff; opacity:.85; font-size:12px; letter-spacing:2px;
             text-transform:uppercase; margin-bottom:2px; }
  .navcard, .navcard:link, .navcard:visited, .navcard * {
      color:#fff !important; text-decoration:none !important; }
  .navcard { display:flex; align-items:center; gap:16px;
             background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.22);
             border-radius:18px; padding:17px 20px;
             -webkit-backdrop-filter:blur(6px); backdrop-filter:blur(6px);
             transition:transform .18s ease, background .18s ease, box-shadow .18s ease; }
  .navcard:hover { background:rgba(255,255,255,.22); transform:translateY(-3px);
                   box-shadow:0 14px 30px rgba(0,0,0,.20); }
  .navcard .ico { width:48px; height:48px; flex:0 0 48px; border-radius:14px;
                  background:rgba(255,255,255,.18); display:flex; align-items:center;
                  justify-content:center; }
  .navcard .ico svg { width:24px; height:24px; }
  .navcard .txt h3 { margin:0; font-size:18px; font-weight:700; }
  .navcard .txt p  { margin:3px 0 0; font-size:12.5px; opacity:.9; line-height:1.4; }
  .navcard .arrow  { margin-left:auto; font-size:22px; opacity:.65;
                     transition:opacity .18s, transform .18s; }
  .navcard:hover .arrow { opacity:1; transform:translateX(3px); }
</style>
"""

_HOME_TILES = [
    ("overview",  "Vue globale",
     "État de santé des données aériennes : vols réconciliés, anomalies détectées et alertes opérationnelles."),
    ("dashboard", "Tableau de bord",
     "Investigation et suivi des anomalies : filtrez, analysez et marquez les écarts de saisie traités."),
    ("tables",    "Tables",
     "Consultation des données sources : vols opérations, vols maintenance et résultats de l'analyse."),
    ("detection", "Détection",
     "Analyser un vol spécifique : saisissez ses informations et obtenez un diagnostic immédiat."),
]


def page_home():
    st.markdown(_HOME_CSS, unsafe_allow_html=True)
    b64 = _logo_b64()
    logo_html = (f'<div class="logo-chip"><img src="data:image/png;base64,{b64}"/></div>'
                 if b64 else '<h2 style="color:#fff;margin:0;">DOMESTIC AIRLINES</h2>')
    cards = "".join(
        f'<a class="navcard" href="?nav={key}" target="_self">'
        f'<div class="ico">{_NAV_ICONS[key]}</div>'
        f'<div class="txt"><h3>{title}</h3><p>{desc}</p></div>'
        f'<div class="arrow">→</div></a>' for key, title, desc in _HOME_TILES)
    st.markdown(f"""
        <div class="cover">
          <div class="brand">
            {logo_html}
            <h1>Domestic Airlines</h1>
            <div class="accent-bar"></div>
            <div class="lead">Plateforme de contrôle qualité des données de vol</div>
            <div class="desc">Rapprochement automatique des vols entre les systèmes
              d'opérations et de maintenance, détection intelligente des écarts
              de saisie et suivi des anomalies non traitées.</div>
            <div class="illu">{_HERO_SVG}</div>
          </div>
          <div class="navcol">
            <div class="navhead">Explorer la plateforme</div>
            {cards}
          </div>
        </div>""", unsafe_allow_html=True)


# ============================================================================
#  PAGE — OVERVIEW
# ============================================================================
def page_overview():
    k = global_kpis()
    anom = load_all_anomalies()

    non_traitees = anom[anom["Traitee"] == 0] if "Traitee" in anom.columns else anom
    # Dernier mois disponible dans les données (pas datetime.today())
    if not non_traitees.empty and "Date" in non_traitees.columns and non_traitees["Date"].notna().any():
        dernier_mois = non_traitees["Date"].dropna().dt.to_period("M").max()
        anom_ce_mois = non_traitees[non_traitees["Date"].dt.to_period("M") == dernier_mois]
        lbl_mois = str(dernier_mois)
    else:
        anom_ce_mois = pd.DataFrame()
        lbl_mois = "—"
    critiques = non_traitees[non_traitees["Statut"] == "Très Suspect"] if not non_traitees.empty else pd.DataFrame()

    taux_aims = k["taux"]
    taux_mro = 100 * k["valides"] / k["total_mro"] if k["total_mro"] else 0.0

    # ── Ligne 1 — Volume ───────────────────────────────────────────────────
    section("Volume des données", "stats")
    _bi = lambda name: f'<i class="bi bi-{name}" style="font-size:21px;line-height:1;"></i>'
    row1 = [
        stat("Vols AIMS", fnum(k["total_aims"]), "système opérations", BLUE, _bi("airplane")),
        stat("Vols MRO", fnum(k["total_mro"]), "système maintenance", TEAL, _bi("wrench")),
        stat_clickable("Non-matchs AIMS", fnum(k["nm_aims"]),
                       "sans contrepartie MRO", AMBER, _bi("search"),
                       "?nav=tables&drill=nm_aims"),
        stat_clickable("Non-matchs MRO", fnum(k["nm_mro"]),
                       "sans contrepartie AIMS", VIOLET, _bi("search"),
                       "?nav=tables&drill=nm_mro"),
    ]
    st.markdown(grid(row1, 4), unsafe_allow_html=True)

    # ── Ligne 2 — Qualité ──────────────────────────────────────────────────
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    section("Qualité des données", "monitor")
    row2 = [
        stat("Réconciliation", f"{taux_aims:.1f}%",
             f"AIMS {taux_aims:.1f}% · MRO {taux_mro:.1f}%", OLIVE, _bi("check-circle-fill")),
        stat_clickable("Anomalies non traitées", fnum(len(non_traitees)),
                       "en attente de traitement", ROSE, _bi("exclamation-triangle-fill"),
                       "?nav=dashboard&drill=nt"),
        stat_clickable("Anomalies ce mois", fnum(len(anom_ce_mois)),
                       lbl_mois, AMBER, _bi("calendar3"),
                       f"?nav=dashboard&drill=mois&mval={lbl_mois}"),
        stat_clickable("Critiques", fnum(len(critiques)),
                       "Très Suspects non traités", CRIMSON, _bi("bell-fill"),
                       "?nav=dashboard&drill=crit"),
    ]
    st.markdown(grid(row2, 4), unsafe_allow_html=True)

    # ── Graphiques d'analyse : répartition AIMS / MRO ───────────────────────
    section("Graphiques d'analyse", "analytics")
    leg = ["Vols matchés", "Non-matchs normaux", "Anomalies détectées"]
    cols = [GREEN, AMBER, CRIMSON]
    d1, d2 = st.columns(2)
    with d1:
        with st.container(border=True):
            st.markdown('<div class="ttl">Répartition AIMS</div>', unsafe_allow_html=True)
            tot = max(k["total_aims"], 1)
            st.plotly_chart(
                donut(leg, [k["valides"], k["norm_aims"], k["anom_aims"]], cols, h=300,
                      center=f"{100*k['valides']/tot:.1f}%", sub="matchés"),
                use_container_width=True, config={"displayModeBar": False})
    with d2:
        with st.container(border=True):
            st.markdown('<div class="ttl">Répartition MRO</div>', unsafe_allow_html=True)
            tot = max(k["total_mro"], 1)
            st.plotly_chart(
                donut(leg, [k["valides"], k["norm_mro"], k["anom_mro"]], cols, h=300,
                      center=f"{100*k['valides']/tot:.1f}%", sub="matchés"),
                use_container_width=True, config={"displayModeBar": False})

    # ── Anomalies par mois + Top 3 routes ─────────────────────────────────
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    ev1, ev2 = st.columns([1, 1])
    with ev1:
        with st.container(border=True):
            st.markdown('<div class="ttl">Anomalies détectées par mois (AIMS vs MRO)</div>',
                        unsafe_allow_html=True)
            if not non_traitees.empty and "Date" in non_traitees.columns and non_traitees["Date"].notna().any():
                g = (non_traitees.dropna(subset=["Date"])
                         .assign(mois=lambda d: d["Date"].dt.to_period("M").astype(str))
                         .groupby(["mois", "Source"]).size().unstack(fill_value=0).sort_index())
                st.plotly_chart(
                    split_trend(g.index.tolist(),
                                g.get("AIMS", pd.Series(0, index=g.index)).tolist(),
                                g.get("MRO", pd.Series(0, index=g.index)).tolist(), h=320),
                    use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Données de dates insuffisantes")
    with ev2:
        with st.container(border=True):
            st.markdown('<div class="ttl">Top 3 routes les plus anomaliques</div>',
                        unsafe_allow_html=True)
            if not non_traitees.empty and "AeroDepart" in non_traitees.columns:
                routes = (non_traitees["AeroDepart"].astype(str) + " → "
                          + non_traitees["AeroArriv"].astype(str))
                top3 = routes.value_counts().head(3)
                if not top3.empty:
                    colors_t3 = [CRIMSON, AMBER, "#e8920c"][:len(top3)]
                    fig_r = go.Figure(go.Bar(
                        x=top3.values.tolist(), y=top3.index.tolist(), orientation="h",
                        marker=dict(color=colors_t3),
                        text=top3.values.tolist(), textposition="outside",
                        textfont=dict(color=INK, size=12),
                        hovertemplate="%{y} : %{x} anomalies<extra></extra>"))
                    fig_r.update_layout(yaxis=dict(autorange="reversed"))
                    _light(fig_r, h=320)
                    st.plotly_chart(fig_r, use_container_width=True,
                                    config={"displayModeBar": False})
                    st.markdown("<div class='sub'>Ces liaisons concentrent le plus "
                                "d'erreurs de saisie AIMS ↔ MRO</div>",
                                unsafe_allow_html=True)
                else:
                    st.info("Aucune route anomalique")
            else:
                st.info("Données insuffisantes")



# ============================================================================
#  PAGE — DASHBOARD  (graphique principal interactif "Analyser par")
# ============================================================================
DIMS = ["Mois", "Matricule", "Numéro de vol", "Route",
        "Aéroport départ", "Aéroport arrivée", "Type d'anomalie"]


def breakdown(df: pd.DataFrame, dim: str, topn: int = 12):
    """Renvoie (labels, values, kind) selon la dimension choisie."""
    is_time = dim in ("Mois", "Jour", "Année")
    if is_time:
        d = df.dropna(subset=["Date"])
        if d.empty:
            return [], [], "time"
        if dim == "Mois":
            s = d["Date"].dt.to_period("M").astype(str)
        elif dim == "Jour":
            s = d["Date"].dt.date.astype(str)
        else:
            s = d["Date"].dt.year.astype(str)
        g = s.value_counts().sort_index()
        return g.index.tolist(), g.values.tolist(), "time"

    colmap = {"Matricule": "Matricule", "Numéro de vol": "NumVol",
              "Aéroport départ": "AeroDepart", "Aéroport arrivée": "AeroArriv",
              "Type d'anomalie": "TypeAnomalie", "Statut": "Statut"}
    if dim == "Route":
        s = df["AeroDepart"].astype(str) + " → " + df["AeroArriv"].astype(str)
    else:
        s = df[colmap[dim]].astype(str)
    g = s.value_counts().head(topn)
    return g.index.tolist(), g.values.tolist(), "cat"


def page_dashboard():
    anom = load_all_anomalies()

    _drill = st.session_state.pop("_drill", None)
    _drill_mval = st.session_state.pop("_drill_mval", None)

    stat_opts  = ["Tous", "Suspect", "Très Suspect", "Écart détecté"]
    trait_opts = ["Toutes", "Non traitées", "Traitées"]
    def_stat  = 0
    def_trait = 0
    if _drill in ("nt", "crit", "mois"):
        def_trait = 1
    if _drill == "crit":
        def_stat = 2

    with st.container(border=True):
        f1, f2, f3, f4, f5 = st.columns([1, 1.2, 1, 1, 1.3])
        with f1:
            source = st.selectbox("Source", ["AIMS + MRO", "AIMS seule", "MRO seule"])
        with f2:
            dim = st.selectbox("Analyser par", DIMS,
                               help="Change la dimension du graphique principal")
        with f3:
            stat_f = st.selectbox("Statut", stat_opts, index=def_stat)
        with f4:
            trait_f = st.selectbox("Traitement", trait_opts, index=def_trait)
        with f5:
            if not anom.empty and anom["Date"].notna().any():
                dmin, dmax = anom["Date"].min().date(), anom["Date"].max().date()
                rng = st.date_input("Période", (dmin, dmax), min_value=dmin, max_value=dmax)
            else:
                rng = None

    df = anom.copy()
    if source == "AIMS seule":
        df = df[df["Source"] == "AIMS"]
    elif source == "MRO seule":
        df = df[df["Source"] == "MRO"]
    if stat_f != "Tous":
        df = df[df["Statut"] == stat_f]
    if "Traitee" in df.columns:
        if trait_f == "Non traitées":
            df = df[df["Traitee"] == 0]
        elif trait_f == "Traitées":
            df = df[df["Traitee"] == 1]
    if rng and isinstance(rng, (list, tuple)) and len(rng) == 2:
        df = df[(df["Date"].dt.date >= rng[0]) & (df["Date"].dt.date <= rng[1])]
    if _drill_mval and "Date" in df.columns:
        df = df[df["Date"].dt.to_period("M").astype(str) == _drill_mval]

    k = global_kpis()
    tres = int((df["Statut"] == "Très Suspect").sum())
    susp = int((df["Statut"] == "Suspect").sum())
    fn_c = int((df["Statut"] == "Écart détecté").sum())
    cards = [
        stat("Vols validés", fnum(k["valides"]), "rapprochés", GREEN,
             '<i class="bi bi-check-circle-fill" style="font-size:21px;line-height:1;"></i>'),
        stat("Anomalies", fnum(len(df)), "selon filtres", ROSE,
             '<i class="bi bi-exclamation-triangle-fill" style="font-size:21px;line-height:1;"></i>'),
        stat("Très suspects", fnum(tres), "score IF très bas", CRIMSON,
             '<i class="bi bi-circle-fill" style="font-size:21px;line-height:1;"></i>'),
        stat("Suspects", fnum(susp), "à vérifier", AMBER,
             '<i class="bi bi-exclamation-circle" style="font-size:21px;line-height:1;"></i>'),
        stat("Écarts détectés", fnum(fn_c), "écarts détectés", VIOLET,
             '<i class="bi bi-arrow-repeat" style="font-size:21px;line-height:1;"></i>'),
    ]
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    st.markdown(grid(cards, 5), unsafe_allow_html=True)
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.7, 1])
    with c1:
        with st.container(border=True):
            if dim == "Numéro de vol":
                st.markdown('<div class="ttl">Top 10 — Numéros de vol les plus anomaliques</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ttl">Anomalies par {dim.lower()}</div>',
                            unsafe_allow_html=True)
            labels, values, kind = breakdown(df, dim, topn=10 if dim == "Numéro de vol" else 12)
            if labels:
                if kind == "time":
                    st.plotly_chart(area_trend(labels, values, BLUE, h=320),
                                    use_container_width=True, config={"displayModeBar": False})
                else:
                    st.plotly_chart(cbar(labels, values, h=320),
                                    use_container_width=True, config={"displayModeBar": False})
                if dim == "Numéro de vol":
                    st.caption("Les 10 numéros de vol concentrant le plus d'anomalies "
                               "sur la période sélectionnée")
            else:
                st.info("Aucune donnée pour ce filtre")
    with c2:
        with st.container(border=True):
            st.markdown('<div class="ttl">Répartition par statut</div>', unsafe_allow_html=True)
            sc = df["Statut"].value_counts()
            if not sc.empty:
                cmap = {"Suspect": AMBER, "Très Suspect": CRIMSON,
                        "Écart détecté": VIOLET, "Normal": GREEN}
                colors = [cmap.get(s, BLUE) for s in sc.index]
                st.plotly_chart(donut(sc.index.tolist(), sc.values.tolist(), colors, h=300),
                                use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("—")

    # ── Heatmap + Anomalies par source ────────────────────────────────────
    h1, h2 = st.columns([1, 1])
    with h1:
        with st.container(border=True):
            st.markdown('<div class="ttl">Quand planifier les contrôles qualité de saisie ?</div>',
                        unsafe_allow_html=True)
            hm_result = heatmap_chart(df, h=380)
            hm_fig = hm_result[0] if hm_result else None
            if hm_fig:
                st.plotly_chart(hm_fig, use_container_width=True,
                                config={"displayModeBar": False, "scrollZoom": False})
                st.caption("Plus la cellule est rouge, plus ce jour/mois concentre "
                           "d'erreurs de saisie")
                fc1, fc2 = st.columns(2)
                with fc1:
                    sel_j = st.selectbox("Jour", ["—"] + _JOURS_HM, key="hm_jour")
                with fc2:
                    sel_m = st.selectbox("Mois", ["—"] + _MOIS_HM, key="hm_mois")
                if sel_j != "—" and "jour_semaine" in df.columns:
                    df = df[df["jour_semaine"] == _JOURS_HM.index(sel_j)]
                if sel_m != "—" and "mois_num" in df.columns:
                    df = df[df["mois_num"] == _MOIS_HM.index(sel_m) + 1]
            else:
                st.info("Données de dates insuffisantes")
    with h2:
        with st.container(border=True):
            st.markdown('<div class="ttl">Anomalies par système source</div>'
                        '<div class="sub">Quel système génère le plus d\'écarts ?</div>',
                        unsafe_allow_html=True)
            if "Source" in df.columns and not df.empty:
                sc_src = df["Source"].value_counts()
                src_total = max(sc_src.sum(), 1)
                src_labs = sc_src.index.tolist()
                src_vals = sc_src.values.tolist()
                src_colors = [BLUE if s == "AIMS" else TEAL for s in src_labs]
                src_text = [f"{v} ({100*v/src_total:.0f}%)" for v in src_vals]
                fig_s = go.Figure(go.Bar(
                    x=src_labs, y=src_vals, marker=dict(color=src_colors),
                    text=src_text, textposition="outside",
                    textfont=dict(color=INK, size=12),
                    hovertemplate="%{x} : %{y} anomalies<extra></extra>"))
                _light(fig_s, h=380)
                st.plotly_chart(fig_s, use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.info("Données insuffisantes")

    if _drill:
        st.markdown('<script>window.addEventListener("load",()=>'
                    'document.getElementById("tbl-anomalies")?.scrollIntoView({behavior:"smooth"}))'
                    '</script>', unsafe_allow_html=True)
    st.markdown('<div id="tbl-anomalies"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="ttl">Détail des anomalies filtrées</div>', unsafe_allow_html=True)
        show_cols = ["ID", "Source", "Date", "NumVol", "Matricule", "AeroDepart",
                     "AeroArriv", "Statut", "TypeAnomalie", "ScoreAnomalie",
                     "RaisonAnomalie", "Traitee"]
        avail = [c for c in show_cols if c in df.columns]
        show = df[avail].copy().sort_values("ScoreAnomalie").reset_index(drop=True)
        if "Traitee" in show.columns:
            show["Traitee"] = show["Traitee"].astype(bool)

        def _on_editor_change():
            st.session_state["_anom_pending_save"] = True

        edited = st.data_editor(
            show,
            column_config={
                "Traitee": st.column_config.CheckboxColumn("Traitée"),
                "ID": st.column_config.NumberColumn("ID", width="small"),
            },
            disabled=[c for c in avail if c != "Traitee"],
            use_container_width=True, height=400, hide_index=True,
            key="anomaly_editor", on_change=_on_editor_change)

        has_changes = st.session_state.get("_anom_pending_save", False)
        bc1, bc2 = st.columns([1, 1])
        with bc1:
            if st.button("💾 Sauvegarder les modifications", type="primary",
                         use_container_width=True, disabled=not has_changes):
                if "Traitee" in show.columns and "ID" in show.columns:
                    n = 0
                    for i in range(len(show)):
                        orig = show.loc[i, "Traitee"]
                        curr = edited.loc[i, "Traitee"]
                        if orig != curr:
                            id_v = int(edited.loc[i, "ID"])
                            src = str(edited.loc[i, "Source"])
                            if curr:
                                _mark_traitee(id_v, src)
                            else:
                                _unmark_traitee(id_v, src)
                            n += 1
                    st.session_state["_anom_pending_save"] = False
                    if n:
                        st.cache_data.clear()
                        st.toast(f"✅ {n} anomalie(s) mise(s) à jour.")
                        st.rerun()
        with bc2:
            st.download_button("⬇ Exporter (CSV)",
                               show.to_csv(index=False).encode("utf-8"),
                               "anomalies_filtrees.csv", "text/csv",
                               use_container_width=True)


# ============================================================================
#  PAGE — TABLES
# ============================================================================
_VIRTUAL_TABLES = {
    "Anomalies (toutes)": "Toutes les anomalies IF + RF unifiées",
    "Vols normaux (non-matchés)": "Vols non rapprochés sans anomalie détectée",
    "Non-matchés AIMS": "Vols AIMS sans contrepartie MRO",
    "Non-matchés MRO": "Vols MRO sans contrepartie AIMS",
}


@st.cache_data(ttl=300, show_spinner=False)
def _load_non_matches(source):
    if source == "AIMS":
        raw = load("AIMS")
        vv = load("VolsValidesEtape1")
        if "IDAIMS" in raw.columns and "IDAIMS" in vv.columns:
            return raw[~raw["IDAIMS"].isin(set(vv["IDAIMS"].dropna()))]
    else:
        raw = load("MRO")
        vv = load("VolsValidesEtape1")
        if "IDMRO" in raw.columns and "IDMRO" in vv.columns:
            return raw[~raw["IDMRO"].isin(set(vv["IDMRO"].dropna()))]
    return pd.DataFrame()


def page_tables():
    tables = ["AIMS", "MRO", "VolsValidesEtape1",
              "Anomalies (toutes)", "Vols normaux (non-matchés)",
              "Non-matchés AIMS", "Non-matchés MRO"]
    _drill = st.session_state.pop("_drill", None)
    def_table = 0
    if _drill == "nm_aims":
        def_table = tables.index("Non-matchés AIMS")
    elif _drill == "nm_mro":
        def_table = tables.index("Non-matchés MRO")
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.4, 1, 1.4])
        with c1:
            tname = st.selectbox("Table", tables, index=def_table)
        with c2:
            if tname not in _VIRTUAL_TABLES:
                try:
                    _cnt = read_query(f"SELECT COUNT(*) AS n FROM {tname}")
                    _total = int(_cnt.iloc[0, 0])
                except Exception:
                    _total = 1000
            else:
                _total = 100000
            limit = st.number_input("Lignes max", 1, max(_total, 1), _total, step=100)
        with c3:
            search = st.text_input("Rechercher", placeholder="NumVol, matricule, aéroport…")

    if tname == "Anomalies (toutes)":
        df = load_all_anomalies()
    elif tname == "Vols normaux (non-matchés)":
        df = load_vrais_normaux()
    elif tname == "Non-matchés AIMS":
        df = _load_non_matches("AIMS")
    elif tname == "Non-matchés MRO":
        df = _load_non_matches("MRO")
    else:
        df = load(tname, top=int(limit))
    if df.empty:
        st.warning("Table vide ou indisponible.")
        return

    view = df.copy()
    if search:
        s = search.strip().lower()
        mask = view.apply(lambda r: r.astype(str).str.lower().str.contains(s, na=False).any(), axis=1)
        view = view[mask]

    desc = _VIRTUAL_TABLES.get(tname, f"table {tname}")
    cards = [
        stat("Lignes", fnum(len(df)), desc, BLUE,
             '<i class="bi bi-database" style="font-size:21px;line-height:1;"></i>'),
        stat("Colonnes", str(df.shape[1]), "champs", TEAL,
             '<i class="bi bi-layout-three-columns" style="font-size:21px;line-height:1;"></i>'),
        stat("Résultats", fnum(len(view)), "après recherche" if search else "affichés", GREEN,
             '<i class="bi bi-search" style="font-size:21px;line-height:1;"></i>'),
    ]
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    st.markdown(grid(cards, 3), unsafe_allow_html=True)
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    with st.expander("Choisir les colonnes à afficher", expanded=False):
        toutes_colonnes = view.columns.tolist()
        colonnes_non_vides = [c for c in toutes_colonnes
                              if view[c].notna().any()
                              and not (view[c].astype(str) == "None").all()]
        colonnes_choisies = st.multiselect(
            "Colonnes", options=toutes_colonnes,
            default=colonnes_non_vides,
            help="Sélectionnez les colonnes à afficher dans la table")
    if colonnes_choisies:
        view = view[colonnes_choisies]

    with st.container(border=True):
        st.markdown(f'<div class="ttl">{tname}</div>', unsafe_allow_html=True)
        st.dataframe(view, use_container_width=True, height=460, hide_index=True)
        st.download_button("⬇ Exporter (CSV)", view.to_csv(index=False).encode("utf-8"),
                           f"{tname.replace(' ', '_')}.csv", "text/csv")


# ============================================================================
#  PAGE — DÉTECTION
# ============================================================================
def _historique_ref():
    if db_alive():
        try:
            from predict import charger_ref_historique
            return charger_ref_historique()
        except Exception:
            pass
    vv = load("VolsValidesEtape1")
    return pd.DataFrame({
        "NumVol": vv["NumVolAIMS"].astype(str), "Matricule": vv["MatriculeAIMS"].astype(str),
        "AeroDepart": vv["AeroDepartAIMS"].astype(str), "AeroArriv": vv["AeroArrivAIMS"].astype(str),
        "Date": pd.to_datetime(vv["DateAIMS"], errors="coerce")}).dropna(subset=["Date"])


def score_flight(date, mat, numvol, dep, arr):
    import pickle
    from config import MODEL_PATH, RF_MODEL_PATH
    from features import FlightFeatureEncoder
    from predict import determiner_type_anomalie, analyser_raison_anomalie

    enc = FlightFeatureEncoder.load()
    with open(MODEL_PATH, "rb") as f:
        iso = pickle.load(f)
    rf = None
    try:
        with open(RF_MODEL_PATH, "rb") as f:
            rf = pickle.load(f)
    except Exception:
        pass

    df_ref = _historique_ref()
    row_df = pd.DataFrame([{
        "Matricule": str(mat).strip().upper(), "NumVol": str(numvol).strip().upper(),
        "AeroDepart": str(dep).strip().upper(), "AeroArriv": str(arr).strip().upper(),
        "Date": pd.to_datetime(date)}])
    X = enc.transform(row_df)
    score = float(iso.score_samples(X)[0])
    if_anom = score < iso.offset_
    row = row_df.iloc[0]

    if not if_anom:
        if_statut, type_anom, raison = "Normal", "Aucune", "Aucune"
    else:
        if_statut = "Suspect"
        type_anom = determiner_type_anomalie(df_ref, row)
        raison = analyser_raison_anomalie(df_ref, row, score, None)

    rf_anom = rf_proba = None
    if not if_anom and rf is not None:
        rf_anom  = bool(rf.predict(X[:, :8])[0] == 1)
        rf_proba = float(rf.predict_proba(X[:, :8])[0, 1])
        if rf_anom:
            type_anom = determiner_type_anomalie(df_ref, row)
            raison = analyser_raison_anomalie(df_ref, row, score, None)

    return {"score": score, "if_anom": if_anom, "type": type_anom,
            "raison": raison, "rf_anom": rf_anom, "rf_proba": rf_proba,
            "if_offset": float(iso.offset_)}, df_ref


def page_detection():
    ref = _historique_ref()
    mats = sorted(set(_FLEET) | set(ref["Matricule"].astype(str)))
    apts = sorted(set(_APTS) | set(ref["AeroDepart"].astype(str)) | set(ref["AeroArriv"].astype(str)))

    with st.container(border=True):
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            date = st.date_input("Date du vol", dt.date(2019, 6, 15))
        with r1c2:
            numvol = st.text_input("Numéro de vol", "1010")
        with r1c3:
            mat = st.selectbox("Matricule", mats,
                               index=mats.index("7T-VCA") if "7T-VCA" in mats else 0)
        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            dep = st.selectbox("Aéroport départ", apts,
                               index=apts.index("ALG") if "ALG" in apts else 0)
        with r2c2:
            arr = st.selectbox("Aéroport arrivée", apts,
                               index=apts.index("HME") if "HME" in apts else 0)
        with r2c3:
            st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
            st.markdown("""<style>
                div[data-testid="stButton"] button[kind="primary"]::before {
                    font-family:"bootstrap-icons"; content:"\\F52A";
                    margin-right:8px; font-size:16px; }
            </style>""", unsafe_allow_html=True)
            run = st.button("Analyser le vol", type="primary", use_container_width=True)

    if not run:
        return

    try:
        res, df_ref = score_flight(date, mat, numvol, dep, arr)
    except Exception as e:
        st.error(f"Scoring réel indisponible (base ou modèles .pkl introuvables). Détail : {e}")
        st.info("Vérifiez la connexion SQL Server et la présence de "
                "`models/isolation_forest.pkl`, `random_forest.pkl`, `label_encoders.pkl`.")
        return

    if_bad = bool(res["if_anom"])
    rf_bad = bool(res["rf_anom"])

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    if if_bad:
        st.markdown(
            f'<div style="max-width:640px;margin:0 auto;background:{CARD};border:2px solid {CRIMSON};'
            f'border-radius:18px;padding:28px 32px;text-align:center;'
            f'box-shadow:0 4px 18px rgba(212,12,28,.10);">'
            f'<i class="bi bi-shield-exclamation" style="font-size:48px;color:{CRIMSON};"></i>'
            f'<div style="font-size:24px;font-weight:800;color:{CRIMSON};margin-top:10px;">'
            f'Anomalie détectée</div>'
            f'<hr style="border:none;border-top:1px solid {BORDER};margin:14px 0;">'
            f'<span style="background:{_tint(CRIMSON,.10)};color:{CRIMSON};padding:4px 14px;'
            f'border-radius:10px;font-size:13px;font-weight:600;">Score IF : {res["score"]:.4f}</span>'
            f'<hr style="border:none;border-top:1px solid {BORDER};margin:14px 0;">'
            f'<div style="color:{INK};font-size:14px;line-height:1.6;">{res["raison"]}</div>'
            f'</div>', unsafe_allow_html=True)
    elif rf_bad:
        st.markdown(
            f'<div style="max-width:640px;margin:0 auto;background:{CARD};border:2px solid {CRIMSON};'
            f'border-radius:18px;padding:28px 32px;text-align:center;'
            f'box-shadow:0 4px 18px rgba(212,12,28,.10);">'
            f'<i class="bi bi-shield-exclamation" style="font-size:48px;color:{CRIMSON};"></i>'
            f'<div style="font-size:24px;font-weight:800;color:{CRIMSON};margin-top:10px;">'
            f'Anomalie détectée</div>'
            f'<div style="color:{MUTED};font-size:13px;margin-top:4px;">'
            f'Détecté par analyse approfondie</div>'
            f'<hr style="border:none;border-top:1px solid {BORDER};margin:14px 0;">'
            f'<span style="background:{_tint(CRIMSON,.10)};color:{CRIMSON};padding:4px 14px;'
            f'border-radius:10px;font-size:13px;font-weight:600;">'
            f'Probabilité : {res["rf_proba"]:.1%}</span>'
            f'<hr style="border:none;border-top:1px solid {BORDER};margin:14px 0;">'
            f'<div style="color:{INK};font-size:14px;line-height:1.6;">{res["raison"]}</div>'
            f'</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div style="max-width:640px;margin:0 auto;background:{CARD};border:2px solid {GREEN};'
            f'border-radius:18px;padding:28px 32px;text-align:center;'
            f'box-shadow:0 4px 18px rgba(90,168,49,.10);">'
            f'<i class="bi bi-shield-check" style="font-size:48px;color:{GREEN};"></i>'
            f'<div style="font-size:24px;font-weight:800;color:{GREEN};margin-top:10px;">'
            f'Vol normal</div>'
            f'<hr style="border:none;border-top:1px solid {BORDER};margin:14px 0;">'
            f'<span style="background:{_tint(GREEN,.10)};color:{OLIVE_D};padding:4px 14px;'
            f'border-radius:10px;font-size:13px;font-weight:600;">Score IF : {res["score"]:.4f}</span>'
            f'</div>', unsafe_allow_html=True)



# ============================================================================
#  MAIN
# ============================================================================
def main():
    st.set_page_config(page_title="Domestic Airlines — Fiabilité",
                       page_icon="✈️", layout="wide", initial_sidebar_state="collapsed")
    _ensure_traitement_table()

    if "page" not in st.session_state:
        st.session_state.page = "home"

    qp = st.query_params
    if "refresh" in qp:
        st.cache_data.clear()
    if "nav" in qp:
        st.session_state.page = qp.get("nav")
    if "drill" in qp:
        d = qp.get("drill")
        st.session_state["_drill"] = d
        if "mval" in qp:
            st.session_state["_drill_mval"] = qp.get("mval")
    for kk in ("nav", "refresh", "drill", "mval"):
        try:
            if kk in qp:
                del st.query_params[kk]
        except Exception:
            pass

    css()
    page = st.session_state.page

    if page == "home":
        page_home()
        return

    appbar(page)
    if page == "overview":
        pagehead("Vue d'ensemble", "Réconciliation AIMS ↔ MRO et indice de fiabilité des données.",
                 '<i class="bi bi-speedometer2"></i>')
        page_overview()
    elif page == "dashboard":
        pagehead("Détection — Statistiques", "Explorez, filtrez et traitez les anomalies de saisie détectées.",
                 '<i class="bi bi-bar-chart-line"></i>')
        page_dashboard()
    elif page == "tables":
        pagehead("Exploration SQL", "Consultation des tables sources et des résultats du pipeline.",
                 '<i class="bi bi-table"></i>')
        page_tables()
    elif page == "detection":
        pagehead("Détection d'anomalie", "Analysez un vol et obtenez un diagnostic immédiat sur ses données.",
                 '<i class="bi bi-cpu"></i>')
        page_detection()


if __name__ == "__main__":
    main()