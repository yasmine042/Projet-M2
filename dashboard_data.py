"""
dashboard_data.py — Fonctions et style partagés entre les pages du dashboard.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from db import read_table, read_query
from config import TABLE_VOLS_VALIDES

# ── Palette ──────────────────────────────────────────────────────────────────
VERT   = "#2ca02c"
ORANGE = "#ff7f0e"
ROUGE  = "#d62728"
BLEU   = "#1f4e79"
ACCENT = "#5b5fc7"
INK    = "#1a1c2e"
MUTED  = "#8b8fa3"


# ══════════════════════════════════════════════════════════════════════════════
# STYLE — cards façon "Finexy"
# ══════════════════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown(f"""
    <style>
    .stApp {{
        background-color: #f4f6fb;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {INK};
    }}
    [data-testid="stSidebar"] * {{
        color: #e6e6f0 !important;
    }}
    [data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.1);
    }}

    /* KPI metric cards */
    div[data-testid="stMetric"] {{
        background: #ffffff;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #eef0f6;
    }}
    div[data-testid="stMetricLabel"] {{
        font-size: 0.85rem;
        color: {MUTED};
    }}
    div[data-testid="stMetricValue"] {{
        font-size: 1.6rem;
        color: {INK};
        font-weight: 700;
    }}

    /* Conteneurs avec bordure -> cartes */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: #ffffff;
        border-radius: 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }}

    /* Titres de page */
    .dash-title {{
        color: {INK};
        font-weight: 800;
        margin-bottom: 0;
    }}
    .dash-subtitle {{
        color: {MUTED};
        margin-top: 4px;
    }}
    </style>
    """, unsafe_allow_html=True)


def page_header(title, subtitle=""):
    st.markdown(
        f"<h1 class='dash-title'>{title}</h1>"
        f"<p class='dash-subtitle'>{subtitle}</p><hr>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT (mis en cache — partagé entre toutes les pages)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    aims_total = len(read_table("AIMS"))
    mro_total  = len(read_table("MRO"))
    vv_total   = len(read_table(TABLE_VOLS_VALIDES))

    nm_aims = int(read_query(
        f"SELECT COUNT(*) FROM AIMS WHERE IDAIMS NOT IN "
        f"(SELECT IDAIMS FROM {TABLE_VOLS_VALIDES} WHERE IDAIMS IS NOT NULL)"
    ).iloc[0, 0])

    nm_mro = int(read_query(
        f"SELECT COUNT(*) FROM MRO WHERE IDMRO NOT IN "
        f"(SELECT IDMRO FROM {TABLE_VOLS_VALIDES} WHERE IDMRO IS NOT NULL)"
    ).iloc[0, 0])

    try:
        df_aims = read_table("AnomaliesAIMS")
    except Exception:
        df_aims = pd.DataFrame()

    try:
        df_mro = read_table("AnomaliesMRO")
    except Exception:
        df_mro = pd.DataFrame()

    return dict(
        aims_total=aims_total, mro_total=mro_total, vv_total=vv_total,
        nm_aims=nm_aims, nm_mro=nm_mro,
        df_aims=df_aims, df_mro=df_mro,
    )


@st.cache_data
def load_normaux():
    try:
        df_na = read_table("VoisNormalesAIMS")
    except Exception:
        df_na = pd.DataFrame()
    try:
        df_nm = read_table("VoisNormalesMRO")
    except Exception:
        df_nm = pd.DataFrame()
    return df_na, df_nm


# ══════════════════════════════════════════════════════════════════════════════
# PRÉPARATION DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════════

def build_anomaly_table(df_aims, df_mro):
    rows = []
    specs = [
        (df_aims, "AIMS", "IDAIMS",  "NumVolAIMS",  "AeroDepartAIMS",
         "AeroArrivAIMS",  "DateAIMS",   "MatriculeAIMS"),
        (df_mro,  "MRO",  "IDMRO",   "NumVolMRO",   "AeroDepartMRO",
         "AeroArrivMRO",   "DateMRO",    "MatriculeMRO"),
    ]
    for df, src, id_c, nv_c, dep_c, arr_c, dat_c, mat_c in specs:
        if df.empty:
            continue
        for _, r in df.iterrows():
            rows.append({
                "Source":    src,
                "ID":        str(r.get(id_c,  "")),
                "Num Vol":   str(r.get(nv_c,  "")),
                "Départ":    str(r.get(dep_c, "")),
                "Arrivée":   str(r.get(arr_c, "")),
                "Date":      str(r.get(dat_c, ""))[:10],
                "Matricule": str(r.get(mat_c, "")),
                "Statut":    str(r.get("Statut",         "")),
                "Score IF":  round(float(r.get("ScoreAnomalie",  0)), 4),
                "Raison":    str(r.get("RaisonAnomalie", "")),
            })
    return pd.DataFrame(rows)


def build_ranking(df_aims, df_mro):
    mats = {}
    for df, src, mat_c in [(df_aims, "AIMS", "MatriculeAIMS"),
                            (df_mro,  "MRO",  "MatriculeMRO")]:
        if df.empty or mat_c not in df.columns:
            continue
        for _, r in df.iterrows():
            m = str(r[mat_c])
            s = float(r.get("ScoreAnomalie", 0))
            if m not in mats:
                mats[m] = {"Matricule": m, "AIMS": 0, "MRO": 0, "scores": []}
            mats[m][src] += 1
            mats[m]["scores"].append(s)

    rows = [{"Matricule":       m,
             "Anomalies AIMS":  v["AIMS"],
             "Anomalies MRO":   v["MRO"],
             "Total":           v["AIMS"] + v["MRO"],
             "Score moyen IF":  round(sum(v["scores"]) / len(v["scores"]), 4)}
            for m, v in mats.items()]
    df = pd.DataFrame(rows).sort_values("Total", ascending=False).reset_index(drop=True)
    df.index += 1
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES PLOTLY
# ══════════════════════════════════════════════════════════════════════════════

def make_gauge(score, title):
    color = VERT if score >= 95 else (ORANGE if score >= 85 else ROUGE)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%", "font": {"size": 32, "color": color}},
        title={"text": title, "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": color, "thickness": 0.28},
            "steps": [
                {"range": [0,  80],  "color": "#fde8e8"},
                {"range": [80, 95],  "color": "#fef3e2"},
                {"range": [95, 100], "color": "#e8f5e9"},
            ],
            "threshold": {"line": {"color": "red", "width": 3},
                          "thickness": 0.75, "value": 95},
        },
    ))
    fig.update_layout(height=210, margin=dict(t=55, b=5, l=25, r=25))
    return fig


def make_donut(label, total, nm, n_anom):
    fig = go.Figure(go.Pie(
        labels=["Vols matchés", "Non-matchs normaux", "Anomalies IF"],
        values=[total - nm, max(nm - n_anom, 0), n_anom],
        hole=0.55,
        marker_colors=[VERT, ORANGE, ROUGE],
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:,}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Répartition {label}", font_size=13),
        height=300, margin=dict(t=45, b=5, l=5, r=5),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def make_bar_statuts(df_aims, df_mro):
    def vc(df):
        if df.empty or "Statut" not in df.columns:
            return 0, 0
        v = df["Statut"].value_counts()
        return v.get("Suspect", 0), v.get("Très Suspect", 0)
    s_a, ts_a = vc(df_aims)
    s_m, ts_m = vc(df_mro)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="AIMS", x=["Suspect","Très Suspect"],
                         y=[s_a, ts_a], marker_color=BLEU))
    fig.add_trace(go.Bar(name="MRO",  x=["Suspect","Très Suspect"],
                         y=[s_m, ts_m], marker_color=ORANGE))
    fig.update_layout(title="Anomalies par statut", barmode="group",
                      height=300, margin=dict(t=45, b=5, l=5, r=5))
    return fig


def _cat(r):
    if not r or r == "Aucune":
        return None
    r = r.split(" | ")[0]
    if "jamais vu"    in r: return "NumVol inconnu"
    if "AeroDepart"   in r: return "Départ inhabituel"
    if "AeroArriv"    in r: return "Arrivée inhabituelle"
    if "Matricule"    in r: return "Matricule inhabituel"
    if "jamais opéré" in r: return "Jour inhabituel"
    if "faux positif" in r: return "Possible faux positif"
    if "absente"      in r: return "Combinaison inconnue"
    return "Autre"


def make_bar_raisons(df_aims, df_mro):
    cats = [_cat(r) for df in [df_aims, df_mro]
            if not df.empty and "RaisonAnomalie" in df.columns
            for r in df["RaisonAnomalie"]]
    cats = [c for c in cats if c]
    if not cats:
        return None
    vc = pd.Series(cats).value_counts()
    fig = go.Figure(go.Bar(
        x=vc.values, y=vc.index, orientation="h",
        marker_color=ROUGE,
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    fig.update_layout(title="Raisons d'anomalie", height=300,
                      margin=dict(t=45, b=5, l=150, r=5),
                      yaxis=dict(autorange="reversed"))
    return fig


def make_hist_scores(df_aims, df_mro):
    fig = go.Figure()
    for df, nm, col in [(df_aims, "AIMS", BLEU), (df_mro, "MRO", ORANGE)]:
        if df.empty or "ScoreAnomalie" not in df.columns:
            continue
        fig.add_trace(go.Histogram(x=df["ScoreAnomalie"], name=nm,
                                   marker_color=col, opacity=0.7, nbinsx=20))
    fig.update_layout(title="Distribution des scores IF", barmode="overlay",
                      height=300, margin=dict(t=45, b=5, l=5, r=5),
                      xaxis_title="Score (plus bas = plus suspect)",
                      yaxis_title="Nombre de vols")
    return fig
