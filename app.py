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

PAGES = [("home", "Accueil"), ("overview", "Overview"),
         ("dashboard", "Dashboard"), ("tables", "Tables"),
         ("detection", "Détection")]


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
        statut = (["Faux Négatif IF"]*k if is_fn
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
def global_kpis() -> dict:
    aims, mro, vv = load("AIMS"), load("MRO"), load("VolsValidesEtape1")
    anom, fn = load_anomalies(), load_faux_negatifs()
    total_aims, total_mro, valides = len(aims), len(mro), len(vv)
    etapes = {}
    if "EtapeValidation" in vv.columns and not vv.empty:
        etapes = {int(k): int(v) for k, v in vv["EtapeValidation"].value_counts().items()
                  if str(k).strip().isdigit()}
    a_aims = int((anom["Source"] == "AIMS").sum())
    a_mro  = int((anom["Source"] == "MRO").sum())
    taux   = 100 * valides / total_aims if total_aims else 0.0
    nm_aims = max(0, total_aims - valides)          # AIMS non rapprochés
    nm_mro  = max(0, total_mro  - valides)          # MRO non rapprochés
    norm_aims = max(0, nm_aims - a_aims)            # non-matchs classés normaux (IF)
    norm_mro  = max(0, nm_mro  - a_mro)
    return {"total_aims": total_aims, "total_mro": total_mro, "valides": valides,
            "etapes": etapes, "anom_aims": a_aims, "anom_mro": a_mro,
            "anom_total": a_aims + a_mro, "faux_neg": len(fn), "taux": taux,
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
    fig.update_layout(yaxis=dict(autorange="reversed"))
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


# ============================================================================
#  STYLES GLOBAUX  (clair)
# ============================================================================
def css():
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
    ("overview",  "Overview",
     "Vue d'ensemble des sources AIMS et MRO, taux de réconciliation et indice de fiabilité."),
    ("dashboard", "Dashboard",
     "Statistiques et graphiques de détection : validés vs anomalies, filtres par source et période."),
    ("tables",    "Tables",
     "Exploration des tables SQL : AIMS, MRO, anomalies, faux négatifs, vols validés."),
    ("detection", "Détection",
     "Scoring d'un vol saisi manuellement par Isolation Forest et Random Forest."),
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
            <div class="lead">Plateforme d'analyse de la fiabilité des données aériennes</div>
            <div class="desc">Réconciliation des systèmes <b>AIMS</b> (opérations) et
              <b>MRO</b> (maintenance), et détection d'anomalies par intelligence
              artificielle — Isolation Forest &amp; Random Forest — intégrée à SQL Server.</div>
            <div class="illu">{_HERO_SVG}</div>
            <div class="foot">PFE · Master 2 — Base TALEXPDWH</div>
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
    anom = load_anomalies()

    # ── Indicateurs globaux ─────────────────────────────────────────────────
    section("Indicateurs globaux", "stats")
    cards = [
        stat("Vols AIMS", fnum(k["total_aims"]), "système opérations", BLUE, icon_img("plane")),
        stat("Vols MRO", fnum(k["total_mro"]), "système maintenance", TEAL, icon_img("database")),
        stat("Vols matchés", fnum(k["valides"]), f"{k['taux']:.1f}% des AIMS", GREEN, "✅"),
        stat("Non-matchs AIMS", fnum(k["nm_aims"]), "non rapprochés", AMBER, "🔍"),
        stat("Non-matchs MRO", fnum(k["nm_mro"]), "non rapprochés", VIOLET, "🔍"),
        stat("Anomalies détectées", fnum(k["anom_total"]),
             f"AIMS {k['anom_aims']} · MRO {k['anom_mro']}", ROSE, icon_img("anomaly")),
    ]
    st.markdown(grid(cards, 6), unsafe_allow_html=True)

    # ── Graphiques d'analyse : répartition AIMS / MRO ───────────────────────
    section("Graphiques d'analyse", "analytics")
    leg = ["Vols matchés", "Non-matchs normaux", "Anomalies IF"]
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

    # ── Réconciliation · étapes · anomalies par source ──────────────────────
    section("Réconciliation & validation", "monitor")
    c1, c2, c3 = st.columns([1, 1.2, 1.2])
    with c1:
        with st.container(border=True):
            st.markdown('<div class="ttl">Taux de réconciliation</div>', unsafe_allow_html=True)
            st.plotly_chart(ring(k["taux"], "AIMS rapprochés", OLIVE),
                            use_container_width=True, config={"displayModeBar": False})
            st.markdown(f"<div class='sub' style='text-align:center;'>"
                        f"{fnum(k['valides'])} / {fnum(k['total_aims'])} vols AIMS</div>",
                        unsafe_allow_html=True)
    with c2:
        with st.container(border=True):
            st.markdown('<div class="ttl">Vols validés par étape</div>', unsafe_allow_html=True)
            et = k["etapes"]
            if et:
                xs = [f"Étape {i}" for i in sorted(et)]
                st.plotly_chart(vbar(xs, [et[i] for i in sorted(et)], BLUE),
                                use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Étapes indisponibles")
    with c3:
        with st.container(border=True):
            st.markdown('<div class="ttl">Anomalies par source</div>', unsafe_allow_html=True)
            st.plotly_chart(cbar(["AIMS", "MRO"], [k["anom_aims"], k["anom_mro"]],
                                 h=230, palette=[BLUE, TEAL]),
                            use_container_width=True, config={"displayModeBar": False})

    # ── Tendance mensuelle + synthèse ───────────────────────────────────────
    c4, c5 = st.columns([1.4, 1])
    with c4:
        with st.container(border=True):
            st.markdown('<div class="ttl">Anomalies détectées par mois (AIMS vs MRO)</div>',
                        unsafe_allow_html=True)
            if not anom.empty and anom["Date"].notna().any():
                g = (anom.dropna(subset=["Date"])
                         .assign(mois=lambda d: d["Date"].dt.to_period("M").astype(str))
                         .groupby(["mois", "Source"]).size().unstack(fill_value=0).sort_index())
                st.plotly_chart(
                    split_trend(g.index.tolist(),
                                g.get("AIMS", pd.Series(0, index=g.index)).tolist(),
                                g.get("MRO", pd.Series(0, index=g.index)).tolist()),
                    use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Aucune date d'anomalie disponible")
    with c5:
        with st.container(border=True):
            st.markdown('<div class="ttl">Synthèse</div>', unsafe_allow_html=True)
            mini = [
                stat("Faux négatifs (RF)", fnum(k["faux_neg"]),
                     "rattrapés par Random Forest", AMBER, "🟠"),
                stat("Non-matchs normaux", fnum(k["norm_aims"] + k["norm_mro"]),
                     "classés normaux par l'IF", TEAL, "🟢"),
            ]
            st.markdown(grid(mini, 1), unsafe_allow_html=True)


# ============================================================================
#  PAGE — DASHBOARD  (graphique principal interactif "Analyser par")
# ============================================================================
DIMS = ["Mois", "Jour", "Année", "Matricule", "Numéro de vol", "Route",
        "Aéroport départ", "Aéroport arrivée", "Type d'anomalie", "Statut"]


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
    anom = load_anomalies()

    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([1, 1.2, 1, 1.3])
        with f1:
            source = st.selectbox("Source", ["AIMS + MRO", "AIMS seule", "MRO seule"])
        with f2:
            dim = st.selectbox("Analyser par", DIMS,
                               help="Change la dimension du graphique principal")
        with f3:
            stat_f = st.selectbox("Statut", ["Tous", "Suspect", "Très Suspect"])
        with f4:
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
    if rng and isinstance(rng, (list, tuple)) and len(rng) == 2:
        df = df[(df["Date"].dt.date >= rng[0]) & (df["Date"].dt.date <= rng[1])]

    k = global_kpis()
    tres = int((df["Statut"] == "Très Suspect").sum())
    susp = int((df["Statut"] == "Suspect").sum())
    cards = [
        stat("Vols validés", fnum(k["valides"]), "rapprochés", GREEN, "✅"),
        stat("Anomalies", fnum(len(df)), "selon filtres", ROSE, "⚠️"),
        stat("Très suspects", fnum(tres), "score IF très bas", CRIMSON, "🔴"),
        stat("Suspects", fnum(susp), "à vérifier", AMBER, "🟠"),
    ]
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    st.markdown(grid(cards, 4), unsafe_allow_html=True)
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.7, 1])
    with c1:
        with st.container(border=True):
            st.markdown(f'<div class="ttl">Anomalies par {dim.lower()}</div>',
                        unsafe_allow_html=True)
            labels, values, kind = breakdown(df, dim)
            if labels:
                if kind == "time":
                    st.plotly_chart(area_trend(labels, values, BLUE, h=320),
                                    use_container_width=True, config={"displayModeBar": False})
                else:
                    st.plotly_chart(cbar(labels, values, h=320),
                                    use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Aucune donnée pour ce filtre")
    with c2:
        with st.container(border=True):
            st.markdown('<div class="ttl">Répartition par statut</div>', unsafe_allow_html=True)
            sc = df["Statut"].value_counts()
            if not sc.empty:
                cmap = {"Suspect": AMBER, "Très Suspect": CRIMSON,
                        "Faux Négatif IF": VIOLET, "Normal": GREEN}
                colors = [cmap.get(s, BLUE) for s in sc.index]
                st.plotly_chart(donut(sc.index.tolist(), sc.values.tolist(), colors, h=300),
                                use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("—")

    with st.container(border=True):
        st.markdown('<div class="ttl">Détail des anomalies filtrées</div>', unsafe_allow_html=True)
        show = df[["Source", "Date", "NumVol", "Matricule", "AeroDepart", "AeroArriv",
                   "Statut", "TypeAnomalie", "ScoreAnomalie", "RaisonAnomalie"]]\
            .sort_values("ScoreAnomalie")
        st.dataframe(show, use_container_width=True, height=320, hide_index=True)
        st.download_button("⬇ Exporter (CSV)", show.to_csv(index=False).encode("utf-8"),
                           "anomalies_filtrees.csv", "text/csv")


# ============================================================================
#  PAGE — TABLES
# ============================================================================
def page_tables():
    tables = ["AIMS", "MRO", "VolsValidesEtape1", "AnomaliesAIMS", "AnomaliesMRO",
              "FauxNegatifsAIMS", "FauxNegatifsMRO"]
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.4, 1, 1.4])
        with c1:
            tname = st.selectbox("Table SQL", tables)
        with c2:
            limit = st.number_input("Lignes max", 100, 100000, 1000, step=100)
        with c3:
            search = st.text_input("Rechercher", placeholder="NumVol, matricule, aéroport…")

    df = load(tname, top=int(limit))
    if df.empty:
        st.warning("Table vide ou indisponible.")
        return

    view = df.copy()
    if search:
        s = search.strip().lower()
        mask = view.apply(lambda r: r.astype(str).str.lower().str.contains(s, na=False).any(), axis=1)
        view = view[mask]

    cards = [
        stat("Lignes", fnum(len(df)), f"table {tname}", BLUE, "🗃️"),
        stat("Colonnes", str(df.shape[1]), "champs", TEAL, "📋"),
        stat("Résultats", fnum(len(view)), "après recherche" if search else "affichés", GREEN, "🔎"),
    ]
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    st.markdown(grid(cards, 3), unsafe_allow_html=True)
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(f'<div class="ttl">{tname}</div>', unsafe_allow_html=True)
        st.dataframe(view, use_container_width=True, height=460, hide_index=True)
        st.download_button("⬇ Exporter (CSV)", view.to_csv(index=False).encode("utf-8"),
                           f"{tname}.csv", "text/csv")


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
    if rf is not None:
        rf_anom  = bool(rf.predict(X[:, :8])[0] == 1)
        rf_proba = float(rf.predict_proba(X[:, :8])[0, 1])

    return {"score": score, "if_statut": if_statut, "type": type_anom,
            "raison": raison, "rf_anom": rf_anom, "rf_proba": rf_proba}, df_ref


def page_detection():
    ref = _historique_ref()
    mats = sorted(set(_FLEET) | set(ref["Matricule"].astype(str)))
    apts = sorted(set(_APTS) | set(ref["AeroDepart"].astype(str)) | set(ref["AeroArriv"].astype(str)))

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            date = st.date_input("Date du vol", dt.date(2019, 6, 15))
            numvol = st.text_input("Numéro de vol", "1010")
        with c2:
            mat = st.selectbox("Matricule", mats, index=mats.index("7T-VCA") if "7T-VCA" in mats else 0)
            dep = st.selectbox("Aéroport départ", apts, index=apts.index("ALG") if "ALG" in apts else 0)
        with c3:
            arr = st.selectbox("Aéroport arrivée", apts, index=apts.index("HME") if "HME" in apts else 0)
            st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
            run = st.button("🔍 Analyser le vol", type="primary", use_container_width=True)

    if not run:
        return

    try:
        res, df_ref = score_flight(date, mat, numvol, dep, arr)
    except Exception as e:
        st.error(f"Scoring réel indisponible (base ou modèles .pkl introuvables). Détail : {e}")
        st.info("Vérifiez la connexion SQL Server et la présence de "
                "`models/isolation_forest.pkl`, `random_forest.pkl`, `label_encoders.pkl`.")
        return

    if_bad = res["if_statut"] != "Normal"
    rf_bad = bool(res["rf_anom"])
    final_bad = if_bad or rf_bad

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    v1, v2, v3 = st.columns(3)
    with v1:
        cls = "bad" if final_bad else "ok"
        txt = "⚠️ Anomalie détectée" if final_bad else "✅ Vol cohérent"
        st.markdown(f'<div class="verdict {cls}"><div class="big">{txt}</div>'
                    f'<div style="opacity:.9;margin-top:6px;">Verdict combiné</div></div>',
                    unsafe_allow_html=True)
    with v2:
        cls = "bad" if if_bad else "ok"
        st.markdown(f'<div class="verdict {cls}"><div class="big">Isolation Forest</div>'
                    f'<div style="margin-top:6px;">{res["if_statut"]}</div>'
                    f'<div style="opacity:.85;font-size:12px;">score {res["score"]:.4f}</div></div>',
                    unsafe_allow_html=True)
    with v3:
        if res["rf_proba"] is None:
            st.markdown('<div class="verdict na"><div class="big">Random Forest</div>'
                        '<div style="margin-top:6px;">modèle absent</div></div>', unsafe_allow_html=True)
        else:
            cls = "bad" if rf_bad else "ok"
            lab = "Anomalie" if rf_bad else "Normal"
            st.markdown(f'<div class="verdict {cls}"><div class="big">Random Forest</div>'
                        f'<div style="margin-top:6px;">{lab}</div>'
                        f'<div style="opacity:.85;font-size:12px;">P(anomalie) '
                        f'{res["rf_proba"]*100:.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    d1, d2 = st.columns([1, 1])
    with d1:
        with st.container(border=True):
            st.markdown('<div class="ttl">Diagnostic</div>', unsafe_allow_html=True)
            if final_bad:
                st.markdown(f"<span class='pill r'>{res['type']}</span>", unsafe_allow_html=True)
                st.markdown(f"<div style='margin-top:10px;color:{INK};font-size:13px;'>"
                            f"{res['raison']}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='pill g'>Conforme à l'historique</span>", unsafe_allow_html=True)
                st.markdown(f"<div style='margin-top:10px;color:{MUTED};font-size:13px;'>"
                            "Le vol correspond aux schémas connus (numéro, route, "
                            "famille d'appareil et jour habituels).</div>", unsafe_allow_html=True)
    with d2:
        with st.container(border=True):
            st.markdown('<div class="ttl">Historique de ce vol</div>', unsafe_allow_html=True)
            h = df_ref[df_ref["NumVol"].astype(str) == str(numvol).strip().upper()]
            if h.empty:
                st.markdown(f"<div class='sub'>Numéro {numvol} jamais observé dans "
                            "les vols validés.</div>", unsafe_allow_html=True)
            else:
                deps = ", ".join(sorted(h["AeroDepart"].astype(str).unique())[:6])
                arrs = ", ".join(sorted(h["AeroArriv"].astype(str).unique())[:6])
                mts  = ", ".join(sorted(h["Matricule"].astype(str).unique())[:6])
                jn = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
                jours = ", ".join(jn[j] for j in sorted(h["Date"].dt.dayofweek.unique()))
                st.markdown(
                    f"<div style='font-size:13px;line-height:1.9;'>"
                    f"<b>{len(h)}</b> vols historiques<br>"
                    f"<span class='sub'>Départs :</span> {deps}<br>"
                    f"<span class='sub'>Arrivées :</span> {arrs}<br>"
                    f"<span class='sub'>Matricules :</span> {mts}<br>"
                    f"<span class='sub'>Jours :</span> {jours}</div>", unsafe_allow_html=True)


# ============================================================================
#  MAIN
# ============================================================================
def main():
    st.set_page_config(page_title="Domestic Airlines — Fiabilité",
                       page_icon="✈️", layout="wide", initial_sidebar_state="collapsed")

    if "page" not in st.session_state:
        st.session_state.page = "home"

    qp = st.query_params
    if "refresh" in qp:
        st.cache_data.clear()
    if "nav" in qp:
        st.session_state.page = qp.get("nav")
    for kk in ("nav", "refresh"):
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
        pagehead("Vue d'ensemble", "Réconciliation AIMS ↔ MRO et indice de fiabilité des données.", "📊")
        page_overview()
    elif page == "dashboard":
        pagehead("Détection — Statistiques", "Validés vs anomalies, avec filtres et vues dynamiques.", "📈")
        page_dashboard()
    elif page == "tables":
        pagehead("Exploration SQL", "Consultation des tables sources et des résultats du pipeline.", "🗃️")
        page_tables()
    elif page == "detection":
        pagehead("Détection d'anomalie", "Scoring d'un vol par Isolation Forest & Random Forest.", "🔍")
        page_detection()


if __name__ == "__main__":
    main()