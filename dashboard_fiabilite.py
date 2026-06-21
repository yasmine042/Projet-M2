# ============================================================================
#  dashboard_fiabilite.py
#  Tableau de bord — Fiabilité des données analytiques aériennes (AIMS / MRO)
#  Lancement :  streamlit run dashboard_fiabilite.py
# ============================================================================

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from config import DB_CONFIG
from db import read_query

SQL_SERVER = DB_CONFIG["server"]
SQL_DB     = DB_CONFIG["database"]

# ─── Palette ────────────────────────────────────────────────────────────────
BG       = "#0e0d10"
CARD     = "#19181c"
BORDER   = "#262329"
ACCENT   = "#ff7e1d"
ACCENT_2 = "#ffa24c"
TRACK    = "#2a2730"
TEXT     = "#f5f4f6"
MUTED    = "#8d8a93"
GREEN    = "#34c759"
RED      = "#ff453a"
YELLOW   = "#ffd60a"

# ============================================================================
#  COUCHE DONNÉES  (utilise db.py → SQLAlchemy + config.py)
# ============================================================================

def _q(sql: str) -> pd.DataFrame | None:
    try:
        return read_query(sql)
    except Exception:
        return None


def _scalar(sql: str, default: int = 0) -> int:
    r = _q(sql)
    if r is not None and not r.empty:
        return int(r.iloc[0, 0])
    return default


def _rf_auc() -> float:
    try:
        p = Path("models/rf_metrics.json")
        if p.exists():
            return float(json.loads(p.read_text())["auc"])
    except Exception:
        pass
    return 0.0


@st.cache_data(ttl=300, show_spinner=False)
def load_kpis() -> dict:
    d = {
        "valides_total":  0,
        "anomalies_aims": 0,
        "anomalies_mro":  0,
        "manques":        0,
        "dates_susp":     0,
        "faux_negatifs":  0,
        "auc_rf":         _rf_auc(),
        "etapes":         {},
        "total":          0,
    }

    # Vols validés par étape
    df = _q("SELECT EtapeValidation, COUNT(*) n FROM VolsValidesEtape1 "
            "GROUP BY EtapeValidation")
    if df is not None and not df.empty:
        etapes = {int(r.EtapeValidation): int(r.n)
                  for r in df.itertuples() if str(r.EtapeValidation).isdigit()}
        if etapes:
            d["etapes"] = etapes
            d["valides_total"] = sum(etapes.values())

    # Anomalies détectées par Isolation Forest
    d["anomalies_aims"] = _scalar("SELECT COUNT(*) n FROM AnomaliesAIMS")
    d["anomalies_mro"]  = _scalar("SELECT COUNT(*) n FROM AnomaliesMRO")

    # Vols manquants — table Manque si elle existe, sinon écart AIMS total
    r = _q("SELECT COUNT(*) n FROM Manque")
    if r is not None and not r.empty:
        d["manques"] = int(r.iloc[0, 0])
    else:
        total_aims = _scalar("SELECT COUNT(*) n FROM AIMS")
        d["manques"] = max(0, total_aims - d["valides_total"] - d["anomalies_aims"])

    # Dates suspectes (TypeAnomalie contient 'Date')
    d["dates_susp"] = (
        _scalar("SELECT COUNT(*) n FROM AnomaliesAIMS WHERE TypeAnomalie LIKE '%Date%'") +
        _scalar("SELECT COUNT(*) n FROM AnomaliesMRO  WHERE TypeAnomalie LIKE '%Date%'")
    )

    # Faux négatifs rattrapés par le Random Forest
    d["faux_negatifs"] = (
        _scalar("SELECT COUNT(*) n FROM FauxNegatifsAIMS") +
        _scalar("SELECT COUNT(*) n FROM FauxNegatifsMRO")
    )

    d["total"] = (d["valides_total"] + d["anomalies_aims"]
                  + d["anomalies_mro"] + d["manques"])
    return d


@st.cache_data(ttl=300, show_spinner=False)
def load_daily_volume() -> pd.DataFrame:
    df = _q(
        "SELECT CAST(DateAIMS AS date) date, COUNT(*) vols "
        "FROM VolsValidesEtape1 "
        "WHERE DateAIMS >= DATEADD(day,-7,GETDATE()) "
        "GROUP BY CAST(DateAIMS AS date) ORDER BY date"
    )
    return df if (df is not None and not df.empty) else pd.DataFrame(columns=["date", "vols"])


@st.cache_data(ttl=300, show_spinner=False)
def load_daily_anom() -> pd.DataFrame:
    df = _q(
        "SELECT CAST(DateAIMS AS date) date, COUNT(*) anomalies "
        "FROM AnomaliesAIMS "
        "WHERE DateAIMS >= DATEADD(day,-12,GETDATE()) "
        "GROUP BY CAST(DateAIMS AS date) ORDER BY date"
    )
    return df if (df is not None and not df.empty) else pd.DataFrame(columns=["date", "anomalies"])


# ============================================================================
#  INDICATEURS DÉRIVÉS
# ============================================================================

def compute_indicators(k: dict) -> tuple[float, float]:
    total = max(k["total"], 1)
    taux_reconc = 100 * k["valides_total"] / total
    penalite = (
        1.6 * (k["anomalies_aims"] + k["anomalies_mro"]) / total +
        2.2 * k["manques"]    / total +
        1.0 * k["dates_susp"] / total
    ) * 100
    return taux_reconc, max(0.0, min(100.0, 100.0 - penalite))


def quality_label(score: float) -> tuple[str, str]:
    if score >= 95: return "Fiabilité excellente", GREEN
    if score >= 90: return "Fiabilité élevée",     GREEN
    if score >= 80: return "Fiabilité correcte",   YELLOW
    if score >= 65: return "À surveiller",          ACCENT
    return "Fiabilité dégradée", RED


# ============================================================================
#  FIGURES PLOTLY
# ============================================================================

def fig_ring(value: float, center_label: str, sub_label: str, color: str):
    value = max(0.0, min(100.0, value))
    fig = go.Figure(go.Pie(
        values=[value, 100 - value],
        hole=0.74, sort=False, direction="clockwise", rotation=0,
        marker=dict(colors=[color, TRACK], line=dict(color=BG, width=2)),
        textinfo="none", hoverinfo="skip",
    ))
    fig.update_layout(
        showlegend=False, margin=dict(t=4, b=4, l=4, r=4),
        height=230, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        annotations=[
            dict(text=center_label, x=0.5, y=0.56, showarrow=False,
                 font=dict(size=40, color=TEXT, family="Arial Black")),
            dict(text=sub_label, x=0.5, y=0.38, showarrow=False,
                 font=dict(size=13, color=color)),
        ],
    )
    return fig


def fig_trend(df: pd.DataFrame):
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            height=230, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text="Données indisponibles", x=0.5, y=0.5,
                              showarrow=False, font=dict(color=MUTED, size=13))])
        return fig
    x = pd.to_datetime(df["date"]).dt.strftime("%d/%m")
    y = df["anomalies"].astype(int)
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="lines+markers+text",
        line=dict(color=ACCENT, width=3, shape="spline"),
        marker=dict(size=7, color=ACCENT, line=dict(color=BG, width=2)),
        fill="tozeroy", fillcolor="rgba(255,126,29,0.12)",
        text=y, textposition="top center",
        textfont=dict(color=MUTED, size=11), hoverinfo="skip",
    ))
    fig.update_layout(
        height=230, margin=dict(t=24, b=8, l=8, r=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color=MUTED, tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   color=MUTED, zeroline=False,
                   range=[0, max(int(y.max()) * 1.45, 4)]),
    )
    return fig


# ============================================================================
#  COMPOSANTS HTML
# ============================================================================

def css():
    st.markdown(f"""
    <style>
      .stApp {{ background:{BG}; }}
      #MainMenu, footer, header {{ visibility:hidden; }}
      .block-container {{ padding:1.4rem 1.6rem 2rem; max-width:1480px; }}
      [data-testid="stVerticalBlockBorderWrapper"] {{
          background:{CARD}; border:1px solid {BORDER} !important;
          border-radius:20px; padding:14px 16px;
      }}
      .ttl {{ color:{TEXT}; font-size:15px; font-weight:600; margin-bottom:4px; }}
      .card {{ background:{CARD}; border:1px solid {BORDER}; border-radius:20px;
               padding:16px 18px; }}
      .hero {{ background:linear-gradient(140deg,{ACCENT} 0%,#ff5436 100%);
               border-radius:24px; padding:22px 24px; color:#fff;
               box-shadow:0 10px 30px rgba(255,90,50,.25); }}
      .chip {{ display:inline-block; background:rgba(255,255,255,.16);
               border-radius:14px; padding:8px 12px; margin-right:8px;
               font-size:12px; }}
      .chip b {{ display:block; font-size:15px; }}
      .day {{ flex:1; text-align:center; background:{CARD};
              border:1px solid {BORDER}; border-radius:16px; padding:12px 6px; }}
      .day .dn {{ color:{MUTED}; font-size:11px; }}
      .day .ic {{ font-size:20px; margin:6px 0; }}
      .day .vv {{ color:{TEXT}; font-weight:700; font-size:14px; }}
      .kpi {{ background:{CARD}; border:1px solid {BORDER}; border-radius:18px;
              padding:16px; }}
      .kpi .lab {{ color:{MUTED}; font-size:12px; }}
      .kpi .val {{ color:{TEXT}; font-size:26px; font-weight:800; margin-top:6px; }}
      .kpi .sub {{ font-size:11px; margin-top:2px; }}
      .dot {{ height:9px; width:9px; border-radius:50%; display:inline-block;
              margin-right:8px; }}
      .met {{ display:flex; align-items:center; padding:7px 0; }}
      .met .mv {{ color:{TEXT}; font-weight:700; font-size:17px; min-width:54px; }}
      .met .ml {{ color:{MUTED}; font-size:12px; }}
      .barrow {{ display:flex; align-items:center; gap:10px; margin:9px 0; }}
      .barrow .bl {{ color:{MUTED}; font-size:12px; width:78px; }}
      .bartrack {{ flex:1; height:8px; background:{TRACK}; border-radius:6px; }}
      .barfill {{ height:8px; border-radius:6px;
                  background:linear-gradient(90deg,{ACCENT},{ACCENT_2}); }}
      .barpct {{ color:{TEXT}; font-size:12px; width:46px; text-align:right; }}
    </style>""", unsafe_allow_html=True)


def hero(taux: float, fiab: float, k: dict, updated: str) -> str:
    lab, _ = quality_label(fiab)
    anom   = k["anomalies_aims"] + k["anomalies_mro"]
    return f"""
    <div class="hero">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div style="font-size:14px;">📍 Domestic Airlines — {SQL_DB}</div>
        <div style="font-size:11px;opacity:.85;">MAJ {updated}</div>
      </div>
      <div style="font-size:13px;opacity:.9;margin-top:18px;">Taux de réconciliation AIMS ↔ MRO</div>
      <div style="font-size:58px;font-weight:800;line-height:1;margin:4px 0;">{taux:.1f}%</div>
      <div style="font-size:16px;font-weight:600;">{lab}</div>
      <div style="margin-top:20px;">
        <span class="chip">Total vols<b>{k['total']:,}</b></span>
        <span class="chip">Validés<b>{k['valides_total']:,}</b></span>
        <span class="chip">Anomalies<b>{anom:,}</b></span>
      </div>
    </div>""".replace(",", " ")


def week_strip(df: pd.DataFrame) -> str:
    if df.empty:
        return (f'<div style="color:{MUTED};font-size:13px;padding:16px 0;">'
                'Volume journalier indisponible</div>')
    cells = ""
    for r in df.itertuples():
        d = pd.to_datetime(r.date)
        cells += (f'<div class="day"><div class="dn">{d.strftime("%a")}</div>'
                  f'<div class="ic">✈️</div><div class="vv">{int(r.vols)}</div></div>')
    return f'<div style="display:flex;gap:10px;">{cells}</div>'


def kpi_grid(k: dict) -> str:
    def cell(lab, val, sub, color):
        return (f'<div class="kpi"><div class="lab">{lab}</div>'
                f'<div class="val">{val}</div>'
                f'<div class="sub" style="color:{color};">{sub}</div></div>')
    anom = k["anomalies_aims"] + k["anomalies_mro"]
    return f"""<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
      {cell("✅ Vols validés",    f"{k['valides_total']:,}".replace(","," "), "rapprochés 5 systèmes", GREEN)}
      {cell("⚠️ Anomalies IF",   f"{anom:,}".replace(","," "), f"AIMS {k['anomalies_aims']} · MRO {k['anomalies_mro']}", ACCENT)}
      {cell("🔍 Vols manquants", f"{k['manques']:,}".replace(","," "), "absents côté MRO", RED)}
      {cell("📅 Dates suspectes",f"{k['dates_susp']:,}".replace(","," "), "TypeAnomalie contient Date", YELLOW)}
    </div>"""


def metrics_grid(k: dict) -> str:
    items = [
        (RED,      k["anomalies_aims"],  "Anomalies AIMS"),
        (ACCENT,   k["anomalies_mro"],   "Anomalies MRO"),
        (YELLOW,   k["manques"],         "Vols manquants"),
        (ACCENT_2, k["dates_susp"],      "Dates suspectes"),
        (GREEN,    k["faux_negatifs"],   "Faux Négatifs RF"),
        (GREEN,    f"{k['auc_rf']:.2f}", "AUC Random Forest"),
    ]
    rows = ""
    for col, val, lab in items:
        v = f"{val:,}".replace(",", " ") if isinstance(val, int) else val
        rows += (f'<div class="met"><span class="dot" style="background:{col};"></span>'
                 f'<span class="mv">{v}</span><span class="ml">{lab}</span></div>')
    return f'<div>{rows}</div>'


def sources_card(k: dict) -> str:
    aims_t = k["valides_total"] + k["anomalies_aims"]
    mro_t  = k["valides_total"] + k["anomalies_mro"]
    return f"""<div class="card">
      <div class="ttl">Sources de données</div>
      <div class="met" style="padding:14px 0;">
        <span style="font-size:26px;margin-right:14px;">🛫</span>
        <div><div style="color:{MUTED};font-size:12px;">AIMS — Opérations</div>
        <div style="color:{TEXT};font-size:22px;font-weight:800;">{aims_t:,}</div></div>
      </div>
      <div style="height:1px;background:{BORDER};"></div>
      <div class="met" style="padding:14px 0;">
        <span style="font-size:26px;margin-right:14px;">🔧</span>
        <div><div style="color:{MUTED};font-size:12px;">MRO — Maintenance</div>
        <div style="color:{TEXT};font-size:22px;font-weight:800;">{mro_t:,}</div></div>
      </div>
    </div>""".replace(",", " ")


def stages_card(k: dict) -> str:
    total = max(k["total"], 1)
    rows  = ""
    for et, n in sorted(k["etapes"].items()):
        pct   = 100 * n / total
        rows += (f'<div class="barrow"><div class="bl">Étape {et}</div>'
                 f'<div class="bartrack"><div class="barfill" style="width:{max(pct,1):.0f}%;"></div></div>'
                 f'<div class="barpct">{pct:.1f}%</div></div>')
    if k["etapes"]:
        non = k["anomalies_aims"] + k["manques"]
        pct = 100 * non / total
        rows += (f'<div class="barrow"><div class="bl" style="color:{RED};">Non rappr.</div>'
                 f'<div class="bartrack"><div class="barfill" style="width:{max(pct,1):.0f}%;'
                 f'background:linear-gradient(90deg,{RED},#ff7a6e);"></div></div>'
                 f'<div class="barpct">{pct:.1f}%</div></div>')
    else:
        rows = f'<div style="color:{MUTED};font-size:12px;padding:8px 0;">Données d\'étapes indisponibles</div>'
    return (f'<div class="card"><div class="ttl">Répartition par étape de validation</div>'
            f'<div style="margin-top:6px;">{rows}</div></div>')


# ============================================================================
#  PAGE
# ============================================================================

def main():
    st.set_page_config(page_title="Fiabilité données aériennes",
                       page_icon="✈️", layout="wide")
    css()

    with st.spinner("Chargement des données SQL Server…"):
        k = load_kpis()

    if k["total"] == 0:
        st.warning(f"⚠️  Connexion SQL Server indisponible — {SQL_SERVER} / {SQL_DB}")

    taux, fiab = compute_indicators(k)
    lab, col   = quality_label(fiab)
    updated    = dt.datetime.now().strftime("%d/%m %H:%M")

    vol_df  = load_daily_volume()
    anom_df = load_daily_anom()

    col_l, col_m, col_r = st.columns([1.0, 1.55, 1.0], gap="medium")

    with col_l:
        st.markdown(hero(taux, fiab, k, updated), unsafe_allow_html=True)
        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        st.markdown(kpi_grid(k), unsafe_allow_html=True)

    with col_m:
        st.markdown(week_strip(vol_df), unsafe_allow_html=True)
        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="ttl">Anomalies détectées par jour</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_trend(anom_df), use_container_width=True,
                            config={"displayModeBar": False})
        with st.container(border=True):
            st.markdown('<div class="ttl">Indice de fiabilité des données</div>',
                        unsafe_allow_html=True)
            g1, g2 = st.columns([1, 1])
            with g1:
                st.plotly_chart(fig_ring(fiab, f"{fiab:.0f}", lab, col),
                                use_container_width=True,
                                config={"displayModeBar": False})
            with g2:
                st.markdown(metrics_grid(k), unsafe_allow_html=True)
            st.markdown(
                f"<div style='color:{MUTED};font-size:12px;margin-top:4px;'>"
                "Score composite : pénalise anomalies, vols manquants et dates "
                "incohérentes (0 = critique, 100 = parfait).</div>",
                unsafe_allow_html=True)

    with col_r:
        st.markdown(sources_card(k), unsafe_allow_html=True)
        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        st.markdown(stages_card(k), unsafe_allow_html=True)

    src = "SQL Server" if _q("SELECT 1 AS n") is not None else "⚠️ BD indisponible"
    st.markdown(
        f"<div style='color:{MUTED};font-size:11px;text-align:right;margin-top:10px;'>"
        f"Source : {src} · {SQL_SERVER} / {SQL_DB}</div>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
