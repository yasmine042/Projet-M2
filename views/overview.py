import math

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from ui import (
    global_kpis, load_all_anomalies, load_faux_negatifs,
    split_trend, _light, _tint,
    fnum,
    GREEN, AMBER, CRIMSON, BLUE, TEAL, VIOLET, ROSE, OLIVE, INK, MUTED,
)

# ─────────────────────────────────────────────────────────────────────────────
#  CSS — injecté une seule fois via st.markdown avec unsafe_allow_html.
#  Streamlit wraps each st.markdown in its own shadow-like div, but <style>
#  blocks still bubble up to the document head, so one inject suffices.
# ─────────────────────────────────────────────────────────────────────────────
_OV_CSS = """
<style>
/* ── Bootstrap Icons (fallback if not already loaded by app.py) ── */
@import url("https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css");

@keyframes ovRise {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Hero santé (feu tricolore) ── */
.ov-hero {
    position: relative;
    display: flex;
    align-items: center;
    gap: 30px;
    background: linear-gradient(135deg, #ffffff 0%, #fbfdfa 55%, #f4f8ef 100%);
    border: 1px solid #e6eaef;
    border-radius: 22px;
    padding: 26px 32px 26px 36px;
    box-shadow: 0 8px 30px rgba(20,40,25,.07);
    overflow: hidden;
    margin-bottom: 8px;
    animation: ovRise .5s ease-out both;
}
.ov-hero-glow {
    position: absolute; right: -60px; top: -70px;
    width: 280px; height: 280px; border-radius: 50%;
    pointer-events: none; z-index: 0;
}
.ov-hero-accent {
    position: absolute; left: 0; top: 0; bottom: 0; width: 5px; z-index: 1;
}
.ov-hero-ring {
    position: relative; z-index: 1; flex-shrink: 0;
    width: 128px; height: 128px;
}
.ov-hero-ring .pctxt {
    position: absolute; inset: 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
}
.ov-hero-pct {
    font-size: 28px; font-weight: 900; color: #1f2a33;
    letter-spacing: -.02em; line-height: 1; font-variant-numeric: tabular-nums;
}
.ov-hero-pctlbl {
    font-size: 9.5px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .09em; color: #7b8794; margin-top: 4px;
}
.ov-hero-body { position: relative; z-index: 1; flex: 1; min-width: 0; }
.ov-hero-status {
    display: inline-flex; align-items: center; gap: 9px;
    font-size: 21px; font-weight: 800; line-height: 1.1; letter-spacing: -.01em;
}
.ov-hero-desc {
    font-size: 13px; color: #7b8794; margin-top: 6px;
    max-width: 540px; line-height: 1.55;
}
.ov-hero-stats { display: flex; gap: 11px; margin-top: 16px; flex-wrap: wrap; }
.ov-hero-pill {
    display: flex; align-items: center; gap: 10px;
    background: #ffffff; border: 1px solid #e8ecf0; border-radius: 13px;
    padding: 9px 15px 9px 11px;
    box-shadow: 0 1px 6px rgba(20,40,25,.04);
    transition: transform .18s ease, box-shadow .18s ease;
}
.ov-hero-pill:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(20,40,25,.09); }
.ov-hero-pill .pi {
    width: 32px; height: 32px; border-radius: 9px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 15px;
}
.ov-hero-pill .pv {
    font-size: 18px; font-weight: 800; color: #1f2a33; line-height: 1;
    font-variant-numeric: tabular-nums;
}
.ov-hero-pill .pl { font-size: 10.5px; color: #7b8794; margin-top: 3px; }

/* ── Section header ── */
.ov-section {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 30px 0 14px;
}
.ov-section-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.ov-section-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: #1f2a33;
    white-space: nowrap;
}
.ov-section-line {
    flex: 1;
    height: 1.5px;
    border-radius: 2px;
    opacity: .20;
}

/* ── KPI grid ── */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin-bottom: 2px;
}

/* ── KPI card ── */
.kpi2 {
    background: #ffffff;
    border: 1px solid #e8ecf0;
    border-top: 2px solid #3b6fe0;
    border-radius: 0 0 18px 18px;
    padding: 18px 18px 14px;
    box-shadow: 0 1px 8px rgba(20,40,25,.04);
    display: flex;
    flex-direction: column;
    position: relative;
    overflow: hidden;
    transition: transform .22s ease, box-shadow .22s ease;
    min-width: 0;
    animation: ovRise .45s ease-out both;
}
.kpi2:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 26px rgba(20,40,25,.10);
}
.kpi2-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 12px;
}
.kpi2-icon {
    width: 42px; height: 42px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 19px;
    flex-shrink: 0;
}
.kpi2-badge {
    font-size: 10.5px;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 20px;
    white-space: nowrap;
    align-self: flex-start;
}
.kpi2-badge.up   { background: rgba(212,12,28,.08);  color: #c00018; }
.kpi2-badge.down { background: rgba(63,115,32,.09);  color: #2d6a10; }
.kpi2-badge.neu  { background: #f4f6f9; color: #8a95a0; }

.kpi2-value {
    font-size: 34px;
    font-weight: 900;
    color: #1f2a33;
    line-height: 1.05;
    letter-spacing: -.025em;
    font-variant-numeric: tabular-nums;
}
.kpi2-value small {
    font-size: 17px;
    font-weight: 600;
    opacity: .50;
}
.kpi2-label {
    font-size: 12px;
    font-weight: 600;
    color: #7b8794;
    margin-top: 5px;
}
.kpi2-footer {
    margin-top: 12px;
    padding-top: 10px;
    border-top: 1px solid #f0f2f5;
    font-size: 11px;
    color: #7b8794;
    display: flex;
    align-items: center;
    gap: 5px;
}
.fdot {
    width: 7px; height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* clickable card */
a.kpi2-link {
    text-decoration: none !important;
    color: inherit !important;
    display: block;
}

/* ── Synthèse qualité (bottom row) ── */
.qrow {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin-bottom: 2px;
}
.qcard {
    background: #ffffff;
    border: 1px solid #e6eaef;
    border-radius: 14px;
    padding: 15px 16px;
    box-shadow: 0 2px 10px rgba(20,40,25,.04);
    transition: transform .2s ease;
}
.qcard:hover { transform: translateY(-2px); }
.qcard-head {
    display: flex;
    align-items: center;
    gap: 9px;
    margin-bottom: 10px;
}
.qcard-icon {
    width: 32px; height: 32px;
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
    flex-shrink: 0;
}
.qcard-lbl { font-size: 11px; font-weight: 600; color: #7b8794; }
.qcard-val { font-size: 28px; font-weight: 900; color: #1f2a33; letter-spacing: -.02em; line-height: 1; }
.qcard-sub { font-size: 10.5px; color: #7b8794; margin-top: 4px; }

/* ── Chart card ── */
.ov-chart-card {
    background: #ffffff;
    border: 1px solid #e6eaef;
    border-radius: 18px;
    padding: 18px 20px 12px;
    box-shadow: 0 2px 12px rgba(20,40,25,.04);
}
.ov-chart-title { font-size: 14px; font-weight: 700; color: #1f2a33; margin-bottom: 2px; }
.ov-chart-sub   { font-size: 11px; color: #7b8794; margin-bottom: 10px; }
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers HTML
# ─────────────────────────────────────────────────────────────────────────────

def _rgba(hexc: str, a: float) -> str:
    r, g, b = int(hexc[1:3], 16), int(hexc[3:5], 16), int(hexc[5:7], 16)
    return f"rgba({r},{g},{b},{a})"


def _health_hero(taux, n_nt, n_crit, n_mois, lbl_mois,
                 ok_col, warn_col, bad_col) -> str:
    """Bandeau de santé global (feu tricolore) basé sur le taux de réconciliation."""
    if taux >= 95:
        col, status, icon = ok_col, "Fiabilité élevée", "shield-fill-check"
        desc = ("Les données AIMS et MRO sont fortement réconciliées. "
                )
    elif taux >= 85:
        col, status, icon = warn_col, "Fiabilité à surveiller", "shield-fill-exclamation"
        desc = ("Des écarts subsistent entre AIMS et MRO. "
                "Un suivi des anomalies non traitées est recommandé.")
    else:
        col, status, icon = bad_col, "Fiabilité critique", "shield-fill-x"
        desc = ("Le taux de réconciliation est insuffisant. "
                "Une revue des écarts AIMS↔MRO est prioritaire.")

    C   = 2 * math.pi * 52
    off = C * (1 - max(0.0, min(100.0, taux)) / 100)

    def _pill(bi, color, value, label):
        return (
            f'<div class="ov-hero-pill">'
            f'<div class="pi" style="background:{_rgba(color,.10)};color:{color};">'
            f'<i class="bi bi-{bi}"></i></div>'
            f'<div><div class="pv">{value}</div><div class="pl">{label}</div></div></div>')

    pills = (
        _pill("bell-fill",          ROSE,    fnum(n_nt),   "Non traitées")
        + _pill("shield-exclamation", CRIMSON, fnum(n_crit), "Critiques")
        + _pill("calendar3",        AMBER,   fnum(n_mois),  f"Ce mois · {lbl_mois}")
    )

    return (
        f'<div class="ov-hero">'
        f'<div class="ov-hero-glow" style="background:radial-gradient(circle at center,'
        f'{_rgba(col,.13)},transparent 70%);"></div>'
        f'<div class="ov-hero-accent" style="background:{col};"></div>'
        f'<div class="ov-hero-ring">'
        f'<svg width="128" height="128" viewBox="0 0 124 124">'
        f'<circle cx="62" cy="62" r="52" fill="none" stroke="#edf1f5" stroke-width="11"/>'
        f'<circle cx="62" cy="62" r="52" fill="none" stroke="{col}" stroke-width="11" '
        f'stroke-linecap="round" stroke-dasharray="{C:.1f}" stroke-dashoffset="{off:.1f}" '
        f'transform="rotate(-90 62 62)"/></svg>'
        f'<div class="pctxt"><div class="ov-hero-pct">{taux:.1f}%</div>'
        f'<div class="ov-hero-pctlbl">Réconciliation</div></div></div>'
        f'<div class="ov-hero-body">'
        f'<div class="ov-hero-status" style="color:{col};">'
        f'<i class="bi bi-{icon}"></i>{status}</div>'
        f'<div class="ov-hero-desc">{desc}</div>'
        f'<div class="ov-hero-stats">{pills}</div>'
        f'</div></div>')


def _section_hdr(title: str, accent: str) -> str:
    return (
        f'<div class="ov-section">'
        f'<div class="ov-section-label">'
        f'<div class="ov-section-dot" style="background:{accent};"></div>'
        f'{title}'
        f'</div>'
        f'<div class="ov-section-line" style="background:{accent};"></div>'
        f'</div>'
    )


def _badge(val: int, zero_is_good: bool = True) -> str:
    """Trend badge: zero_is_good=True means positive delta → bad (red)."""
    if val > 0:
        cls  = "up"   if zero_is_good else "down"
        text = f"↑ +{val}"
    elif val < 0:
        cls  = "down" if zero_is_good else "up"
        text = f"↓ {val}"
    else:
        cls  = "neu"
        text = "— stable"
    return f'<span class="kpi2-badge {cls}">{text}</span>'


def _kpi(
    label: str, value: str, footer: str,
    color: str, bi_icon: str,
    badge_html: str = "",
    href: str = "",
    fdot_color: str = "#7b8794",
) -> str:
    # Icône : fond très léger (.07 au lieu de .13), couleur légèrement désaturée
    icon_bg = _rgba(color, .08)
    # Bordure top : fine (2px) et légèrement transparente pour ne pas crier
    inner = (
        f'<div class="kpi2" style="border-top-color:{color};">'
        f'  <div class="kpi2-header">'
        f'    <div class="kpi2-icon" style="background:{icon_bg};">'
        f'      <i class="bi bi-{bi_icon}" style="color:{color};font-size:18px;line-height:1;opacity:.85;"></i>'
        f'    </div>'
        f'    {badge_html}'
        f'  </div>'
        f'  <div class="kpi2-value">{value}</div>'
        f'  <div class="kpi2-label">{label}</div>'
        f'  <div class="kpi2-footer">'
        f'    <div class="fdot" style="background:{fdot_color};"></div>'
        f'    {footer}'
        f'  </div>'
        f'</div>'
    )
    if href:
        return f'<a class="kpi2-link" href="{href}" target="_self">{inner}</a>'
    return inner


def _kpi_grid(cards: list) -> str:
    return f'<div class="kpi-row">{"".join(cards)}</div>'


def _qcard(label: str, value: str, sub: str, color: str, bi_icon: str) -> str:
    icon_bg = _rgba(color, .08)
    return (
        f'<div class="qcard" style="border-top:2px solid {color};">'
        f'  <div class="qcard-head">'
        f'    <div class="qcard-icon" style="background:{icon_bg};color:{color};opacity:.9;">'
        f'      <i class="bi bi-{bi_icon}" style="font-size:15px;line-height:1;"></i>'
        f'    </div>'
        f'    <span class="qcard-lbl">{label}</span>'
        f'  </div>'
        f'  <div class="qcard-val">{value}</div>'
        f'  <div class="qcard-sub">{sub}</div>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Page principale
# ─────────────────────────────────────────────────────────────────────────────

def page_overview():
    # Inject global CSS once
    st.markdown(_OV_CSS, unsafe_allow_html=True)

    # ── Données ───────────────────────────────────────────────────────────
    k    = global_kpis()
    anom = load_all_anomalies()

    non_traitees = anom[anom["Traitee"] == 0] if "Traitee" in anom.columns else anom.copy()

    if (not non_traitees.empty
            and "Date" in non_traitees.columns
            and non_traitees["Date"].notna().any()):
        dernier_mois = non_traitees["Date"].dropna().dt.to_period("M").max()
        anom_ce_mois = non_traitees[non_traitees["Date"].dt.to_period("M") == dernier_mois]
        lbl_mois     = str(dernier_mois)
        try:
            mois_prec  = dernier_mois - 1
            anom_prec  = non_traitees[non_traitees["Date"].dt.to_period("M") == mois_prec]
            delta_mois = len(anom_ce_mois) - len(anom_prec)
        except Exception:
            delta_mois = 0
    else:
        anom_ce_mois = pd.DataFrame()
        lbl_mois     = "—"
        delta_mois   = 0

    critiques = (
        non_traitees[non_traitees["Statut"] == "Très Suspect"]
        if not non_traitees.empty else pd.DataFrame()
    )

    taux_aims = k["taux"]
    total_aims = max(k["total_aims"], 1)
    total_mro  = max(k["total_mro"],  1)
    taux_mro   = 100 * k["valides"] / total_mro

    pct_nm_aims = int(k["nm_aims"] * 100 // total_aims)
    pct_nm_mro  = int(k["nm_mro"]  * 100 // total_mro)

    # ── Couleurs des points de statut ─────────────────────────────────────
    ok_col   = "#3f7320"
    warn_col = "#e8920c"
    bad_col  = "#D40C1C"
    neu_col  = "#7b8794"

    rec_dot  = ok_col  if taux_aims >= 95 else (warn_col if taux_aims >= 80 else bad_col)
    nt_dot   = ok_col  if len(non_traitees) == 0 else (warn_col if len(non_traitees) < 50 else bad_col)
    crit_dot = ok_col  if len(critiques) == 0 else (warn_col if len(critiques) < 10 else bad_col)

    # ─────────────────────────────────────────────────────────────────────
    #  BLOC 0 — Hero santé (feu tricolore)
    # ─────────────────────────────────────────────────────────────────────
    st.markdown(
        _health_hero(taux_aims, len(non_traitees), len(critiques),
                     len(anom_ce_mois), lbl_mois, ok_col, warn_col, bad_col),
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────────────────
    #  BLOC 1 — Volume des données
    # ─────────────────────────────────────────────────────────────────────
    html_vol = _section_hdr("Volume des données", BLUE)
    html_vol += _kpi_grid([
        _kpi(
            label    = "Vols AIMS",
            value    = fnum(k["total_aims"]),
            footer   = "Système opérations",
            color    = BLUE,
            bi_icon  = "airplane-fill",
            fdot_color = ok_col,
        ),
        _kpi(
            label    = "Vols MRO",
            value    = fnum(k["total_mro"]),
            footer   = "Système maintenance",
            color    = TEAL,
            bi_icon  = "tools",
            fdot_color = ok_col,
        ),
        _kpi(
            label    = "Non-matchs AIMS",
            value    = fnum(k["nm_aims"]),
            footer   = f"{pct_nm_aims}% des vols AIMS",
            color    = AMBER,
            bi_icon  = "exclamation-circle-fill",
            badge_html = _badge(0, zero_is_good=True),
            href     = "?nav=tables&drill=nm_aims",
            fdot_color = warn_col,
        ),
        _kpi(
            label    = "Non-matchs MRO",
            value    = fnum(k["nm_mro"]),
            footer   = f"{pct_nm_mro}% des vols MRO",
            color    = VIOLET,
            bi_icon  = "exclamation-circle-fill",
            badge_html = _badge(0, zero_is_good=True),
            href     = "?nav=tables&drill=nm_mro",
            fdot_color = warn_col,
        ),
    ])
    st.markdown(html_vol, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    #  BLOC 3 — Répartition (donuts)
    # ─────────────────────────────────────────────────────────────────────
    st.markdown(_section_hdr("Répartition des données", TEAL), unsafe_allow_html=True)

    leg        = ["Vols matchés", "Non-matchs normaux", "Anomalies détectées"]
    cols_donut = [GREEN, AMBER, CRIMSON]

    d1, d2 = st.columns(2)
    with d1:
        with st.container(border=True):
            st.markdown(
                '<div class="ov-chart-title">Répartition AIMS</div>'
                '<div class="ov-chart-sub">Ventilation par statut de réconciliation</div>',
                unsafe_allow_html=True,
            )
            tot = max(k["total_aims"], 1)
            st.plotly_chart(
                _donut_v2(
                    leg, [k["valides"], k["norm_aims"], k["anom_aims"]],
                    cols_donut, h=300,
                    center=f"{100*k['valides']/tot:.1f}%",
                    sub=f"{fnum(k['valides'])} matchés",
                ),
                use_container_width=True, config={"displayModeBar": False},
            )
    with d2:
        with st.container(border=True):
            st.markdown(
                '<div class="ov-chart-title">Répartition MRO</div>'
                '<div class="ov-chart-sub">Ventilation par statut de réconciliation</div>',
                unsafe_allow_html=True,
            )
            tot = max(k["total_mro"], 1)
            st.plotly_chart(
                _donut_v2(
                    leg, [k["valides"], k["norm_mro"], k["anom_mro"]],
                    cols_donut, h=300,
                    center=f"{100*k['valides']/tot:.1f}%",
                    sub=f"{fnum(k['valides'])} matchés",
                ),
                use_container_width=True, config={"displayModeBar": False},
            )

    # ─────────────────────────────────────────────────────────────────────
    #  BLOC 4 — Tendances temporelles + Top routes
    # ─────────────────────────────────────────────────────────────────────
    st.markdown(_section_hdr("Analyses temporelles", VIOLET), unsafe_allow_html=True)

    ev1, ev2 = st.columns([3, 2])
    with ev1:
        with st.container(border=True):
            st.markdown(
                '<div class="ov-chart-title">Anomalies détectées par mois</div>'
                '<div class="ov-chart-sub">Évolution mensuelle — AIMS vs MRO</div>',
                unsafe_allow_html=True,
            )
            if (not non_traitees.empty
                    and "Date" in non_traitees.columns
                    and non_traitees["Date"].notna().any()):
                g = (
                    non_traitees.dropna(subset=["Date"])
                    .assign(mois=lambda d: d["Date"].dt.to_period("M").astype(str))
                    .groupby(["mois", "Source"]).size()
                    .unstack(fill_value=0).sort_index()
                )
                fig_trend = split_trend(
                    g.index.tolist(),
                    g.get("AIMS", pd.Series(0, index=g.index)).tolist(),
                    g.get("MRO",  pd.Series(0, index=g.index)).tolist(),
                    h=320,
                )
                fig_trend.update_layout(
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                )
                st.plotly_chart(
                    fig_trend,
                    use_container_width=True, config={"displayModeBar": False},
                )
            else:
                st.info("Données de dates insuffisantes")

    with ev2:
        with st.container(border=True):
            st.markdown(
                '<div class="ov-chart-title">Top 5 routes anomaliques</div>'
                '<div class="ov-chart-sub">Liaisons concentrant le plus d\'écarts AIMS↔MRO</div>',
                unsafe_allow_html=True,
            )
            if not non_traitees.empty and "AeroDepart" in non_traitees.columns:
                routes = (
                    non_traitees["AeroDepart"].astype(str)
                    + " → "
                    + non_traitees["AeroArriv"].astype(str)
                )
                top5 = routes.value_counts().head(5)
                if not top5.empty:
                    pal = [CRIMSON, "#e8470f", AMBER, "#d96cc0", VIOLET][: len(top5)]
                    fig_r = go.Figure(go.Bar(
                        x=top5.values.tolist(),
                        y=top5.index.tolist(),
                        orientation="h",
                        marker=dict(color=pal, line=dict(color="rgba(0,0,0,0)", width=0)),
                        text=[f"  {v}" for v in top5.values.tolist()],
                        textposition="outside",
                        textfont=dict(color=INK, size=12,
                                     family="Inter, Segoe UI, sans-serif"),
                        hovertemplate="%{y} : %{x} anomalies<extra></extra>",
                    ))
                    fig_r.update_layout(
                        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                        xaxis=dict(showgrid=True, gridcolor="#f0f2f5"),
                    )
                    _light(fig_r, h=320)
                    fig_r.update_layout(
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#ffffff",
                    )
                    st.plotly_chart(fig_r, use_container_width=True,
                                    config={"displayModeBar": False})
                else:
                    st.info("Aucune route anomalique identifiée")
            else:
                st.info("Données insuffisantes")

# ─────────────────────────────────────────────────────────────────────────────
#  Donut Plotly amélioré
# ─────────────────────────────────────────────────────────────────────────────

def _donut_v2(labels, values, colors, h=280, center=None, sub=None):
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.66, sort=False,
        marker=dict(colors=colors, line=dict(color="#ffffff", width=4)),
        textinfo="percent",
        textfont=dict(color="#fff", size=11,
                      family="Inter, Segoe UI, sans-serif"),
        hovertemplate="%{label}<br><b>%{value}</b> vols (%{percent})<extra></extra>",
    ))
    ann = []
    if center is not None:
        ann.append(dict(
            text=f"<b>{center}</b>", x=.5, y=.57, showarrow=False,
            font=dict(size=26, color="#1f2a33", family="Arial Black, Inter"),
        ))
        ann.append(dict(
            text=sub or "", x=.5, y=.41, showarrow=False,
            font=dict(size=11, color="#7b8794",
                      family="Inter, Segoe UI, sans-serif"),
        ))
    fig.update_layout(
        height=h,
        margin=dict(t=14, b=14, l=14, r=14),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        showlegend=True,
        legend=dict(
            orientation="h", y=-0.04, x=0.5, xanchor="center",
            font=dict(color="#1f2a33", size=11,
                      family="Inter, Segoe UI, sans-serif"),
            itemsizing="constant",
            bgcolor="rgba(0,0,0,0)",
        ),
        annotations=ann,
    )
    return fig