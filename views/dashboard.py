import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from ui import (
    load_all_anomalies, global_kpis, _mark_traitee, _unmark_traitee,
    stat, grid, fnum, area_trend, cbar, donut, heatmap_chart, _light,
    _JOURS_HM, _MOIS_HM, _tint,
    GREEN, AMBER, CRIMSON, BLUE, TEAL, VIOLET, ROSE, INK, MUTED, CARD, BORDER,
)

DIMS = ["Mois", "Matricule", "Numéro de vol", "Route",
        "Aéroport départ", "Aéroport arrivée", "Type d'anomalie"]

# ─────────────────────────────────────────────────────────────────────────────
#  Hauteurs fixes — une seule source de vérité pour l'alignement
# ─────────────────────────────────────────────────────────────────────────────
_H_MAIN   = 340   # graphe principal (analyse)  + donut
_H_HEATMAP = 300  # heatmap (grille 7×12)
_H_SOURCE  = 300  # barres source  ← même valeur = alignement parfait

# ─────────────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────────────
_DB_CSS = """
<style>
@import url("https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css");

@keyframes dbRise {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── KPI strip ── */
.db-kpi-strip {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 12px;
    margin: 4px 0 4px;
    animation: dbRise .4s ease-out both;
}
.db-kpi {
    background: #ffffff;
    border: 1px solid #e8ecf0;
    border-radius: 16px;
    padding: 16px 16px 14px;
    overflow: hidden;
    transition: transform .2s ease, box-shadow .2s ease;
}
.db-kpi:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 28px rgba(20,40,25,.09);
}
.db-kpi-icon {
    width: 36px; height: 36px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 12px;
}
.db-kpi-val {
    font-size: 30px;
    font-weight: 900;
    color: #1f2a33;
    line-height: 1;
    letter-spacing: -.025em;
    font-variant-numeric: tabular-nums;
}
.db-kpi-lbl {
    font-size: 12px;
    font-weight: 600;
    color: #7b8794;
    margin-top: 5px;
}
.db-kpi-sub {
    font-size: 10.5px;
    color: #a0aab5;
    margin-top: 3px;
}

/* ── Section header ── */
.db-section {
    display: flex; align-items: center; gap: 10px;
    margin: 26px 0 14px;
}
.db-section-label {
    display: flex; align-items: center; gap: 8px;
    font-size: 12px; font-weight: 700;
    text-transform: uppercase; letter-spacing: .08em;
    color: #1f2a33; white-space: nowrap;
}
.db-section-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.db-section-line { flex: 1; height: 1.5px; border-radius: 2px; opacity: .18; }

/* ── Chart titles ── */
.db-chart-title {
    font-size: 13.5px; font-weight: 700;
    color: #1f2a33; margin-bottom: 2px;
}
.db-chart-sub {
    font-size: 11px; color: #7b8794; margin-bottom: 10px;
}

/* ── Alert banner ── */
.db-alert {
    background: #fff8f0;
    border-left: 4px solid #e8920c;
    padding: 10px 16px;
    border-radius: 0 10px 10px 0;
    margin-top: 8px; margin-bottom: 6px;
    font-size: 12px; color: #7a4a06;
    display: flex; align-items: center; gap: 8px;
}

/* ── Table header ── */
.db-table-header {
    display: flex; align-items: center;
    justify-content: space-between; margin-bottom: 12px;
}
.db-table-title { font-size: 14px; font-weight: 700; color: #1f2a33; }
.db-table-count {
    font-size: 11px; font-weight: 600; color: #7b8794;
    background: #f4f6f9; padding: 3px 10px; border-radius: 20px;
}

/* ── Custom select labels (Bootstrap Icons above selects) ── */
.db-field-label {
    display: flex; align-items: center; gap: 6px;
    font-size: 12px; font-weight: 600; color: #374151;
    margin-bottom: 4px;
}
.db-field-label i { font-size: 13px; }

/* ── Streamlit widget overrides ── */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stDateInput"] input {
    background: #f8fafb !important;
    border-color: #dde2e9 !important;
    border-radius: 10px !important;
    font-size: 13px !important;
}
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px !important;
    border-color: #e6eaef !important;
    box-shadow: 0 2px 12px rgba(20,40,25,.04) !important;
    background: #ffffff !important;
}

/* ── hide default selectbox label (we render our own above) ── */
.db-no-label label[data-testid="stWidgetLabel"] {
    display: none !important;
}
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rgba(hexc: str, a: float) -> str:
    r, g, b = int(hexc[1:3], 16), int(hexc[3:5], 16), int(hexc[5:7], 16)
    return f"rgba({r},{g},{b},{a})"


def _section_hdr(title: str, icon: str, accent: str) -> str:
    return (
        f'<div class="db-section">'
        f'<div class="db-section-label">'
        f'<div class="db-section-dot" style="background:{accent};"></div>'
        f'<i class="bi bi-{icon}" style="color:{accent};font-size:13px;"></i>'
        f'{title}</div>'
        f'<div class="db-section-line" style="background:{accent};"></div>'
        f'</div>'
    )


def _db_kpi(label: str, value: str, sub: str, color: str, bi_icon: str) -> str:
    icon_bg = _rgba(color, .09)
    return (
        f'<div class="db-kpi" style="border-top:3px solid {color};">'
        f'<div class="db-kpi-icon" style="background:{icon_bg};color:{color};">'
        f'<i class="bi bi-{bi_icon}" style="font-size:16px;line-height:1;"></i></div>'
        f'<div class="db-kpi-val">{value}</div>'
        f'<div class="db-kpi-lbl">{label}</div>'
        f'<div class="db-kpi-sub">{sub}</div>'
        f'</div>'
    )


def _kpi_strip(cards: list) -> str:
    return f'<div class="db-kpi-strip">{"".join(cards)}</div>'


def _field_label(icon: str, text: str, color: str = "#6b7280") -> None:
    """Renders a Bootstrap-Icon label above a Streamlit widget."""
    st.markdown(
        f'<div class="db-field-label">'
        f'<i class="bi bi-{icon}" style="color:{color};"></i>'
        f'<span>{text}</span></div>',
        unsafe_allow_html=True,
    )


def _plotly_base(fig, h: int, bg: str = "#ffffff") -> None:
    """Apply consistent base layout to every Plotly figure."""
    fig.update_layout(
        height=h,
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(family="Inter, Segoe UI, sans-serif", size=12),
        margin=dict(t=20, b=20, l=10, r=10),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Breakdown
# ─────────────────────────────────────────────────────────────────────────────

def breakdown(df: pd.DataFrame, dim: str, topn: int = 12):
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

    colmap = {
        "Matricule":         "Matricule",
        "Numéro de vol":     "NumVol",
        "Aéroport départ":   "AeroDepart",
        "Aéroport arrivée":  "AeroArriv",
        "Type d'anomalie":   "TypeAnomalie",
        "Statut":            "Statut",
    }
    if dim == "Route":
        s = df["AeroDepart"].astype(str) + " → " + df["AeroArriv"].astype(str)
    else:
        s = df[colmap[dim]].astype(str)
    g = s.value_counts().head(topn)
    return g.index.tolist(), g.values.tolist(), "cat"


# ─────────────────────────────────────────────────────────────────────────────
#  Page
# ─────────────────────────────────────────────────────────────────────────────

def page_dashboard():
    st.markdown(_DB_CSS, unsafe_allow_html=True)

    anom = load_all_anomalies()

    _drill      = st.session_state.pop("_drill",      None)
    _drill_mval = st.session_state.pop("_drill_mval", None)

    stat_opts  = ["Tous", "Suspect", "Très Suspect", "Écart détecté"]
    trait_opts = ["Toutes", "Non traitées", "Traitées"]
    def_stat   = 0
    def_trait  = 0
    if _drill in ("nt", "crit", "mois"):
        def_trait = 1
    if _drill == "crit":
        def_stat  = 2

    # ── Filtre header ────────────────────────────────────────────────────
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">'
        f'<i class="bi bi-funnel-fill" style="color:{BLUE};font-size:15px;"></i>'
        f'<span style="font-size:13px;font-weight:700;color:#1f2a33;">Filtres</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Barre de filtres avec labels BI ──────────────────────────────────
    with st.container(border=True):
        f1, f2, f3, f4, f5 = st.columns([1, 1.2, 1, 1, 1.3])
        with f1:
            _field_label("airplane-fill", "Source", BLUE)
            source = st.selectbox(" ", ["AIMS + MRO", "AIMS", "MRO"],
                                  label_visibility="collapsed")
        with f2:
            _field_label("bar-chart-fill", "Analyser par", VIOLET)
            dim = st.selectbox(" ", DIMS,
                               help="Change la dimension du graphique principal",
                               label_visibility="collapsed")
        with f3:
            _field_label("search", "Statut", CRIMSON)
            stat_f = st.selectbox(" ", stat_opts, index=def_stat,
                                  label_visibility="collapsed")
        with f4:
            _field_label("check2-circle", "Traitement", GREEN)
            trait_f = st.selectbox(" ", trait_opts, index=def_trait,
                                   label_visibility="collapsed")
        with f5:
            _field_label("calendar3", "Période", AMBER)
            if not anom.empty and anom["Date"].notna().any():
                dmin, dmax = anom["Date"].min().date(), anom["Date"].max().date()
                rng = st.date_input(" ", (dmin, dmax), min_value=dmin, max_value=dmax,
                                    label_visibility="collapsed")
            else:
                rng = None

    # ── Filtrage ─────────────────────────────────────────────────────────
    df = anom.copy()
    if source == "AIMS":
        df = df[df["Source"] == "AIMS"]
    elif source == "MRO":
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

    # ── KPI Strip ────────────────────────────────────────────────────────
    k    = global_kpis()
    tres = int((df["Statut"] == "Très Suspect").sum())
    susp = int((df["Statut"] == "Suspect").sum())
    fn_c = int((df["Statut"] == "Écart détecté").sum())

    st.markdown(_section_hdr("Indicateurs clés", "activity", BLUE), unsafe_allow_html=True)
    st.markdown(
        _kpi_strip([
            _db_kpi("Vols validés",    fnum(k["valides"]), "Rapprochés AIMS↔MRO", GREEN,   "check-circle-fill"),
            _db_kpi("Anomalies",       fnum(len(df)),      "Selon filtres actifs",  ROSE,    "exclamation-triangle-fill"),
            _db_kpi("Très suspects",   fnum(tres),         "Score IF très bas",     CRIMSON, "shield-fill-x"),
            _db_kpi("Suspects",        fnum(susp),         "À vérifier",            AMBER,   "exclamation-circle-fill"),
            _db_kpi("Écarts détectés", fnum(fn_c),         "Divergence AIMS↔MRO",  VIOLET,  "arrow-left-right"),
        ]),
        unsafe_allow_html=True,
    )

    # ════════════════════════════════════════════════════════════════════
    #  ROW 1 — Analyse principale (1.7) + Répartition statut (1)
    # ════════════════════════════════════════════════════════════════════
    st.markdown(_section_hdr("Analyse des anomalies", "bar-chart-fill", VIOLET),
                unsafe_allow_html=True)

    c1, c2 = st.columns([1.7, 1])

    with c1:
        with st.container(border=True):
            title_main = (
                "Top 10 — Numéros de vol les plus anomaliques"
                if dim == "Numéro de vol"
                else f"Anomalies par {dim.lower()}"
            )
            sub_main = (
                "Vols concentrant le plus d'écarts sur la période"
                if dim == "Numéro de vol"
                else "Distribution selon la dimension sélectionnée"
            )
            st.markdown(
                f'<div class="db-chart-title">{title_main}</div>'
                f'<div class="db-chart-sub">{sub_main}</div>',
                unsafe_allow_html=True,
            )
            labels, values, kind = breakdown(
                df, dim, topn=10 if dim == "Numéro de vol" else 12
            )
            if labels:
                fig_main = area_trend(labels, values, GREEN, h=_H_MAIN) \
                    if kind == "time" else cbar(labels, values, h=_H_MAIN)
                _plotly_base(fig_main, _H_MAIN)
                st.plotly_chart(fig_main, use_container_width=True,
                                config={"displayModeBar": False})
                if dim == "Numéro de vol":
                    st.caption(
                        "Les 10 numéros de vol concentrant le plus d'anomalies "
                        "sur la période sélectionnée."
                    )
            else:
                st.info("Aucune donnée pour ce filtre")

    with c2:
        with st.container(border=True):
            st.markdown(
                '<div class="db-chart-title">Répartition par statut</div>'
                '<div class="db-chart-sub">Part de chaque niveau de sévérité</div>',
                unsafe_allow_html=True,
            )
            sc = df["Statut"].value_counts()
            if not sc.empty:
                cmap   = {"Suspect": AMBER, "Très Suspect": CRIMSON,
                          "Écart détecté": VIOLET, "Normal": GREEN}
                colors = [cmap.get(s, BLUE) for s in sc.index]
                fig_d  = donut(sc.index.tolist(), sc.values.tolist(), colors, h=_H_MAIN)
                _plotly_base(fig_d, _H_MAIN)
                st.plotly_chart(fig_d, use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.info("Aucune donnée")

    # ════════════════════════════════════════════════════════════════════
    #  ROW 2 — Heatmap (1) + Source (1)  ← hauteur identique des deux
    # ════════════════════════════════════════════════════════════════════
    st.markdown(_section_hdr("Temporalité & Sources", "clock-history", TEAL),
                unsafe_allow_html=True)

    h1, h2 = st.columns(2)

    with h1:
        with st.container(border=True):
            st.markdown(
                '<div class="db-chart-title">Quand planifier les contrôles qualité ?</div>'
                '<div class="db-chart-sub">Concentration des anomalies par jour × mois</div>',
                unsafe_allow_html=True,
            )
            hm_result = heatmap_chart(df)
            hm_fig    = hm_result[0] if hm_result else None
            hm_pivot  = hm_result[1] if hm_result else None

            if hm_fig:
                # ← hauteur fixe identique à h2
                _plotly_base(hm_fig, _H_HEATMAP)
                hm_fig.update_layout(margin=dict(t=10, b=10, l=40, r=10))
                st.plotly_chart(hm_fig, use_container_width=True,
                                config={"displayModeBar": False, "scrollZoom": False})

                if hm_pivot is not None and hm_pivot.values.max() > 0:
                    mx_val = int(hm_pivot.values.max())
                    mx_pos = hm_pivot.stack().idxmax()
                    jour_mx = _JOURS_HM[mx_pos[0]] if mx_pos[0] < 7 else str(mx_pos[0])
                    mois_mx = _MOIS_HM[mx_pos[1]-1] if 1 <= mx_pos[1] <= 12 else str(mx_pos[1])
                    st.markdown(
                        f'<div style="margin-top:8px;font-size:12px;color:{MUTED};">'
                        f'<i class="bi bi-info-circle"></i> '
                        f'Dans la carte thermique, plus une cellule est rouge, plus le nombre '
                        f'd\'anomalies détectées pour cette période est élevé.'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                fc1, fc2 = st.columns(2)
                with fc1:
                    _field_label("calendar-week", "Jour", TEAL)
                    sel_j = st.selectbox(" ", ["—"] + _JOURS_HM, key="hm_jour",
                                         label_visibility="collapsed")
                with fc2:
                    _field_label("calendar-month", "Mois", TEAL)
                    sel_m = st.selectbox(" ", ["—"] + _MOIS_HM, key="hm_mois",
                                         label_visibility="collapsed")
                if sel_j != "—" and "jour_semaine" in df.columns:
                    df = df[df["jour_semaine"] == _JOURS_HM.index(sel_j)]
                if sel_m != "—" and "mois_num" in df.columns:
                    df = df[df["mois_num"] == _MOIS_HM.index(sel_m) + 1]
            else:
                st.info("Données de dates insuffisantes pour la heatmap")

    with h2:
        with st.container(border=True):
            st.markdown(
                '<div class="db-chart-title">Anomalies par système source</div>'
                '<div class="db-chart-sub">Quel système génère le plus d\'écarts ?</div>',
                unsafe_allow_html=True,
            )
            if "Source" in df.columns and not df.empty:
                sc_src     = df["Source"].value_counts()
                src_total  = max(sc_src.sum(), 1)
                src_labs   = sc_src.index.tolist()
                src_vals   = sc_src.values.tolist()
                src_colors = [BLUE if s == "AIMS" else TEAL for s in src_labs]
                src_text   = [f"{v}  ({100*v/src_total:.0f}%)" for v in src_vals]

                fig_s = go.Figure(go.Bar(
                    x=src_labs, y=src_vals,
                    marker=dict(color=src_colors,
                                line=dict(color="rgba(0,0,0,0)", width=0),
                                opacity=0.88),
                    text=src_text,
                    textposition="outside",
                    textfont=dict(color=INK, size=12,
                                  family="Inter, Segoe UI, sans-serif"),
                    hovertemplate="%{x} : %{y} anomalies<extra></extra>",
                ))
                # ← même hauteur que hm_fig
                _plotly_base(fig_s, _H_SOURCE)
                fig_s.update_layout(
                    bargap=0.40,
                    xaxis=dict(
                        tickfont=dict(size=13, color="#1f2a33"),
                        showgrid=False,
                        showline=False,
                    ),
                    yaxis=dict(
                        gridcolor="#f0f2f5",
                        gridwidth=1,
                        showline=False,
                        # range auto mais on ajoute 20 % de marge haute
                        # pour que les labels "text outside" ne débordent pas
                        range=[0, max(src_vals) * 1.25] if src_vals else [0, 1],
                    ),
                    margin=dict(t=30, b=20, l=10, r=10),
                )
                st.plotly_chart(fig_s, use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.info("Données insuffisantes")

    # ════════════════════════════════════════════════════════════════════
    #  TABLE — Detail des anomalies
    # ════════════════════════════════════════════════════════════════════
    st.markdown(_section_hdr("Detail des anomalies", "table", "#3b6fe0"),
                unsafe_allow_html=True)

    if _drill:
        st.markdown(
            '<script>window.addEventListener("load",()=>'
            'document.getElementById("tbl-anomalies")'
            '?.scrollIntoView({behavior:"smooth"}))</script>',
            unsafe_allow_html=True,
        )
    st.markdown('<div id="tbl-anomalies"></div>', unsafe_allow_html=True)

    with st.container(border=True):
        n_rows = len(df)
        st.markdown(
            f'<div class="db-table-header">'
            f'<div class="db-table-title">Anomalies filtrees</div>'
            f'<div class="db-table-count">'
            f'{fnum(n_rows)} enregistrement{"s" if n_rows != 1 else ""}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        show_cols = ["ID", "Source", "Date", "NumVol", "Matricule",
                     "AeroDepart", "AeroArriv", "Statut", "TypeAnomalie",
                     "ScoreAnomalie", "RaisonAnomalie", "Traitee"]
        avail = [c for c in show_cols if c in df.columns]
        show  = df[avail].copy().sort_values("ScoreAnomalie").reset_index(drop=True)
        if "Traitee" in show.columns:
            show["Traitee"] = show["Traitee"].astype(bool)

        def _on_editor_change():
            st.session_state["_anom_pending_save"] = True

        edited = st.data_editor(
            show,
            column_config={
                "Traitee": st.column_config.CheckboxColumn("Traitee"),
                "ID":      st.column_config.NumberColumn("ID", width="small"),
                "Statut":  st.column_config.TextColumn("Statut"),
            },
            disabled=[c for c in avail if c != "Traitee"],
            use_container_width=True,
            height=400,
            hide_index=True,
            key="anomaly_editor",
            on_change=_on_editor_change,
        )

        # ── Boutons centres — FIX ──────────────────────────────────────
        # Avant : st.columns([1.2, 1.2, 2])  -> pousse tout a gauche
        # Apres : st.columns([1.5, 1.2, 1.2, 1.5]) -> colonne vide symetrique
        #         de chaque cote = centrage parfait
        has_changes = st.session_state.get("_anom_pending_save", False)

        _gap_l, _btn_save, _btn_exp, _gap_r = st.columns([1.5, 1.2, 1.2, 1.5])

        with _btn_save:
            if st.button("Sauvegarder", type="primary",
                         use_container_width=True, disabled=not has_changes):
                if "Traitee" in show.columns and "ID" in show.columns:
                    n = 0
                    for i in range(len(show)):
                        orig = show.loc[i,   "Traitee"]
                        curr = edited.loc[i, "Traitee"]
                        if orig != curr:
                            id_v = int(edited.loc[i, "ID"])
                            src  = str(edited.loc[i, "Source"])
                            _mark_traitee(id_v, src) if curr else _unmark_traitee(id_v, src)
                            n += 1
                    st.session_state["_anom_pending_save"] = False
                    if n:
                        st.cache_data.clear()
                        st.toast(f"{n} anomalie(s) mise(s) a jour.")
                        st.rerun()

        with _gap_r:
            pass  # espace vide droit

        with _btn_exp:
            st.download_button(
                "Exporter CSV",
                show.to_csv(index=False).encode("utf-8"),
                "anomalies_filtrees.csv",
                "text/csv",
                use_container_width=True,
            )