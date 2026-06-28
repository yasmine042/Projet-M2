import pandas as pd
import streamlit as st
from ui import (
    load, load_all_anomalies, load_vrais_normaux, _load_non_matches,
    stat, grid, fnum, db_alive,
    BLUE, TEAL, GREEN,
)
try:
    from db import read_query
except Exception:
    read_query = None

_VIRTUAL_TABLES = {
    "Anomalies (toutes)": "Toutes les anomalies IF + RF unifiées",
    "Vols normaux (non-matchés)": "Vols non rapprochés sans anomalie détectée",
    "Non-matchés AIMS": "Vols AIMS sans contrepartie MRO",
    "Non-matchés MRO": "Vols MRO sans contrepartie AIMS",
}

# ── CSS : champs + expander en blanc ──────────────────────────────────────────
_FORM_WHITE_CSS = """
<style>
/* Inputs, selects, number inputs */
div[data-testid="stTextInput"]   input,
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"]   div[data-baseweb="select"] > div:first-child {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
}

/* Fond blanc de l'expander */
div[data-testid="stExpander"] > details {
    background-color: #ffffff !important;
    border: 1px solid #d0d0d0 !important;
    border-radius: 6px !important;
}

/* Contenu intérieur de l'expander */
div[data-testid="stExpander"] details > div[data-testid="stExpanderDetails"] {
    background-color: #ffffff !important;
}

/* Multiselect en blanc */
div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div:first-child {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
}
div[data-baseweb="tag"] {
    background-color: #e8f0fe !important;
}
</style>
"""

def page_tables():
    st.markdown(_FORM_WHITE_CSS, unsafe_allow_html=True)

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