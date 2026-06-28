# ============================================================================
#  app.py — Domestic Airlines · Plateforme de fiabilité des données
#  Point d'entrée : routing entre les 5 pages (views/)
#  Lancement : streamlit run app.py
# ============================================================================

import streamlit as st
from ui import css, appbar, pagehead, _ensure_traitement_table

from views.home import page_home
from views.overview import page_overview
from views.dashboard import page_dashboard
from views.tables import page_tables
from views.detection import page_detection


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
        st.session_state["_drill"] = qp.get("drill")
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
        pagehead("Vue d'ensemble",
                 "Réconciliation AIMS ↔ MRO et indice de fiabilité des données.",
                 '<i class="bi bi-speedometer2"></i>')
        page_overview()
    elif page == "dashboard":
        pagehead("Détection — Statistiques",
                 "Explorez, filtrez et traitez les anomalies de saisie détectées.",
                 '<i class="bi bi-bar-chart-line"></i>')
        page_dashboard()
    elif page == "tables":
        pagehead("Exploration SQL",
                 "Consultation des tables sources et des résultats du pipeline.",
                 '<i class="bi bi-table"></i>')
        page_tables()
    elif page == "detection":
        pagehead("Détection d'anomalie",
                 "Analysez un vol et obtenez un diagnostic immédiat sur ses données.",
                 '<i class="bi bi-cpu"></i>')
        page_detection()


if __name__ == "__main__":
    main()
