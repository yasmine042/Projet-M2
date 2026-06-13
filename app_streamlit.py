"""
app_streamlit.py — Dashboard qualité des données de vols (multi-pages).
Usage : streamlit run app_streamlit.py
"""

import streamlit as st

from dashboard_data import inject_css

st.set_page_config(
    page_title="Qualité des données de vols",
    page_icon="✈",
    layout="wide",
)

inject_css()

with st.sidebar:
    st.markdown("## ✈ FlightAnomaly")
    st.caption("Qualité des données de vols — Air Algérie")
    st.markdown("---")

pg = st.navigation([
    st.Page("dash_pages/overview.py",     title="Vue d'ensemble", icon="🏠", default=True),
    st.Page("dash_pages/anomalies.py",    title="Anomalies",      icon="⚠️"),
    st.Page("dash_pages/vols_normaux.py", title="Vols normaux",   icon="✅"),
    st.Page("dash_pages/classement.py",   title="Classement",     icon="🏆"),
    st.Page("dash_pages/assistant.py",    title="Assistant IA",   icon="🤖"),
])

pg.run()
