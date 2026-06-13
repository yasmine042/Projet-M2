"""
dash_pages/classement.py — Classement des matricules les plus fréquents en anomalie.
"""

import streamlit as st

from dashboard_data import inject_css, page_header, load_data, build_ranking

inject_css()
page_header(
    "🏆 Classement des matricules",
    "Appareils les plus fréquemment associés à une anomalie",
)

with st.spinner("Connexion SQL Server et chargement des données..."):
    data = load_data()

df_rank = build_ranking(data["df_aims"], data["df_mro"])

if df_rank.empty:
    st.info("Aucune anomalie trouvée dans la base.")
else:
    max_total = int(df_rank["Total"].max())
    st.dataframe(
        df_rank,
        use_container_width=True,
        column_config={
            "Total": st.column_config.ProgressColumn(
                "Total anomalies",
                min_value=0,
                max_value=max_total,
                format="%d",
            ),
            "Score moyen IF": st.column_config.NumberColumn(format="%.4f"),
        },
    )
