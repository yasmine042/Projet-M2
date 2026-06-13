"""
dash_pages/vols_normaux.py — Vols non-matchés classés normaux par l'Isolation Forest.
"""

import streamlit as st

from dashboard_data import inject_css, page_header, load_normaux

inject_css()
page_header(
    "✅ Vols non-matchés classés normaux",
    "Audit des vols sans correspondance AIMS/MRO jugés statistiquement normaux par l'IF",
)

st.caption(
    "Ces vols n'ont pas trouvé de correspondance AIMS/MRO dans le pipeline ETL, "
    "mais l'Isolation Forest les a jugés statistiquement normaux (predict = 1). "
    "À auditer pour vérifier que l'IF n'a pas manqué des anomalies évidentes."
)

with st.spinner("Chargement des vols normaux..."):
    df_norm_aims, df_norm_mro = load_normaux()

if df_norm_aims.empty and df_norm_mro.empty:
    st.warning(
        "Tables VoisNormalesAIMS / VoisNormalesMRO introuvables. "
        "Relancez `python predict.py` pour les générer."
    )
else:
    kn1, kn2 = st.columns(2)
    kn1.metric("Non-matchés normaux AIMS", len(df_norm_aims))
    kn2.metric("Non-matchés normaux MRO",  len(df_norm_mro))

    tab_na, tab_nm = st.tabs(["AIMS normaux", "MRO normaux"])

    with tab_na:
        if df_norm_aims.empty:
            st.info("Aucun vol AIMS classé normal.")
        else:
            st.dataframe(df_norm_aims, use_container_width=True, height=380)
            csv_na = df_norm_aims.to_csv(index=False, sep=";", encoding="utf-8-sig")
            st.download_button(
                "⬇ Exporter AIMS normaux CSV",
                data=csv_na,
                file_name="vols_normaux_aims.csv",
                mime="text/csv",
            )

    with tab_nm:
        if df_norm_mro.empty:
            st.info("Aucun vol MRO classé normal.")
        else:
            st.dataframe(df_norm_mro, use_container_width=True, height=380)
            csv_nm = df_norm_mro.to_csv(index=False, sep=";", encoding="utf-8-sig")
            st.download_button(
                "⬇ Exporter MRO normaux CSV",
                data=csv_nm,
                file_name="vols_normaux_mro.csv",
                mime="text/csv",
            )
