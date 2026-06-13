"""
dash_pages/overview.py — Vue d'ensemble : KPIs, jauges de fiabilité, graphiques.
"""

import streamlit as st

from dashboard_data import (
    inject_css, page_header, load_data,
    make_gauge, make_donut, make_bar_statuts, make_bar_raisons, make_hist_scores,
)

inject_css()
page_header(
    "✈ Vue d'ensemble",
    "Détection d'anomalies AIMS / MRO — Isolation Forest + SSIS ETL",
)

with st.spinner("Connexion SQL Server et chargement des données..."):
    data = load_data()

aims_total = data["aims_total"]
mro_total  = data["mro_total"]
vv_total   = data["vv_total"]
nm_aims    = data["nm_aims"]
nm_mro     = data["nm_mro"]
df_aims    = data["df_aims"]
df_mro     = data["df_mro"]
n_a        = len(df_aims)
n_m        = len(df_mro)
sc_global  = round(vv_total   / aims_total * 100, 1)
sc_aims    = round((aims_total - nm_aims)  / aims_total * 100, 1)
sc_mro     = round((mro_total  - nm_mro)   / mro_total  * 100, 1)

# ══════════════════════════════════════════════════════════════════════════════
# KPIs
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Indicateurs globaux")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Vols AIMS",           f"{aims_total:,}")
c2.metric("Vols MRO",            f"{mro_total:,}")
c3.metric("Vols matchés",        f"{vv_total:,}")
c4.metric("Non-matchs AIMS",     f"{nm_aims:,}",
          delta=f"-{nm_aims}", delta_color="inverse")
c5.metric("Non-matchs MRO",      f"{nm_mro:,}",
          delta=f"-{nm_mro}",  delta_color="inverse")
c6.metric("Anomalies détectées", f"{n_a + n_m}",
          delta=f"-{n_a+n_m}", delta_color="inverse")

# ══════════════════════════════════════════════════════════════════════════════
# JAUGES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("Score de fiabilité")
g1, g2, g3 = st.columns(3)
with g1:
    st.plotly_chart(make_gauge(sc_global, "Score fiabilité global (AIMS)"),
                    use_container_width=True)
with g2:
    st.plotly_chart(make_gauge(sc_aims, "Score matching AIMS"),
                    use_container_width=True)
with g3:
    st.plotly_chart(make_gauge(sc_mro,  "Score matching MRO"),
                    use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# GRAPHIQUES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
with st.expander("Graphiques d'analyse", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(make_donut("AIMS", aims_total, nm_aims, n_a),
                        use_container_width=True)
    with c2:
        st.plotly_chart(make_donut("MRO",  mro_total,  nm_mro,  n_m),
                        use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(make_bar_statuts(df_aims, df_mro),
                        use_container_width=True)
    with c4:
        fig_r = make_bar_raisons(df_aims, df_mro)
        if fig_r:
            st.plotly_chart(fig_r, use_container_width=True)

    st.plotly_chart(make_hist_scores(df_aims, df_mro),
                    use_container_width=True)
