"""
dash_pages/anomalies.py — Liste des anomalies détectées (filtres, export, SHAP).
"""

import streamlit as st
import plotly.graph_objects as go

try:
    import shap
    from shap_explainer import load_model_and_encoder, get_shap_values, interpret_shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

from dashboard_data import inject_css, page_header, load_data, build_anomaly_table

inject_css()
page_header(
    "⚠ Anomalies détectées",
    "Liste complète, filtres et explication du modèle par vol (SHAP)",
)

with st.spinner("Connexion SQL Server et chargement des données..."):
    data = load_data()

df_aims = data["df_aims"]
df_mro  = data["df_mro"]

df_all = build_anomaly_table(df_aims, df_mro)

if df_all.empty:
    st.info("Aucune anomalie trouvée dans la base.")
    st.stop()

# ── Filtres ───────────────────────────────────────────────────────────────────
f1, f2, f3, f4, f5 = st.columns(5)

with f1:
    annees_dispo = sorted(df_all["Date"].str[:4].dropna().unique().tolist())
    annee = st.selectbox("Année", ["Toutes"] + annees_dispo)

with f2:
    mois_map = {
        "Tous":"",      "Janvier":"01",   "Février":"02",
        "Mars":"03",    "Avril":"04",      "Mai":"05",
        "Juin":"06",    "Juillet":"07",    "Août":"08",
        "Septembre":"09","Octobre":"10",   "Novembre":"11",
        "Décembre":"12",
    }
    mois_label = st.selectbox("Mois", list(mois_map.keys()))
    mois = mois_map[mois_label]

with f3:
    source = st.selectbox("Source", ["AIMS + MRO", "AIMS", "MRO"])

with f4:
    statut = st.selectbox("Statut", ["Tous", "Suspect", "Très Suspect"])

with f5:
    mat_filtre = st.text_input("Matricule", placeholder="ex: 7T-VCA")

# ── Application des filtres ─────────────────────────────────────────────────────
df_f = df_all.copy()
if annee != "Toutes":
    df_f = df_f[df_f["Date"].str.startswith(annee)]
if mois:
    df_f = df_f[df_f["Date"].str[5:7] == mois]
if source != "AIMS + MRO":
    df_f = df_f[df_f["Source"] == source]
if statut != "Tous":
    df_f = df_f[df_f["Statut"] == statut]
if mat_filtre:
    df_f = df_f[
        df_f["Matricule"].str.upper().str.contains(
            mat_filtre.upper(), na=False
        )
    ]

st.caption(f"{len(df_f)} anomalie(s) affichée(s) sur {len(df_all)} total")

# ── Table interactive ───────────────────────────────────────────────────────────
st.dataframe(
    df_f.reset_index(drop=True),
    use_container_width=True,
    height=430,
    column_config={
        "Score IF": st.column_config.NumberColumn(
            "Score IF",
            help="Plus bas = plus suspect",
            format="%.4f",
        ),
        "Raison": st.column_config.TextColumn(
            "Raison anomalie",
            width="large",
        ),
        "Statut": st.column_config.TextColumn("Statut"),
    },
)

# ── Export CSV ───────────────────────────────────────────────────────────────────
csv_data = df_f.to_csv(index=False, sep=";", encoding="utf-8-sig")
st.download_button(
    label="⬇ Exporter les anomalies filtrées en CSV",
    data=csv_data,
    file_name="anomalies_filtrees.csv",
    mime="text/csv",
)

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSE SHAP — Explication du modèle par vol
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("Analyse SHAP — Explication du modèle par vol")

if not SHAP_OK:
    st.warning("SHAP non installé. Lancez : `pip install shap`")
elif df_f.empty:
    st.info("Aucune anomalie avec les filtres actuels.")
else:
    @st.cache_resource
    def get_explainer():
        model, encoder = load_model_and_encoder()
        return shap.TreeExplainer(model), encoder

    try:
        explainer, encoder = get_explainer()

        options = [
            f"{r['Source']} | {r['Num Vol']} | {r['Départ']}→{r['Arrivée']} | {r['Date']} | {r['Matricule']}"
            for _, r in df_f.iterrows()
        ]
        choix = st.selectbox(
            "Sélectionner un vol pour voir l'analyse SHAP :",
            options,
            index=0,
        )
        idx   = options.index(choix)
        row   = df_f.iloc[idx].to_dict()
        score = row["Score IF"]

        with st.spinner("Calcul SHAP..."):
            shap_dict = get_shap_values(row, encoder, explainer)

        items  = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        labels = [k for k, _ in items]
        values = [v for _, v in items]
        colors = ["#d62728" if v < 0 else "#2ca02c" for v in values]

        fig_shap = go.Figure(go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in values],
            textposition="outside",
        ))
        fig_shap.update_layout(
            title=dict(
                text=(f"SHAP — Vol {row['Num Vol']} | "
                      f"{row['Départ']}→{row['Arrivée']} | "
                      f"Score IF : {score}"),
                font_size=13,
            ),
            xaxis_title="Contribution (rouge = pousse vers anomalie, vert = pousse vers normal)",
            height=320,
            margin=dict(t=50, b=10, l=130, r=80),
            yaxis=dict(autorange="reversed"),
            shapes=[dict(type="line", x0=0, x1=0, y0=-0.5,
                         y1=len(labels)-0.5,
                         line=dict(color="black", width=1, dash="dot"))],
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        st.info(interpret_shap(shap_dict))

        with st.expander("Comment lire ce graphique ?"):
            st.markdown("""
**SHAP (SHapley Additive exPlanations)** mesure la contribution de chaque feature à la décision du modèle pour CE vol spécifique.

| Couleur | Signification |
|---|---|
| 🔴 Rouge (valeur négative) | Cette feature pousse le modèle vers **anomalie** |
| 🟢 Vert (valeur positive)  | Cette feature pousse le modèle vers **normal** |

Le score final IF = somme de toutes les contributions SHAP + valeur de base.

**Différence avec la RaisonAnomalie :**
- *RaisonAnomalie* = règles métier (le vol est-il connu dans l'historique ?)
- *SHAP* = le modèle lui-même explique mathématiquement sa décision
""")

    except Exception as e:
        st.error(f"Erreur SHAP : {e}")
        st.info("Vérifiez que le modèle est entraîné (`python train.py`) "
                "et que les fichiers models/ existent.")
