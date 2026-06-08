"""
app_streamlit.py — Dashboard qualité des données de vols.
Usage : streamlit run app_streamlit.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

try:
    import ollama
    OLLAMA_OK = True
except ImportError:
    OLLAMA_OK = False

try:
    import shap
    from shap_explainer import load_model_and_encoder, get_shap_values, interpret_shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

from db import read_table, read_query
from config import TABLE_VOLS_VALIDES

# ── Config page ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Qualité des données de vols",
    page_icon="✈",
    layout="wide",
)

VERT   = "#2ca02c"
ORANGE = "#ff7f0e"
ROUGE  = "#d62728"
BLEU   = "#1f4e79"


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT (mis en cache — ne recharge pas à chaque interaction)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    aims_total = len(read_table("AIMS"))
    mro_total  = len(read_table("MRO"))
    vv_total   = len(read_table(TABLE_VOLS_VALIDES))

    nm_aims = int(read_query(
        f"SELECT COUNT(*) FROM AIMS WHERE IDAIMS NOT IN "
        f"(SELECT IDAIMS FROM {TABLE_VOLS_VALIDES} WHERE IDAIMS IS NOT NULL)"
    ).iloc[0, 0])

    nm_mro = int(read_query(
        f"SELECT COUNT(*) FROM MRO WHERE IDMRO NOT IN "
        f"(SELECT IDMRO FROM {TABLE_VOLS_VALIDES} WHERE IDMRO IS NOT NULL)"
    ).iloc[0, 0])

    try:
        df_aims = read_table("AnomaliesAIMS")
    except Exception:
        df_aims = pd.DataFrame()

    try:
        df_mro = read_table("AnomaliesMRO")
    except Exception:
        df_mro = pd.DataFrame()

    return dict(
        aims_total=aims_total, mro_total=mro_total, vv_total=vv_total,
        nm_aims=nm_aims, nm_mro=nm_mro,
        df_aims=df_aims, df_mro=df_mro,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PRÉPARATION DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════════

def build_anomaly_table(df_aims, df_mro):
    rows = []
    specs = [
        (df_aims, "AIMS", "IDAIMS",  "NumVolAIMS",  "AeroDepartAIMS",
         "AeroArrivAIMS",  "DateAIMS",   "MatriculeAIMS"),
        (df_mro,  "MRO",  "IDMRO",   "NumVolMRO",   "AeroDepartMRO",
         "AeroArrivMRO",   "DateMRO",    "MatriculeMRO"),
    ]
    for df, src, id_c, nv_c, dep_c, arr_c, dat_c, mat_c in specs:
        if df.empty:
            continue
        for _, r in df.iterrows():
            rows.append({
                "Source":    src,
                "ID":        str(r.get(id_c,  "")),
                "Num Vol":   str(r.get(nv_c,  "")),
                "Départ":    str(r.get(dep_c, "")),
                "Arrivée":   str(r.get(arr_c, "")),
                "Date":      str(r.get(dat_c, ""))[:10],
                "Matricule": str(r.get(mat_c, "")),
                "Statut":    str(r.get("Statut",         "")),
                "Score IF":  round(float(r.get("ScoreAnomalie",  0)), 4),
                "Raison":    str(r.get("RaisonAnomalie", "")),
            })
    return pd.DataFrame(rows)


def build_ranking(df_aims, df_mro):
    mats = {}
    for df, src, mat_c in [(df_aims, "AIMS", "MatriculeAIMS"),
                            (df_mro,  "MRO",  "MatriculeMRO")]:
        if df.empty or mat_c not in df.columns:
            continue
        for _, r in df.iterrows():
            m = str(r[mat_c])
            s = float(r.get("ScoreAnomalie", 0))
            if m not in mats:
                mats[m] = {"Matricule": m, "AIMS": 0, "MRO": 0, "scores": []}
            mats[m][src] += 1
            mats[m]["scores"].append(s)

    rows = [{"Matricule":       m,
             "Anomalies AIMS":  v["AIMS"],
             "Anomalies MRO":   v["MRO"],
             "Total":           v["AIMS"] + v["MRO"],
             "Score moyen IF":  round(sum(v["scores"]) / len(v["scores"]), 4)}
            for m, v in mats.items()]
    df = pd.DataFrame(rows).sort_values("Total", ascending=False).reset_index(drop=True)
    df.index += 1
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES PLOTLY
# ══════════════════════════════════════════════════════════════════════════════

def make_gauge(score, title):
    color = VERT if score >= 95 else (ORANGE if score >= 85 else ROUGE)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%", "font": {"size": 32, "color": color}},
        title={"text": title, "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": color, "thickness": 0.28},
            "steps": [
                {"range": [0,  80],  "color": "#fde8e8"},
                {"range": [80, 95],  "color": "#fef3e2"},
                {"range": [95, 100], "color": "#e8f5e9"},
            ],
            "threshold": {"line": {"color": "red", "width": 3},
                          "thickness": 0.75, "value": 95},
        },
    ))
    fig.update_layout(height=210, margin=dict(t=55, b=5, l=25, r=25))
    return fig


def make_donut(label, total, nm, n_anom):
    fig = go.Figure(go.Pie(
        labels=["Vols matchés", "Non-matchs normaux", "Anomalies IF"],
        values=[total - nm, max(nm - n_anom, 0), n_anom],
        hole=0.55,
        marker_colors=[VERT, ORANGE, ROUGE],
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:,}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Répartition {label}", font_size=13),
        height=300, margin=dict(t=45, b=5, l=5, r=5),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def make_bar_statuts(df_aims, df_mro):
    def vc(df):
        if df.empty or "Statut" not in df.columns:
            return 0, 0
        v = df["Statut"].value_counts()
        return v.get("Suspect", 0), v.get("Très Suspect", 0)
    s_a, ts_a = vc(df_aims)
    s_m, ts_m = vc(df_mro)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="AIMS", x=["Suspect","Très Suspect"],
                         y=[s_a, ts_a], marker_color=BLEU))
    fig.add_trace(go.Bar(name="MRO",  x=["Suspect","Très Suspect"],
                         y=[s_m, ts_m], marker_color=ORANGE))
    fig.update_layout(title="Anomalies par statut", barmode="group",
                      height=300, margin=dict(t=45, b=5, l=5, r=5))
    return fig


def _cat(r):
    if not r or r == "Aucune":
        return None
    r = r.split(" | ")[0]
    if "jamais vu"    in r: return "NumVol inconnu"
    if "AeroDepart"   in r: return "Départ inhabituel"
    if "AeroArriv"    in r: return "Arrivée inhabituelle"
    if "Matricule"    in r: return "Matricule inhabituel"
    if "jamais opéré" in r: return "Jour inhabituel"
    if "faux positif" in r: return "Possible faux positif"
    if "absente"      in r: return "Combinaison inconnue"
    return "Autre"


def make_bar_raisons(df_aims, df_mro):
    cats = [_cat(r) for df in [df_aims, df_mro]
            if not df.empty and "RaisonAnomalie" in df.columns
            for r in df["RaisonAnomalie"]]
    cats = [c for c in cats if c]
    if not cats:
        return None
    vc = pd.Series(cats).value_counts()
    fig = go.Figure(go.Bar(
        x=vc.values, y=vc.index, orientation="h",
        marker_color=ROUGE,
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    fig.update_layout(title="Raisons d'anomalie", height=300,
                      margin=dict(t=45, b=5, l=150, r=5),
                      yaxis=dict(autorange="reversed"))
    return fig


def make_hist_scores(df_aims, df_mro):
    fig = go.Figure()
    for df, nm, col in [(df_aims, "AIMS", BLEU), (df_mro, "MRO", ORANGE)]:
        if df.empty or "ScoreAnomalie" not in df.columns:
            continue
        fig.add_trace(go.Histogram(x=df["ScoreAnomalie"], name=nm,
                                   marker_color=col, opacity=0.7, nbinsx=20))
    fig.update_layout(title="Distribution des scores IF", barmode="overlay",
                      height=300, margin=dict(t=45, b=5, l=5, r=5),
                      xaxis_title="Score (plus bas = plus suspect)",
                      yaxis_title="Nombre de vols")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# APPLICATION PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

def main():

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='color:#1a3a5c;margin-bottom:0'>✈ Qualité des données de vols</h1>"
        "<p style='color:#888;margin-top:4px'>Détection d'anomalies AIMS / MRO "
        "— Isolation Forest + SSIS ETL</p><hr>",
        unsafe_allow_html=True,
    )

    # ── Chargement ────────────────────────────────────────────────────────────
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

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — KPIs
    # ══════════════════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — JAUGES
    # ══════════════════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — GRAPHIQUES
    # ══════════════════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — TABLE ANOMALIES + FILTRES
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Anomalies détectées — Liste complète")

    df_all = build_anomaly_table(df_aims, df_mro)

    if df_all.empty:
        st.info("Aucune anomalie trouvée dans la base.")
        return

    # ── Filtres ───────────────────────────────────────────────────────────────
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

    # ── Application des filtres ───────────────────────────────────────────────
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

    # ── Table interactive ─────────────────────────────────────────────────────
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

    # ── Export CSV ────────────────────────────────────────────────────────────
    csv_data = df_f.to_csv(index=False, sep=";", encoding="utf-8-sig")
    st.download_button(
        label="⬇ Exporter les anomalies filtrées en CSV",
        data=csv_data,
        file_name="anomalies_filtrees.csv",
        mime="text/csv",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4b — ANALYSE SHAP (explication ML par vol)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Analyse SHAP — Explication du modèle par vol")

    if not SHAP_OK:
        st.warning("SHAP non installé. Lancez : `pip install shap`")
    elif df_f.empty:
        st.info("Aucune anomalie avec les filtres actuels.")
    else:
        # Charger modèle + encodeur (mis en cache par Streamlit)
        @st.cache_resource
        def get_explainer():
            model, encoder = load_model_and_encoder()
            return shap.TreeExplainer(model), encoder

        try:
            explainer, encoder = get_explainer()

            # Sélectionner un vol à analyser
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

            # Calcul SHAP
            with st.spinner("Calcul SHAP..."):
                shap_dict = get_shap_values(row, encoder, explainer)

            # Tri par valeur absolue décroissante
            items  = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
            labels = [k for k, _ in items]
            values = [v for _, v in items]
            colors = ["#d62728" if v < 0 else "#2ca02c" for v in values]

            # Graphique SHAP
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

            # Interprétation automatique
            st.info(interpret_shap(shap_dict))

            # Légende pédagogique
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

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4c — VOLS NON-MATCHÉS CLASSÉS NORMAUX PAR IF
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Vols non-matchés classés normaux par IF")
    st.caption(
        "Ces vols n'ont pas trouvé de correspondance AIMS/MRO dans le pipeline ETL, "
        "mais l'Isolation Forest les a jugés statistiquement normaux (predict = 1). "
        "À auditer pour vérifier que l'IF n'a pas manqué des anomalies évidentes."
    )

    @st.cache_data
    def load_normaux():
        try:
            df_na = read_table("VoisNormalesAIMS")
        except Exception:
            df_na = pd.DataFrame()
        try:
            df_nm = read_table("VoisNormalesMRO")
        except Exception:
            df_nm = pd.DataFrame()
        return df_na, df_nm

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

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — CLASSEMENT MATRICULES
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Classement des matricules les plus fréquents en anomalie")

    df_rank = build_ranking(df_aims, df_mro)

    if not df_rank.empty:
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

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — AGENT IA LOCAL (Ollama)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Assistant IA — Posez une question sur les anomalies")

    if not OLLAMA_OK:
        st.warning("Ollama non installé. Lancez : `pip install ollama` puis `ollama pull llama3.2:1b`")
        return

    # Résumé compact des données à passer en contexte
    def build_context(df_all, df_rank):
        n_total   = len(df_all)
        n_aims    = (df_all["Source"] == "AIMS").sum()
        n_mro     = (df_all["Source"] == "MRO").sum()
        n_ts      = (df_all["Statut"] == "Très Suspect").sum()

        top5_mat  = df_rank.head(5)[["Matricule","Total","Score moyen IF"]].to_string(index=False)

        top5_anom = df_all.nsmallest(5, "Score IF")[
            ["Source","Num Vol","Départ","Arrivée","Date","Matricule","Statut","Score IF","Raison"]
        ].to_string(index=False)

        raisons = df_all["Raison"].apply(
            lambda r: r.split(" | ")[0][:60] if r else ""
        ).value_counts().head(5).to_string()

        return f"""Tu es un assistant expert en qualité des données de vols d'une compagnie aérienne algérienne.
Tu analyses les résultats d'un système de détection d'anomalies basé sur SSIS ETL + Isolation Forest.

RÉSUMÉ DES DONNÉES :
- Total anomalies détectées : {n_total} ({n_aims} AIMS, {n_mro} MRO)
- Dont Très Suspects : {n_ts}

TOP 5 MATRICULES EN ANOMALIE :
{top5_mat}

TOP 5 VOLS LES PLUS SUSPECTS (score IF le plus bas) :
{top5_anom}

TOP 5 RAISONS D'ANOMALIE :
{raisons}

Réponds toujours en français, de façon concise et professionnelle.
Si la question dépasse les données disponibles, dis-le clairement."""

    # Initialiser historique de conversation
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Afficher historique
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Exemples de questions
    with st.expander("Exemples de questions", expanded=False):
        ex_cols = st.columns(2)
        exemples = [
            "Quel matricule a le plus d'anomalies ?",
            "Quels sont les 3 vols les plus suspects ?",
            "Quelle est la raison d'anomalie la plus fréquente ?",
            "Y a-t-il plus d'anomalies AIMS ou MRO ?",
            "Que signifie un score IF très négatif ?",
            "Comment interpréter 'possible faux positif' ?",
        ]
        for i, ex in enumerate(exemples):
            with ex_cols[i % 2]:
                if st.button(ex, key=f"ex_{i}"):
                    st.session_state.messages.append({"role": "user", "content": ex})
                    st.rerun()

    # Input utilisateur
    question = st.chat_input("Posez une question sur les anomalies de vols...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyse en cours..."):
                try:
                    context = build_context(df_all, df_rank)
                    messages_ollama = [
                        {"role": "system", "content": context},
                    ] + [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ]

                    response = ollama.chat(
                        model="llama3.2:1b",
                        messages=messages_ollama,
                    )
                    answer = response["message"]["content"]
                except Exception as e:
                    answer = f"Erreur Ollama : {e}\nVérifiez qu'Ollama est lancé (`ollama serve`)."

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    # Bouton effacer conversation
    if st.session_state.messages:
        if st.button("Effacer la conversation"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
