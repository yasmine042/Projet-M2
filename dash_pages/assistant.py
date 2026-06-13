"""
dash_pages/assistant.py — Assistant IA local (Ollama) pour interroger les anomalies.
"""

import streamlit as st

try:
    import ollama
    OLLAMA_OK = True
except ImportError:
    OLLAMA_OK = False

from dashboard_data import inject_css, page_header, load_data, build_anomaly_table, build_ranking

inject_css()
page_header(
    "🤖 Assistant IA",
    "Posez une question sur les anomalies de vols",
)

if not OLLAMA_OK:
    st.warning("Ollama non installé. Lancez : `pip install ollama` puis `ollama pull llama3.2:1b`")
    st.stop()

with st.spinner("Connexion SQL Server et chargement des données..."):
    data = load_data()

df_all  = build_anomaly_table(data["df_aims"], data["df_mro"])
df_rank = build_ranking(data["df_aims"], data["df_mro"])


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
