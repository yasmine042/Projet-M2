import datetime as dt
import pickle

import pandas as pd
import streamlit as st

from ui import (
    load, db_alive, _FLEET, _APTS, fnum, _tint, gauge_chart, pagehead,
    CARD, CRIMSON, GREEN, OLIVE, OLIVE_D, BORDER, INK, MUTED,
)


def _historique_ref():
    if db_alive():
        try:
            from predict import charger_ref_historique
            return charger_ref_historique()
        except Exception:
            pass
    vv = load("VolsValidesEtape1")
    return pd.DataFrame({
        "NumVol": vv["NumVolAIMS"].astype(str), "Matricule": vv["MatriculeAIMS"].astype(str),
        "AeroDepart": vv["AeroDepartAIMS"].astype(str), "AeroArriv": vv["AeroArrivAIMS"].astype(str),
        "Date": pd.to_datetime(vv["DateAIMS"], errors="coerce")}).dropna(subset=["Date"])


def score_flight(date, mat, numvol, dep, arr):
    from config import MODEL_PATH, RF_MODEL_PATH
    from features import FlightFeatureEncoder
    from predict import determiner_type_anomalie, analyser_raison_anomalie

    enc = FlightFeatureEncoder.load()
    with open(MODEL_PATH, "rb") as f:
        iso = pickle.load(f)
    rf = None
    try:
        with open(RF_MODEL_PATH, "rb") as f:
            rf = pickle.load(f)
    except Exception:
        pass

    df_ref = _historique_ref()
    row_df = pd.DataFrame([{
        "Matricule": str(mat).strip().upper(), "NumVol": str(numvol).strip().upper(),
        "AeroDepart": str(dep).strip().upper(), "AeroArriv": str(arr).strip().upper(),
        "Date": pd.to_datetime(date)}])
    X = enc.transform(row_df)
    score = float(iso.score_samples(X)[0])
    if_anom = score < iso.offset_
    row = row_df.iloc[0]

    if not if_anom:
        if_statut, type_anom, raison = "Normal", "Aucune", "Aucune"
    else:
        if_statut = "Suspect"
        type_anom = determiner_type_anomalie(df_ref, row)
        raison = analyser_raison_anomalie(df_ref, row, score, None)

    rf_anom = rf_proba = None
    if not if_anom and rf is not None:
        rf_anom  = bool(rf.predict(X[:, :8])[0] == 1)
        rf_proba = float(rf.predict_proba(X[:, :8])[0, 1])
        if rf_anom:
            type_anom = determiner_type_anomalie(df_ref, row)
            raison = analyser_raison_anomalie(df_ref, row, score, None)

    return {"score": score, "if_anom": if_anom, "type": type_anom,
            "raison": raison, "rf_anom": rf_anom, "rf_proba": rf_proba,
            "if_offset": float(iso.offset_)}, df_ref


# ── Petits composants visuels (alignés sur le design system ui) ──────────────
def _subhead(icon: str, title: str):
    st.markdown(
        f'<div class="section"><div class="sec-ic">'
        f'<i class="bi {icon}" style="font-size:18px;color:{OLIVE};"></i></div>'
        f'<div class="sec-t">{title}</div></div>', unsafe_allow_html=True)


def _form_style():
    """Champs de saisie en blanc et design avancé pour le bouton d'action."""
    st.markdown(f"""<style>
      div[data-baseweb="select"] > div,
      div[data-baseweb="input"],
      .stTextInput input, .stDateInput input, .stNumberInput input {{
          background:{CARD} !important;
          border:1px solid {BORDER} !important;
          border-radius:12px !important;
      }}
      div[data-baseweb="select"] > div:focus-within,
      div[data-baseweb="input"]:focus-within,
      .stTextInput input:focus, .stDateInput input:focus {{
          border-color:{OLIVE} !important;
          box-shadow:0 0 0 3px {_tint(OLIVE,.12)} !important;
      }}
      .stTextInput label, .stDateInput label, .stSelectbox label {{
          font-size:12px !important; font-weight:600 !important;
          color:{MUTED} !important; text-transform:uppercase; letter-spacing:.03em;
      }}
      /* Bouton Analyser le Vol - Alignement et États Interactifs */
      div[data-testid="stButton"] button {{
          background-color: {OLIVE} !important;
          color: white !important;
          border: none !important;
          border-radius: 12px !important;
          height: 42px !important;
          margin-top: 28px !important; /* Calage parfait avec la ligne des inputs */
          font-weight: 600 !important;
          transition: all 0.2s ease-in-out !important;
          box-shadow: 0 4px 12px {_tint(OLIVE,.15)} !important;
      }}
      div[data-testid="stButton"] button:hover {{
          background-color: {OLIVE_D} !important;
          box-shadow: 0 6px 16px {_tint(OLIVE,.3)} !important;
          transform: translateY(-1px);
      }}
      div[data-testid="stButton"] button:active {{
          transform: translateY(1px);
      }}
    </style>""", unsafe_allow_html=True)


def _chip(icon: str, label: str, value: str) -> str:
    return (
        f'<div style="display:flex;align-items:center;gap:10px;background:{CARD};'
        f'border:1px solid {BORDER};border-radius:13px;padding:11px 14px;">'
        f'<div style="width:34px;height:34px;flex:0 0 34px;border-radius:9px;'
        f'background:{_tint(OLIVE,.10)};display:flex;align-items:center;justify-content:center;">'
        f'<i class="bi {icon}" style="color:{OLIVE};font-size:15px;"></i></div>'
        f'<div style="min-width:0;">'
        f'<div style="font-size:10px;color:{MUTED};text-transform:uppercase;'
        f'letter-spacing:.05em;">{label}</div>'
        f'<div style="font-size:14px;font-weight:700;color:{INK};white-space:nowrap;'
        f'overflow:hidden;text-overflow:ellipsis;">{value}</div></div></div>')


def _badge(text: str, color: str) -> str:
    return (f'<span style="background:{_tint(color,.12)};color:{color};padding:6px 14px;'
            f'border-radius:11px;font-size:13px;font-weight:700;white-space:nowrap;">{text}</span>')


def _verdict_card(accent, icon, title, subtitle, badge_html, reason=None) -> str:
    sub = (f'<div style="color:{MUTED};font-size:13px;margin-top:3px;">{subtitle}</div>'
           if subtitle else "")
    rea = (f'<div style="margin-top:16px;padding-top:14px;border-top:1px solid {BORDER};'
           f'color:{INK};font-size:13.5px;line-height:1.6;">{reason}</div>'
           if reason and reason != "Aucune" else "")
    return (
        f'<div style="background:{CARD};border:1px solid {BORDER};border-top:3px solid {accent};'
        f'border-radius:16px;padding:22px 24px;box-shadow:0 4px 18px rgba(20,40,25,.06);">'
        f'<div style="display:flex;align-items:center;gap:16px;">'
        f'<div style="width:56px;height:56px;flex:0 0 56px;border-radius:15px;'
        f'background:{_tint(accent,.12)};display:flex;align-items:center;justify-content:center;">'
        f'<i class="bi {icon}" style="font-size:27px;color:{accent};"></i></div>'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="font-size:21px;font-weight:800;color:{accent};line-height:1.15;">{title}</div>'
        f'{sub}</div><div>{badge_html}</div></div>{rea}</div>')


def _certitude(score: float, offset: float) -> str:
    """Niveau de certitude qualitatif, sans exposer le score brut à l'utilisateur."""
    marge = abs(score - offset)
    if marge >= 0.15:
        return "Certitude élevée"
    if marge >= 0.05:
        return "Certitude modérée"
    return "À confirmer"


def _diagnostic(reason: str, kind: str) -> str:
    """Présente la conclusion en langage clair, sous forme de liste lisible."""
    accent = CRIMSON if kind == "bad" else OLIVE_D
    icon   = "bi-exclamation-circle-fill" if kind == "bad" else "bi-check-circle-fill"
    parts = [p.strip() for p in str(reason).split("|")
             if p.strip() and p.strip().lower() != "aucune"]
    if kind == "bad":
        if not parts:
            parts = ["Ce vol s'écarte des schémas habituels de l'historique réconcilié."]
    else:
        parts = ["Aucune incohérence détectée.",
                 "Le vol correspond aux schémas habituels (route, appareil, calendrier)."]
    lines = "".join(
        f'<div style="display:flex;gap:9px;align-items:flex-start;margin-bottom:9px;">'
        f'<i class="bi {icon}" style="color:{accent};font-size:14px;margin-top:2px;"></i>'
        f'<span style="font-size:13px;color:{INK};line-height:1.5;">{p}</span></div>'
        for p in parts)
    return (
        f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:14px;'
        f'padding:18px 20px;box-shadow:0 2px 12px rgba(20,40,25,.04);height:100%;">'
        f'<div style="font-size:11px;color:{MUTED};text-transform:uppercase;'
        f'letter-spacing:.05em;margin-bottom:13px;">Diagnostic</div>{lines}</div>')


def page_detection():
    _form_style()

    ref = _historique_ref()
    mats = sorted(set(_FLEET) | set(ref["Matricule"].astype(str)))
    apts = sorted(set(_APTS) | set(ref["AeroDepart"].astype(str)) | set(ref["AeroArriv"].astype(str)))

    _subhead("bi-sliders2", "Paramètres du vol")

    # ── Formulaire Haut de Gamme ──
    with st.container(border=True):
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;margin:-4px 0 20px;">'
            f'<div style="background:{_tint(OLIVE,.10)};padding:6px;border-radius:8px;display:flex;align-items:center;">'
            f'<i class="bi bi-airplane-engines" style="color:{OLIVE};font-size:16px;"></i></div>'
            f'<span style="font-size:13.5px;color:{MUTED};font-weight:500;">'
            f'Renseignez les caractéristiques du vol à vérifier, puis lancez l\'analyse.'
            f'</span></div>', unsafe_allow_html=True)

        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            date = st.date_input("Date du vol", dt.date(2019, 6, 15))
        with r1c2:
            numvol = st.text_input("Numéro de vol", "1010")
        with r1c3:
            mat = st.selectbox("Matricule", mats,
                               index=mats.index("7T-VCA") if "7T-VCA" in mats else 0)

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            dep = st.selectbox("Aéroport départ", apts,
                               index=apts.index("ALG") if "ALG" in apts else 0)
        with r2c2:
            arr = st.selectbox("Aéroport arrivée", apts,
                               index=apts.index("HME") if "HME" in apts else 0)
        with r2c3:
            run = st.button("Analyser le vol", type="primary", use_container_width=True)

    if not run:
        return

    try:
        res, df_ref = score_flight(date, mat, numvol, dep, arr)
    except Exception as e:
        st.error(f"Scoring réel indisponible (base ou modèles .pkl introuvables). Détail : {e}")
        st.info("Vérifiez la connexion SQL Server et la présence de "
                "`models/isolation_forest.pkl`, `random_forest.pkl`, `label_encoders.pkl`.")
        return

    if_bad = bool(res["if_anom"])
    rf_bad = bool(res["rf_anom"])
    anomalie = if_bad or rf_bad

    _subhead("bi-clipboard2-data", "Résultat de l'analyse")

    # ── Récapitulatif du vol analysé (Chips Grid) ──
    chips = [
        _chip("bi-calendar3",   "Date",      date.strftime("%d/%m/%Y")),
        _chip("bi-hash",        "N° de vol", str(numvol).strip().upper()),
        _chip("bi-airplane",    "Matricule", str(mat)),
        _chip("bi-signpost-2",  "Route",     f"{dep} → {arr}"),
    ]
    if anomalie and res["type"] and res["type"] != "Aucune":
        chips.append(_chip("bi-exclamation-diamond", "Type d'écart", str(res["type"])))
        
    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));'
        'gap:12px;margin-bottom:24px;">' + "".join(chips) + "</div>",
        unsafe_allow_html=True)

    # ── Verdict ──
    if if_bad:
        verdict = _verdict_card(
            CRIMSON, "bi-shield-exclamation", "Anomalie détectée",
            "Ce vol présente un comportement inhabituel par rapport à l'historique des données.",
            _badge(_certitude(res["score"], res["if_offset"]), CRIMSON))
    elif rf_bad:
        verdict = _verdict_card(
            CRIMSON, "bi-shield-exclamation", "Anomalie détectée",
            "Écart repéré par l'analyse approfondie",
            _badge(f"Probabilité d'écart · {res['rf_proba']:.0%}", CRIMSON))
    else:
        verdict = _verdict_card(
            GREEN, "bi-shield-check", "Vol normal",
            "Ce vol correspond aux schémas habituels de l'historique",
            _badge(_certitude(res["score"], res["if_offset"]), OLIVE_D))

    # ── Layout & Alignement Parfait des Sorties ──
    c1, c2 = st.columns([1.1, 0.9], gap="large", vertical_alignment="top")

    with c1:
        st.markdown(verdict, unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size: 12px; color: {MUTED}; margin-top: 14px; padding-left: 4px; line-height:1.4;">'
            f'<i class="bi bi-info-circle" style="margin-right:6px;"></i>'
            f'L\'analyse vérifie la cohérence de saisie des données du vol, '
            f'non le déroulement opérationnel du vol.'
            f'</div>', 
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(_diagnostic(res["raison"], "bad" if anomalie else "ok"),
                    unsafe_allow_html=True)