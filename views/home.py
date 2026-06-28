import streamlit as st
from ui import _logo_b64

_NAV_ICONS = {
    "overview": ('<svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" '
                 'stroke-linecap="round"><line x1="6" y1="20" x2="6" y2="11"/>'
                 '<line x1="12" y1="20" x2="12" y2="5"/><line x1="18" y1="20" x2="18" y2="14"/></svg>'),
    "dashboard": ('<svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" '
                  'stroke-linecap="round" stroke-linejoin="round">'
                  '<polyline points="3,16 9,10 13,14 21,5"/>'
                  '<circle cx="21" cy="5" r="1.5" fill="#fff" stroke="none"/></svg>'),
    "tables": ('<svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2">'
               '<rect x="3" y="4" width="18" height="16" rx="2"/>'
               '<line x1="3" y1="10" x2="21" y2="10"/><line x1="9" y1="4" x2="9" y2="20"/></svg>'),
    "detection": ('<svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" '
                  'stroke-linecap="round"><circle cx="11" cy="11" r="7"/>'
                  '<line x1="21" y1="21" x2="16.5" y2="16.5"/></svg>'),
}

_HERO_SVG = """
<svg viewBox="0 0 460 300" width="100%" style="max-width:440px;" xmlns="http://www.w3.org/2000/svg">
  <defs><radialGradient id="glow" cx="52%" cy="42%" r="55%">
    <stop offset="0%" stop-color="#ffffff" stop-opacity="0.30"/>
    <stop offset="100%" stop-color="#ffffff" stop-opacity="0"/></radialGradient></defs>
  <circle cx="250" cy="150" r="128" fill="url(#glow)"/>
  <circle cx="250" cy="150" r="120" fill="none" stroke="#fff" stroke-opacity="0.22"/>
  <circle cx="250" cy="150" r="82"  fill="none" stroke="#fff" stroke-opacity="0.15"/>
  <path d="M70,232 Q250,38 432,150" fill="none" stroke="#fff" stroke-opacity="0.6"
        stroke-width="2" stroke-dasharray="2 7" stroke-linecap="round"/>
  <circle cx="70" cy="232" r="6" fill="#fff"/>
  <circle cx="432" cy="150" r="6" fill="#D40C1C"/>
  <text x="52" y="254" fill="#fff" font-size="12" font-family="sans-serif" opacity="0.9">AIMS</text>
  <text x="408" y="136" fill="#fff" font-size="12" font-family="sans-serif" opacity="0.9">MRO</text>
  <g transform="translate(232,58) rotate(78) scale(1.7)">
    <path d="M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z" fill="#fff"/>
  </g>
</svg>
"""

_HOME_CSS = """
<style>
  .stApp {
    background:
      radial-gradient(circle at 84% 6%, rgba(255,255,255,.20), transparent 40%),
      radial-gradient(circle at 4% 96%, rgba(63,115,32,.55), transparent 46%),
      linear-gradient(135deg, #6cb33f 0%, #54942C 46%, #3f7320 100%) !important;
  }
  .block-container { padding-top:2.4rem !important; max-width:1320px; }
  .cover { display:flex; gap:34px; flex-wrap:wrap; align-items:stretch; min-height:84vh; }
  .brand { flex:1 1 500px; color:#fff; display:flex; flex-direction:column; justify-content:center; }
  .logo-chip { background:#fff; border-radius:16px; padding:12px 18px; width:max-content;
               box-shadow:0 10px 26px rgba(0,0,0,.16); }
  .logo-chip img { height:40px; display:block; }
  .brand h1 { font-size:clamp(40px,5.2vw,66px); font-weight:800; margin:24px 0 0;
              letter-spacing:.5px; line-height:1.02; color:#fff; }
  .accent-bar { width:92px; height:5px; background:#D40C1C; border-radius:4px; margin:16px 0 16px; }
  .brand .lead { font-size:18px; opacity:.96; max-width:500px; font-weight:500; }
  .brand .desc { font-size:14px; opacity:.86; max-width:500px; margin-top:14px; line-height:1.65; }
  .brand .foot { margin-top:26px; opacity:.82; font-size:13px;
                 border-top:1px solid rgba(255,255,255,.22); padding-top:16px; width:max-content; }
  .illu { margin-top:8px; }
  .navcol { flex:1 1 370px; display:flex; flex-direction:column; gap:15px; justify-content:center; }
  .navhead { color:#fff; opacity:.85; font-size:12px; letter-spacing:2px;
             text-transform:uppercase; margin-bottom:2px; }
  .navcard, .navcard:link, .navcard:visited, .navcard * {
      color:#fff !important; text-decoration:none !important; }
  .navcard { display:flex; align-items:center; gap:16px;
             background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.22);
             border-radius:18px; padding:17px 20px;
             -webkit-backdrop-filter:blur(6px); backdrop-filter:blur(6px);
             transition:transform .18s ease, background .18s ease, box-shadow .18s ease; }
  .navcard:hover { background:rgba(255,255,255,.22); transform:translateY(-3px);
                   box-shadow:0 14px 30px rgba(0,0,0,.20); }
  .navcard .ico { width:48px; height:48px; flex:0 0 48px; border-radius:14px;
                  background:rgba(255,255,255,.18); display:flex; align-items:center;
                  justify-content:center; }
  .navcard .ico svg { width:24px; height:24px; }
  .navcard .txt h3 { margin:0; font-size:18px; font-weight:700; }
  .navcard .txt p  { margin:3px 0 0; font-size:12.5px; opacity:.9; line-height:1.4; }
  .navcard .arrow  { margin-left:auto; font-size:22px; opacity:.65;
                     transition:opacity .18s, transform .18s; }
  .navcard:hover .arrow { opacity:1; transform:translateX(3px); }
</style>
"""

_HOME_TILES = [
    ("overview",  "Vue globale",
     "État de santé des données aériennes : vols réconciliés, anomalies détectées et alertes opérationnelles."),
    ("dashboard", "Tableau de bord",
     "Investigation et suivi des anomalies : filtrez, analysez et marquez les écarts de saisie traités."),
    ("tables",    "Tables",
     "Consultation des données sources : vols opérations, vols maintenance et résultats de l'analyse."),
    ("detection", "Détection",
     "Analyser un vol spécifique : saisissez ses informations et obtenez un diagnostic immédiat."),
]


def page_home():
    st.markdown(_HOME_CSS, unsafe_allow_html=True)
    b64 = _logo_b64()
    logo_html = (f'<div class="logo-chip"><img src="data:image/png;base64,{b64}"/></div>'
                 if b64 else '<h2 style="color:#fff;margin:0;">DOMESTIC AIRLINES</h2>')
    cards = "".join(
        f'<a class="navcard" href="?nav={key}" target="_self">'
        f'<div class="ico">{_NAV_ICONS[key]}</div>'
        f'<div class="txt"><h3>{title}</h3><p>{desc}</p></div>'
        f'<div class="arrow">→</div></a>' for key, title, desc in _HOME_TILES)
    st.markdown(f"""
        <div class="cover">
          <div class="brand">
            {logo_html}
            <h1>Domestic Airlines</h1>
            <div class="accent-bar"></div>
            <div class="lead">Plateforme de contrôle qualité des données de vol</div>
            <div class="desc">Rapprochement automatique des vols entre les systèmes
              d'opérations et de maintenance, détection intelligente des écarts
              de saisie et suivi des anomalies non traitées.</div>
            <div class="illu">{_HERO_SVG}</div>
          </div>
          <div class="navcol">
            <div class="navhead">Explorer la plateforme</div>
            {cards}
          </div>
        </div>""", unsafe_allow_html=True)
