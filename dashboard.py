"""
dashboard.py — Tableau de bord qualité des données de vols.
Page unique scrollable : KPIs, jauges, graphiques, table filtrée, classement matricules.
Usage : python dashboard.py
"""

import os, json, webbrowser
import pandas as pd
import plotly.graph_objects as go

from db import read_table, read_query
from config import TABLE_VOLS_VALIDES

BLEU   = "#1f4e79"
VERT   = "#2ca02c"
ORANGE = "#ff7f0e"
ROUGE  = "#d62728"


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT
# ══════════════════════════════════════════════════════════════════════════════

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
# DONNÉES JSON POUR JAVASCRIPT
# ══════════════════════════════════════════════════════════════════════════════

def to_table_rows(df_aims, df_mro):
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
                "Source":     src,
                "ID":         str(r.get(id_c,  "")),
                "NumVol":     str(r.get(nv_c,  "")),
                "AeroDepart": str(r.get(dep_c, "")),
                "AeroArriv":  str(r.get(arr_c, "")),
                "Date":       str(r.get(dat_c, ""))[:10],
                "Matricule":  str(r.get(mat_c, "")),
                "Statut":     str(r.get("Statut",         "")),
                "Score":      round(float(r.get("ScoreAnomalie",  0)), 4),
                "Raison":     str(r.get("RaisonAnomalie", "")),
            })
    return rows


def to_ranking(df_aims, df_mro):
    mats = {}
    for df, src, mat_c in [(df_aims, "AIMS", "MatriculeAIMS"),
                            (df_mro,  "MRO",  "MatriculeMRO")]:
        if df.empty or mat_c not in df.columns:
            continue
        for _, r in df.iterrows():
            m = str(r[mat_c])
            s = float(r.get("ScoreAnomalie", 0))
            if m not in mats:
                mats[m] = {"Matricule": m, "AIMS": 0, "MRO": 0, "sc": []}
            mats[m][src] += 1
            mats[m]["sc"].append(s)

    out = [{"Matricule": m,
            "NbAIMS":    v["AIMS"],
            "NbMRO":     v["MRO"],
            "Total":     v["AIMS"] + v["MRO"],
            "ScoreMoyen": round(sum(v["sc"]) / len(v["sc"]), 4)}
           for m, v in mats.items()]
    out.sort(key=lambda x: x["Total"], reverse=True)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES PLOTLY
# ══════════════════════════════════════════════════════════════════════════════

def gauge(score, title):
    color = VERT if score >= 95 else (ORANGE if score >= 85 else ROUGE)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%", "font": {"size": 32, "color": color}},
        title={"text": title, "font": {"size": 12, "color": "#555"}},
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
    fig.update_layout(height=200, margin=dict(t=50, b=5, l=20, r=20),
                      paper_bgcolor="white")
    return fig


def donut(label, total, nm, n_anom):
    fig = go.Figure(go.Pie(
        labels=["Vols matchés", "Non-matchs normaux", "Anomalies IF"],
        values=[total - nm, max(nm - n_anom, 0), n_anom],
        hole=0.58,
        marker_colors=[VERT, ORANGE, ROUGE],
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:,}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Répartition {label}", font_size=13),
        height=280, margin=dict(t=45, b=5, l=5, r=5),
        legend=dict(orientation="h", y=-0.18),
        paper_bgcolor="white",
    )
    return fig


def bar_statuts(df_aims, df_mro):
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
    fig.update_layout(title=dict(text="Anomalies par statut", font_size=13),
                      barmode="group", height=280,
                      margin=dict(t=45, b=5, l=5, r=5),
                      paper_bgcolor="white")
    return fig


def _cat(r):
    if not r or r == "Aucune": return None
    r = r.split(" | ")[0]
    if "jamais vu"    in r: return "NumVol inconnu"
    if "AeroDepart"   in r: return "Départ inhabituel"
    if "AeroArriv"    in r: return "Arrivée inhabituelle"
    if "Matricule"    in r: return "Matricule inhabituel"
    if "jamais opéré" in r: return "Jour inhabituel"
    if "faux positif" in r: return "Possible faux positif"
    if "absente"      in r: return "Combinaison inconnue"
    return "Autre"


def bar_raisons(df_aims, df_mro):
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
    fig.update_layout(title=dict(text="Raisons d'anomalie", font_size=13),
                      height=280, margin=dict(t=45, b=5, l=145, r=5),
                      yaxis=dict(autorange="reversed"),
                      paper_bgcolor="white")
    return fig


def bar_scores(df_aims, df_mro):
    fig = go.Figure()
    for df, nm, col in [(df_aims, "AIMS", BLEU), (df_mro, "MRO", ORANGE)]:
        if df.empty or "ScoreAnomalie" not in df.columns:
            continue
        fig.add_trace(go.Histogram(x=df["ScoreAnomalie"], name=nm,
                                   marker_color=col, opacity=0.7, nbinsx=20))
    fig.update_layout(title=dict(text="Distribution scores IF", font_size=13),
                      barmode="overlay", height=280,
                      margin=dict(t=45, b=5, l=5, r=5),
                      xaxis_title="Score (bas = suspect)",
                      paper_bgcolor="white")
    return fig


def to_div(fig, first=False):
    return fig.to_html(full_html=False,
                       include_plotlyjs="cdn" if first else False,
                       config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUCTION HTML
# ══════════════════════════════════════════════════════════════════════════════

def build_html(data):
    aims_total = data["aims_total"]
    mro_total  = data["mro_total"]
    vv_total   = data["vv_total"]
    nm_aims    = data["nm_aims"]
    nm_mro     = data["nm_mro"]
    df_aims    = data["df_aims"]
    df_mro     = data["df_mro"]

    n_a = len(df_aims)
    n_m = len(df_mro)
    sc_global = round(vv_total / aims_total * 100, 1)
    sc_aims   = round((aims_total - nm_aims) / aims_total * 100, 1)
    sc_mro    = round((mro_total  - nm_mro)  / mro_total  * 100, 1)

    rows    = to_table_rows(df_aims, df_mro)
    ranking = to_ranking(df_aims, df_mro)

    rows_js    = json.dumps(rows,    ensure_ascii=False, default=str)
    ranking_js = json.dumps(ranking, ensure_ascii=False, default=str)

    # Figures
    figs = [
        gauge(sc_global, "Score fiabilité global"),
        gauge(sc_aims,   "Score matching AIMS"),
        gauge(sc_mro,    "Score matching MRO"),
        donut("AIMS", aims_total, nm_aims, n_a),
        donut("MRO",  mro_total,  nm_mro,  n_m),
        bar_statuts(df_aims, df_mro),
        bar_raisons(df_aims, df_mro),
        bar_scores(df_aims, df_mro),
    ]
    figs = [f for f in figs if f is not None]

    # Premier fig charge plotlyjs via CDN, les suivants non
    divs = [to_div(figs[0], first=True)] + [to_div(f) for f in figs[1:]]
    g1, g2, g3 = divs[0], divs[1], divs[2]
    chart_divs = divs[3:]

    def chart_row(*items):
        cells = "".join(f'<div class="cbox">{d}</div>' for d in items)
        return f'<div class="crow">{cells}</div>'

    charts_html = ""
    it = iter(chart_divs)
    for c1 in it:
        charts_html += chart_row(c1, next(it, ""))

    def kpi(val, lbl, color=""):
        return (f'<div class="kpi {color}">'
                f'<div class="kv">{val}</div>'
                f'<div class="kl">{lbl}</div></div>')

    kpis = (
        kpi(f"{aims_total:,}", "Vols AIMS total",        "vert")  +
        kpi(f"{mro_total:,}",  "Vols MRO total",         "vert")  +
        kpi(f"{vv_total:,}",   "Vols valides (matchés)", "vert")  +
        kpi(f"{nm_aims:,}",    "Non-matchs AIMS",        "ora")   +
        kpi(f"{nm_mro:,}",     "Non-matchs MRO",         "ora")   +
        kpi(f"{n_a + n_m}",    "Anomalies IF détectées", "rouge")
    )

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Tableau de bord — Qualité vols</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',sans-serif;background:#eef2f7;color:#222}}

/* header */
.hdr{{background:linear-gradient(135deg,#1a3a5c,#2980b9);color:#fff;padding:16px 28px}}
.hdr h1{{font-size:18px;font-weight:700}}
.hdr p{{font-size:12px;opacity:.75;margin-top:3px}}

/* sections */
.sec{{padding:14px 24px}}
.sec-title{{font-size:13px;font-weight:700;color:#1a3a5c;
            border-left:4px solid #2980b9;padding-left:10px;margin-bottom:12px}}

/* KPI */
.krow{{display:flex;gap:10px;flex-wrap:wrap}}
.kpi{{background:#fff;border-radius:8px;padding:13px 16px;flex:1;min-width:110px;
      box-shadow:0 2px 5px rgba(0,0,0,.07);border-top:4px solid #2980b9}}
.kpi.vert {{border-top-color:#2ca02c}}.kpi.ora{{border-top-color:#e67e22}}
.kpi.rouge{{border-top-color:#c0392b}}
.kv{{font-size:22px;font-weight:800;color:#1a3a5c}}
.kpi.vert  .kv{{color:#2ca02c}}.kpi.ora .kv{{color:#e67e22}}
.kpi.rouge .kv{{color:#c0392b}}
.kl{{font-size:10px;color:#888;margin-top:3px}}

/* gauge row */
.grow{{display:flex;gap:10px;flex-wrap:wrap}}
.gbox{{background:#fff;border-radius:8px;flex:1;min-width:180px;
       box-shadow:0 2px 5px rgba(0,0,0,.07);overflow:hidden}}

/* chart grid */
.crow{{display:flex;gap:10px;flex-wrap:wrap;margin-top:10px}}
.cbox{{background:#fff;border-radius:8px;flex:1;min-width:300px;
       box-shadow:0 2px 5px rgba(0,0,0,.07);overflow:hidden}}

/* filter bar */
.fbar{{background:#1a3a5c;border-radius:8px;padding:12px 16px;margin-bottom:12px;
       display:flex;flex-wrap:wrap;gap:10px;align-items:center}}
.fbar label{{color:#aac;font-size:11px;font-weight:600}}
.fbar select,.fbar input{{border:1px solid #3a5a7c;border-radius:5px;
  padding:5px 8px;font-size:12px;color:#fff;background:#243f5e}}
.fbar select:focus,.fbar input:focus{{outline:none;border-color:#3498db}}
.btn-exp{{background:#2ecc71;color:#fff;border:none;border-radius:5px;
          padding:6px 14px;font-size:12px;cursor:pointer;font-weight:700;
          margin-left:auto}}
.btn-exp:hover{{background:#27ae60}}
#cnt{{color:#aac;font-size:11px}}

/* anomaly table */
.tbl-wrap{{background:#fff;border-radius:8px;box-shadow:0 2px 5px rgba(0,0,0,.07);
           overflow:auto;max-height:480px}}
table{{border-collapse:collapse;width:100%;font-size:12px}}
th{{background:#1a3a5c;color:#fff;padding:9px 10px;text-align:left;
    white-space:nowrap;position:sticky;top:0;z-index:2}}
td{{padding:7px 10px;border-bottom:1px solid #f0f0f0;vertical-align:top}}
tr:hover td{{background:#f5f9ff}}
.badge{{display:inline-block;border-radius:4px;padding:2px 6px;
        font-size:10px;font-weight:700}}
.ba{{background:#d0e8ff;color:#1a3a5c}}.bm{{background:#ffe8cc;color:#7c3d00}}
.ss{{display:inline-block;border-radius:4px;padding:2px 6px;
     font-size:10px;font-weight:700;white-space:nowrap}}
.s1{{background:#fff3cd;color:#856404}}.s2{{background:#fde8e8;color:#721c24}}
.sc{{font-family:monospace;font-size:11px;color:#666}}
.rc{{max-width:260px;color:#444;line-height:1.4}}
.bv{{background:#3498db;color:#fff;border:none;border-radius:4px;
     padding:3px 8px;font-size:11px;cursor:pointer}}
.bv:hover{{background:#2980b9}}
.empty td{{text-align:center;color:#bbb;padding:20px;font-style:italic}}

/* ranking table */
.rtbl th{{background:#2c3e50}}
.rtbl td{{text-align:center}}
.rtbl td:nth-child(2){{text-align:left;font-weight:600}}

/* modal */
#modal{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.45);
        z-index:9999;align-items:center;justify-content:center}}
.mbox{{background:#fff;border-radius:10px;padding:24px 26px;
       width:540px;max-width:94vw;box-shadow:0 8px 32px rgba(0,0,0,.25)}}
.mbox h3{{font-size:14px;color:#1a3a5c;margin-bottom:14px}}
.dtbl{{width:100%;border-collapse:collapse;font-size:12px}}
.dtbl th{{width:120px;text-align:left;color:#555;font-weight:600;
          padding:5px 8px 5px 0;vertical-align:top}}
.dtbl td{{padding:5px 0;color:#222;border-bottom:1px solid #f0f0f0}}
.bcl{{float:right;background:none;border:none;font-size:20px;cursor:pointer;color:#aaa}}
.bcl:hover{{color:#333}}
</style>
</head>
<body>

<!-- Header -->
<div class="hdr">
  <h1>Tableau de bord — Qualité des données de vols</h1>
  <p>Détection d'anomalies AIMS / MRO &nbsp;|&nbsp; Isolation Forest + SSIS ETL</p>
</div>

<!-- ── 1. KPIs ─────────────────────────────────────────────────────────────── -->
<div class="sec">
  <div class="sec-title">Indicateurs globaux</div>
  <div class="krow">{kpis}</div>
</div>

<!-- ── 2. Jauges ──────────────────────────────────────────────────────────── -->
<div class="sec">
  <div class="sec-title">Score de fiabilité</div>
  <div class="grow">
    <div class="gbox">{g1}</div>
    <div class="gbox">{g2}</div>
    <div class="gbox">{g3}</div>
  </div>
</div>

<!-- ── 3. Graphiques (collapsibles) ───────────────────────────────────────── -->
<div class="sec">
  <details open>
    <summary style="cursor:pointer;font-size:13px;font-weight:700;
                    color:#1a3a5c;border-left:4px solid #2980b9;
                    padding-left:10px;margin-bottom:12px;list-style:none">
      ▼ Graphiques d'analyse
    </summary>
    {charts_html}
  </details>
</div>

<!-- ── 4. TABLE DES ANOMALIES ─────────────────────────────────────────────── -->
<div class="sec">
  <div class="sec-title">Anomalies détectées — Liste complète</div>

  <!-- Filtres -->
  <div class="fbar">
    <label>Année</label>
    <select id="f-an"><option value="">Toutes</option></select>

    <label>Mois</label>
    <select id="f-mo">
      <option value="">Tous</option>
      <option value="01">Janvier</option><option value="02">Février</option>
      <option value="03">Mars</option><option value="04">Avril</option>
      <option value="05">Mai</option><option value="06">Juin</option>
      <option value="07">Juillet</option><option value="08">Août</option>
      <option value="09">Septembre</option><option value="10">Octobre</option>
      <option value="11">Novembre</option><option value="12">Décembre</option>
    </select>

    <label>Source</label>
    <select id="f-src">
      <option value="">AIMS + MRO</option>
      <option value="AIMS">AIMS</option>
      <option value="MRO">MRO</option>
    </select>

    <label>Statut</label>
    <select id="f-st">
      <option value="">Tous</option>
      <option value="Suspect">Suspect</option>
      <option value="Très Suspect">Très Suspect</option>
    </select>

    <label>Matricule</label>
    <input id="f-mat" type="text" placeholder="ex: 7T-VCA" style="width:90px">

    <button class="btn-exp" onclick="exportCSV()">&#11015; Exporter CSV</button>
    <span id="cnt"></span>
  </div>

  <!-- Table -->
  <div class="tbl-wrap">
    <table>
      <thead>
        <tr>
          <th>Source</th>
          <th>ID</th>
          <th>Num Vol</th>
          <th>Trajet</th>
          <th>Date</th>
          <th>Matricule</th>
          <th>Statut</th>
          <th>Score IF</th>
          <th>Raison anomalie</th>
          <th>Détail</th>
        </tr>
      </thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>
</div>

<!-- ── 5. CLASSEMENT MATRICULES ───────────────────────────────────────────── -->
<div class="sec">
  <div class="sec-title">Classement des matricules les plus fréquents en anomalie</div>
  <div class="tbl-wrap">
    <table class="rtbl">
      <thead>
        <tr>
          <th>#</th>
          <th>Matricule</th>
          <th>Anomalies AIMS</th>
          <th>Anomalies MRO</th>
          <th>Total</th>
          <th>Score IF moyen</th>
        </tr>
      </thead>
      <tbody id="rtbody"></tbody>
    </table>
  </div>
</div>

<!-- Modal détail -->
<div id="modal">
  <div class="mbox">
    <button class="bcl" onclick="closeModal()">&#x2715;</button>
    <h3>Détail du vol suspect</h3>
    <div id="mbody"></div>
  </div>
</div>

<script>
var ROWS    = {rows_js};
var RANKING = {ranking_js};
var _cur    = [];

// ── Peupler le filtre année ──────────────────────────────────────────────────
(function(){{
  var yrs = [];
  ROWS.forEach(function(r){{
    var y = (r.Date||'').substring(0,4);
    if(y && yrs.indexOf(y)<0) yrs.push(y);
  }});
  yrs.sort();
  var sel = document.getElementById('f-an');
  yrs.forEach(function(y){{
    var o = document.createElement('option');
    o.value = y; o.textContent = y;
    sel.appendChild(o);
  }});
}})();

// ── Filtrage ─────────────────────────────────────────────────────────────────
function getFiltered(){{
  var an  = document.getElementById('f-an').value;
  var mo  = document.getElementById('f-mo').value;
  var src = document.getElementById('f-src').value;
  var st  = document.getElementById('f-st').value;
  var mat = document.getElementById('f-mat').value.toUpperCase().trim();
  return ROWS.filter(function(r){{
    if(an  && !(r.Date||'').startsWith(an))              return false;
    if(mo  && (r.Date||'').substring(5,7)!==mo)          return false;
    if(src && r.Source!==src)                            return false;
    if(st  && r.Statut!==st)                             return false;
    if(mat && r.Matricule.toUpperCase().indexOf(mat)<0)  return false;
    return true;
  }});
}}

// ── Rendu table ──────────────────────────────────────────────────────────────
function renderTable(){{
  _cur = getFiltered();
  var tb = document.getElementById('tbody');
  tb.innerHTML = '';
  document.getElementById('cnt').textContent =
    _cur.length + ' / ' + ROWS.length + ' anomalie(s)';

  if(_cur.length===0){{
    tb.innerHTML='<tr class="empty"><td colspan="10">Aucune anomalie pour ces filtres.</td></tr>';
    return;
  }}

  _cur.forEach(function(r,i){{
    var scls = r.Statut==='Très Suspect'?'s2':'s1';
    var bcls = r.Source==='AIMS'?'ba':'bm';
    var rai  = r.Raison||'';
    var short= rai.length>65 ? rai.substring(0,65)+'…' : rai;
    var tr   = document.createElement('tr');
    tr.innerHTML =
      '<td><span class="badge '+bcls+'">'+r.Source+'</span></td>'+
      '<td style="color:#999;font-size:11px">'+r.ID+'</td>'+
      '<td><strong>'+r.NumVol+'</strong></td>'+
      '<td>'+r.AeroDepart+' &rarr; '+r.AeroArriv+'</td>'+
      '<td style="white-space:nowrap">'+r.Date+'</td>'+
      '<td><code>'+r.Matricule+'</code></td>'+
      '<td><span class="ss '+scls+'">'+r.Statut+'</span></td>'+
      '<td class="sc">'+r.Score+'</td>'+
      '<td class="rc" title="'+rai.replace(/"/g,"&quot;")+'">'+short+'</td>'+
      '<td><button class="bv" onclick="showDetail('+i+')">Voir</button></td>';
    tb.appendChild(tr);
  }});
}}

// ── Modal ────────────────────────────────────────────────────────────────────
function showDetail(i){{
  var r = _cur[i];
  var scls = r.Statut==='Très Suspect'?'s2':'s1';
  var bcls = r.Source==='AIMS'?'ba':'bm';
  document.getElementById('mbody').innerHTML =
    '<table class="dtbl">'+
    '<tr><th>Source</th><td><span class="badge '+bcls+'">'+r.Source+'</span></td></tr>'+
    '<tr><th>ID</th><td>'+r.ID+'</td></tr>'+
    '<tr><th>Num&eacute;ro vol</th><td><strong>'+r.NumVol+'</strong></td></tr>'+
    '<tr><th>Trajet</th><td>'+r.AeroDepart+' &rarr; '+r.AeroArriv+'</td></tr>'+
    '<tr><th>Date</th><td>'+r.Date+'</td></tr>'+
    '<tr><th>Matricule</th><td><code>'+r.Matricule+'</code></td></tr>'+
    '<tr><th>Statut</th><td><span class="ss '+scls+'">'+r.Statut+'</span></td></tr>'+
    '<tr><th>Score IF</th><td class="sc"><strong>'+r.Score+'</strong> &nbsp;<em style="color:#999;font-size:11px">(plus bas = plus suspect)</em></td></tr>'+
    '<tr><th>Raison</th><td style="line-height:1.6;color:#333;white-space:pre-wrap">'+r.Raison+'</td></tr>'+
    '</table>';
  document.getElementById('modal').style.display='flex';
}}
function closeModal(){{document.getElementById('modal').style.display='none';}}
document.getElementById('modal').addEventListener('click',function(e){{if(e.target===this)closeModal();}});

// ── Export CSV ───────────────────────────────────────────────────────────────
function exportCSV(){{
  var cols=['Source','ID','NumVol','AeroDepart','AeroArriv','Date','Matricule','Statut','Score','Raison'];
  var lines=[cols.join(';')];
  _cur.forEach(function(r){{
    lines.push([r.Source,r.ID,r.NumVol,r.AeroDepart,r.AeroArriv,
                r.Date,r.Matricule,r.Statut,r.Score,
                '"'+String(r.Raison).replace(/"/g,'""')+'"'].join(';'));
  }});
  var blob=new Blob(['﻿'+lines.join('\r\n')],{{type:'text/csv;charset=utf-8;'}});
  var url=URL.createObjectURL(blob);
  var a=document.createElement('a');
  a.href=url; a.download='anomalies_filtrees.csv'; a.click();
  URL.revokeObjectURL(url);
}}

// ── Classement matricules ────────────────────────────────────────────────────
(function(){{
  var tb=document.getElementById('rtbody');
  RANKING.forEach(function(r,i){{
    var tr=document.createElement('tr');
    tr.innerHTML=
      '<td style="color:#999">'+(i+1)+'</td>'+
      '<td><strong>'+r.Matricule+'</strong></td>'+
      '<td>'+r.NbAIMS+'</td>'+
      '<td>'+r.NbMRO+'</td>'+
      '<td><strong style="color:#c0392b">'+r.Total+'</strong></td>'+
      '<td class="sc">'+r.ScoreMoyen+'</td>';
    tb.appendChild(tr);
  }});
}})();

// ── Bind filtres + init ───────────────────────────────────────────────────────
['f-an','f-mo','f-src','f-st'].forEach(function(id){{
  document.getElementById(id).addEventListener('change',renderTable);
}});
document.getElementById('f-mat').addEventListener('input',renderTable);
renderTable();
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    print("Chargement des données SQL Server...")
    data = load_data()
    print("Génération du dashboard...")
    html = build_html(data)
    path = os.path.abspath("dashboard.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard généré : {path}")
    webbrowser.open(f"file:///{path}")
