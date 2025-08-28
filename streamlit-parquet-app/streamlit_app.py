# ui_dashboard.py
from __future__ import annotations
import re
from pathlib import Path
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Flight Deal Scanner — Painel", layout="wide")

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "OFERTAS.parquet"

# ======================== ESTILO (CSS + animações) ============================
st.markdown("""
<style>
:root{
  --bg:#0b1220; --card:#0f172a; --muted:#0b1324; --text:#e5e7eb; --sub:#94a3b8;
  --primary:#38bdf8; --primary-glow:#60a5fa; --border:#1f2a44; --ok:#10b981; --bad:#ef4444;
}
html, body, .main { background: var(--bg) !important; color: var(--text); }
.block-container{ padding-top:0 !important; }

.hero {
  position:relative; overflow:hidden; border-radius:16px; border:1px solid var(--border);
  box-shadow: 0 10px 40px rgba(8, 28, 61, .45);
}
.hero::before {
  content:"";
  position:absolute; inset:0;
  background: url('https://images.unsplash.com/photo-1502920917128-1aa500764cbd?q=80&w=1600&auto=format&fit=crop') center/cover no-repeat;
  filter: saturate(1.1) contrast(1.05) brightness(.9);
  transform: scale(1.02);
}
.hero::after {
  content:""; position:absolute; inset:0;
  background: radial-gradient(1200px 400px at 10% 10%, rgba(56,189,248,.30), transparent 60%),
              radial-gradient(1000px 400px at 90% 10%, rgba(99,102,241,.25), transparent 60%),
              linear-gradient(180deg, rgba(2,6,23,.7), rgba(2,6,23,.85));
  animation: glow 10s ease-in-out infinite alternate;
}
@keyframes glow { from{opacity:.85} to{opacity:.95} }

.hero-inner { position:relative; z-index:2; padding:84px 48px; text-align:center; }
.grad {
  background: linear-gradient(90deg, #93c5fd, #38bdf8, #a78bfa, #93c5fd);
  background-size: 200% 200%;
  -webkit-background-clip:text; background-clip:text; color:transparent;
  animation: hue 8s ease-in-out infinite;
}
@keyframes hue { 0%{background-position:0% 50%} 100%{background-position:100% 50%} }

.btn {
  display:inline-flex; align-items:center; gap:10px; padding:14px 20px; border-radius:12px;
  border:1px solid rgba(255,255,255,.12); color:#0b1220; font-weight:700; background:#fff;
  box-shadow: 0 10px 30px rgba(56,189,248,.25);
  transition: transform .2s ease, box-shadow .2s ease;
}
.btn:hover { transform: translateY(-2px); box-shadow: 0 16px 40px rgba(56,189,248,.35); }
.btn-outline {
  background: transparent; color:#fff; border-color: rgba(255,255,255,.25);
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.08);
}
.btn-outline:hover { background: rgba(255,255,255,.06); }

.kpi { background: var(--card); border:1px solid var(--border); border-radius:14px; padding:16px 18px; }
.kpi .label { font-size:12px; color:var(--sub); }
.kpi .value { font-size:28px; font-weight:800; }
.pill { display:inline-flex; align-items:center; gap:6px; padding:4px 8px; border-radius:999px; font-weight:700; font-size:12px; }
.pill.ok { color:#052e1c; background: #bbf7d0; }
.pill.bad{ color:#3d0a0a; background: #fecaca; }

.card { background: var(--card); border:1px solid var(--border); border-radius:14px; }
.card-hover { transition: transform .2s ease, box-shadow .2s ease; }
.card-hover:hover { transform: translateY(-2px); box-shadow: 0 10px 30px rgba(0,0,0,.25); }

.badge { font-size:12px; color:#cbd5e1; background: #0b1324; border:1px solid var(--border); padding:4px 8px; border-radius:999px; }
.tag { display:inline-flex; align-items:center; gap:6px; font-size:12px; color:#cbd5e1; background:#0b1324; border:1px solid var(--border); padding:3px 8px; border-radius:999px; }

.copy-btn { height:26px; width:26px; border-radius:50%; border:1px solid var(--border); display:flex; align-items:center; justify-content:center; background:#0b1324; color:#cbd5e1; }
.copy-btn:hover { background:#111a2c; }

.hstack { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
.grid3 { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:14px; }
.grid2 { display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:14px; }
@media (max-width: 1100px){ .grid3{ grid-template-columns: repeat(2, minmax(0,1fr)); } }
@media (max-width: 700px){ .grid3, .grid2{ grid-template-columns: 1fr; } }
</style>
""", unsafe_allow_html=True)

# =============================== HELPERS ======================================
def fmt_num0_br(x) -> str:
    try:
        return f"{float(x):,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "-"

def fmt_pct2_br(x) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x:.2f}%".replace(".", ",")

def parse_hhmmss(v) -> str | None:
    s = str(v or "").strip()
    m = re.search(r"(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?", s)
    if not m:
        return None
    hh = max(0, min(23, int(m.group(1))))
    mm = max(0, min(59, int(m.group(2))))
    ss = max(0, min(59, int(m.group(3) or 0)))
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

def std_cia(raw: str) -> str:
    s = (str(raw) or "").strip().upper()
    s_simple = "".join(ch for ch in s if ch.isalnum() or ch.isspace())
    if s in {"AD","AZU"} or s.startswith("AZUL") or "AZUL" in s_simple: return "AZUL"
    if s in {"G3"} or s.startswith("GOL") or "GOL" in s_simple: return "GOL"
    if s in {"LA","JJ"} or s.startswith("TAM") or s.startswith("LATAM") or "LATAM" in s_simple or "TAM" in s_simple: return "LATAM"
    if s in {"AZUL","GOL","LATAM"}: return s
    return s

def advp_nearest(x) -> int:
    try: v = float(str(x).replace(",", "."))
    except Exception: v = np.nan
    if np.isnan(v): v = 1
    return min([1,5,11,17,30], key=lambda k: abs(v-k))

@st.cache_data(show_spinner=False)
def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Normalização de colunas esperadas
    rename_map = {
        "IDPESQUISA":"IDPESQUISA","CIA":"CIA","HORA_BUSCA":"HORA_BUSCA","HORA_PARTIDA":"HORA_PARTIDA",
        "HORA_CHEGADA":"HORA_CHEGADA","TIPO_VOO":"TIPO_VOO","DATA_EMBARQUE":"DATA_EMBARQUE",
        "DATAHORA_BUSCA":"DATAHORA_BUSCA","AGENCIA_COMP":"AGENCIA_COMP","PRECO":"PRECO",
        "TRECHO":"TRECHO","ADVP":"ADVP","RANKING":"RANKING"
    }
    for k in list(rename_map):
        if k not in df.columns:  # tenta achar variantes comuns
            if k=="AGENCIA_COMP":
                for alt in ["AGENCIA","AGENCIA_NORM","AGENCIA_COMPRA"]:
                    if alt in df.columns: rename_map[k]=alt; break
            if k=="ADVP":
                for alt in ["ADVP_CANON","ANTECEDENCIA","ANTECEDENCIA_DIAS"]:
                    if alt in df.columns: rename_map[k]=alt; break
            if k=="CIA":
                for alt in ["CIA_NORM","COMPANHIA","CIAEREA"]:
                    if alt in df.columns: rename_map[k]=alt; break
    df = df[list(rename_map.values())].copy()
    df.columns = list(rename_map.keys())

    # Parse datas/horas
    for c in ["DATAHORA_BUSCA"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["HORA_BUSCA","HORA_PARTIDA","HORA_CHEGADA"]:
        df[c] = df[c].apply(parse_hhmmss)

    # PRECO numérico
    df["PRECO"] = (df["PRECO"].astype(str)
        .str.replace(r"[^\d,.-]", "", regex=True)
        .str.replace(",", ".", regex=False))
    df["PRECO"] = pd.to_numeric(df["PRECO"], errors="coerce")

    # Derivadas
    df["CIA_NORM"] = df["CIA"].apply(std_cia)
    df["ADVP_CANON"] = df["ADVP"].apply(advp_nearest)
    df["HORA_HH"] = df["HORA_BUSCA"].str.slice(0,2).fillna("00").astype(int)
    df = df.dropna(subset=["DATAHORA_BUSCA","PRECO"])
    return df

df_raw = load_df(DATA_PATH)

# =============================== HERO =========================================
st.markdown("""
<div class="hero">
  <div class="hero-inner">
    <div class="hstack" style="justify-content:center;margin-bottom:10px;">
      <span class="badge">✈️ Flight Analytics</span>
    </div>
    <h1 style="font-size:58px;font-weight:1000;line-height:1.05;margin:0;">
      Flight Deal <span class="grad">Scanner</span>
    </h1>
    <p style="opacity:.92;font-size:20px;max-width:800px;margin:14px auto 28px;">
      Análise inteligente de ofertas de voo. Compare preços, monitore agências e descubra as melhores oportunidades.
    </p>
    <div class="hstack" style="justify-content:center;">
      <a class="btn" href="#painel">📊 Acessar Painel</a>
      <a class="btn btn-outline" href="#demo">✈️ Ver Demo</a>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div id='painel'></div>", unsafe_allow_html=True)
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# =============================== FILTROS ======================================
min_dt = df_raw["DATAHORA_BUSCA"].min().date()
max_dt = df_raw["DATAHORA_BUSCA"].max().date()

c1,c2,c3,c4,c5 = st.columns([1.2,1,1,1,1])
with c1:
    st.caption("Período")
    dt_ini = st.date_input("Data inicial", value=min_dt, min_value=min_dt, max_value=max_dt, format="DD/MM/YYYY")
    dt_fim = st.date_input("Data final", value=max_dt, min_value=min_dt, max_value=max_dt, format="DD/MM/YYYY")
with c2:
    st.caption("ADVP")
    advp_sel = st.multiselect(" ", options=[1,5,11,17,30], default=[], label_visibility="collapsed")
with c3:
    st.caption("Trecho")
    trechos = sorted([t for t in df_raw["TRECHO"].dropna().unique().tolist() if str(t).strip()!=""])
    trecho_sel = st.multiselect("  ", options=trechos, default=[], label_visibility="collapsed")
with c4:
    st.caption("Hora")
    hh_sel = st.multiselect("   ", options=list(range(24)), default=[], label_visibility="collapsed", format_func=lambda x: f"{x:02d}:00")
with c5:
    st.caption("CIA")
    cia_opts = ["AZUL","GOL","LATAM"]
    cia_sel = st.multiselect("    ", options=cia_opts, default=[], label_visibility="collapsed")

# aplica filtros
mask = (df_raw["DATAHORA_BUSCA"].dt.date >= dt_ini) & (df_raw["DATAHORA_BUSCA"].dt.date <= dt_fim)
if advp_sel: mask &= df_raw["ADVP_CANON"].isin(advp_sel)
if trecho_sel: mask &= df_raw["TRECHO"].isin(trecho_sel)
if hh_sel: mask &= df_raw["HORA_HH"].isin(hh_sel)
if cia_sel: mask &= df_raw["CIA_NORM"].isin(cia_sel)
df = df_raw[mask].copy()

# =============================== MÉTRICAS =====================================
def day_key(d: datetime) -> str: return d.strftime("%Y-%m-%d")

if df.empty:
    st.info("Sem dados para o recorte atual.")
    st.stop()

# totais
total_pesquisas = df["IDPESQUISA"].nunique()
total_ofertas = len(df)
menor_preco = df["PRECO"].min()

# variação vs dia anterior (mantendo demais filtros)
last_dt = df["DATAHORA_BUSCA"].max()
prev_day = last_dt - timedelta(days=1)
def slice_day(base: pd.DataFrame, day: datetime) -> pd.DataFrame:
    return base[base["DATAHORA_BUSCA"].dt.date == day.date()]

cur = slice_day(df, last_dt)
prv = slice_day(df, prev_day)

def dd_pct(curr: float, prev: float) -> tuple[float,bool] | None:
    if prev is None or prev==0 or not np.isfinite(prev): return None
    pct = (curr - prev)/prev*100
    return pct, (pct>=0)

pesq_dd = dd_pct(cur["IDPESQUISA"].nunique() or 0, prv["IDPESQUISA"].nunique() or 0)
of_dd   = dd_pct(len(cur) or 0, len(prv) or 0)
pre_dd  = dd_pct(cur["PRECO"].min() if not cur.empty else np.nan,
                 prv["PRECO"].min() if not prv.empty else np.nan)

# última atualização (data da última linha + hora da coluna C)
last_row = df.loc[df["DATAHORA_BUSCA"].idxmax()]
last_hh = parse_hhmmss(last_row.get("HORA_BUSCA")) or last_row["DATAHORA_BUSCA"].strftime("%H:%M:%S")
last_label = f"{last_row['DATAHORA_BUSCA'].strftime('%d/%m/%Y')} {last_hh}"

# =============================== KPI CARDS ====================================
def pill(delta):
    if delta is None: return "<span class='pill' style='opacity:.65'>—</span>"
    pct, up = delta
    cls = "ok" if up else "bad"
    arrow = "⬆️" if up else "⬇️"
    return f"<span class='pill {cls}'>{arrow} {str(abs(pct)).replace('.', ',')[:5]}%</span>"

k1,k2,k3,k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="kpi card-hover">
      <div class="label">Total de Pesquisas</div>
      <div class="hstack" style="justify-content:space-between;">
        <div class="value">{total_pesquisas:,}</div>
        {pill(pesq_dd)}
      </div>
      <div class="label">Variação vs dia anterior</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="kpi card-hover">
      <div class="label">Total de Ofertas</div>
      <div class="hstack" style="justify-content:space-between;">
        <div class="value">{total_ofertas:,}</div>
        {pill(of_dd)}
      </div>
      <div class="label">Variação vs dia anterior</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="kpi card-hover">
      <div class="label">Menor Preço</div>
      <div class="hstack" style="justify-content:space-between;">
        <div class="value">{fmt_num0_br(menor_preco)}</div>
        {pill(pre_dd)}
      </div>
      <div class="label">Variação vs dia anterior</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="kpi card-hover">
      <div class="label">Última atualização</div>
      <div class="hstack" style="justify-content:flex-start;">
        <div class="value" style="font-size:22px">{last_label}</div>
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ======================= RANKING DE AGÊNCIAS — POR CIA ========================
st.subheader("Ranking de Agências — por CIA")

def ranking_por_cia(df_in: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # para cada (IDPESQUISA, CIA), ordenar agências por menor preço; contar posições
    base = (df_in.groupby(["IDPESQUISA","CIA_NORM","AGENCIA_COMP"], as_index=False)
                 .agg(PRECO_MIN=("PRECO","min")))
    out = {}
    for cia in ["AZUL","GOL","LATAM"]:
        sub = base[base["CIA_NORM"]==cia].copy()
        if sub.empty:
            out[cia] = pd.DataFrame(columns=["AGENCIA","1º%","2º%","3º%"])
            continue
        pos_rows = []
        for _, g in sub.groupby(["IDPESQUISA"]):
            g = g.sort_values("PRECO_MIN", ascending=True).reset_index(drop=True)
            for i in range(min(3, len(g))):
                pos_rows.append({"AGENCIA": g.loc[i,"AGENCIA_COMP"], "pos": i+1})
        pos_df = pd.DataFrame(pos_rows)
        total_ids = sub["IDPESQUISA"].nunique() or 1
        agg = (pos_df.pivot_table(index="AGENCIA", columns="pos", values="pos", aggfunc="count", fill_value=0)
                    .reindex(columns=[1,2,3], fill_value=0)
                    .rename(columns={1:"1º%",2:"2º%",3:"3º%"}))
        agg = (agg/total_ids*100).reset_index()
        out[cia] = agg.sort_values("1º%", ascending=False)
    return out

rank_cia = ranking_por_cia(df)
c1,c2,c3 = st.columns(3)
for cia, col in zip(["AZUL","GOL","LATAM"], [c1,c2,c3]):
    with col:
        st.markdown(f"**{cia}**")
        tbl = rank_cia[cia].copy()
        if tbl.empty:
            st.caption("Sem dados neste recorte.")
        else:
            tbl["1º%"] = tbl["1º%"].map(lambda v: fmt_pct2_br(v).replace("%","") + "%")
            tbl["2º%"] = tbl["2º%"].map(lambda v: fmt_pct2_br(v).replace("%","") + "%")
            tbl["3º%"] = tbl["3º%"].map(lambda v: fmt_pct2_br(v).replace("%","") + "%")
            st.dataframe(tbl.rename(columns={"AGENCIA":"Agência"}), use_container_width=True, hide_index=True)

# ======================== TOP 3 PREÇOS POR TRECHO (ADVP) ======================
st.subheader("Top 3 Preços por Trecho (última pesquisa de cada ADVP)")

# última pesquisa por (TRECHO, ADVP)
dfp = df.copy()
dfp["PAR"] = dfp["TRECHO"].astype(str) + "||" + dfp["ADVP_CANON"].astype(str)
last_per_pair = dfp.groupby("PAR")["DATAHORA_BUSCA"].transform("max")==dfp["DATAHORA_BUSCA"]
df_last = dfp[last_per_pair].copy()

cards = []
for par, sub in df_last.groupby("PAR"):
    trecho, advp = par.split("||")
    # ranking por agência (menor preço) MESMA pesquisa (usa IDPESQUISA mais recente dentro do par)
    id_last = sub.loc[sub["DATAHORA_BUSCA"].idxmax(),"IDPESQUISA"]
    sub_id = sub[sub["IDPESQUISA"]==id_last]
    best_by_ag = (sub_id.groupby("AGENCIA_COMP", as_index=False)["PRECO"].min()
                        .sort_values("PRECO", ascending=True).head(3).reset_index(drop=True))
    data_hora = sub_id["DATAHORA_BUSCA"].max()
    hh = parse_hhmmss(sub_id["HORA_BUSCA"].dropna().iloc[0] if not sub_id["HORA_BUSCA"].dropna().empty else None) \
         or data_hora.strftime("%H:%M:%S")
    label = f"{data_hora.strftime('%d/%m/%Y')} {hh}"

    # % extras
    top1 = best_by_ag.iloc[0]["PRECO"] if len(best_by_ag)>=1 else np.nan
    top2 = best_by_ag.iloc[1]["PRECO"] if len(best_by_ag)>=2 else np.nan
    top3 = best_by_ag.iloc[2]["PRECO"] if len(best_by_ag)>=3 else np.nan
    pct_top2_vs_top1 = ((top2-top1)/top1*100) if np.isfinite(top1) and np.isfinite(top2) else None
    pct_top1_vs_2    = ((top1-top2)/top2*100) if np.isfinite(top1) and np.isfinite(top2) else None
    pct_top1_vs_3    = ((top1-top3)/top3*100) if np.isfinite(top1) and np.isfinite(top3) else None

    # render card
    st.markdown(f"""
    <div class="card card-hover" style="padding:12px 14px; margin-bottom:10px;">
      <div class="hstack" style="justify-content:space-between;">
        <div>
          <div style="font-weight:800;">{trecho} • ADVP {advp}</div>
          <div style="font-size:12px;color:var(--sub)">{label}</div>
        </div>
        <button class="copy-btn" onclick="navigator.clipboard.writeText('{id_last}')"
                title="Copiar ID da pesquisa">?</button>
      </div>
    """, unsafe_allow_html=True)

    for i in range(len(best_by_ag)):
        ag = best_by_ag.iloc[i]["AGENCIA_COMP"]
        pr = best_by_ag.iloc[i]["PRECO"]
        right = ""
        if i==1 and pct_top2_vs_top1 is not None:
            right = f"<div style='font-size:12px;color:var(--sub)'>+{fmt_pct2_br(pct_top2_vs_top1)}</div>"
        if i==0:
            extras = []
            if pct_top1_vs_2 is not None: extras.append(f"{fmt_pct2_br(pct_top1_vs_2)} vs 2º")
            if pct_top1_vs_3 is not None: extras.append(f"{fmt_pct2_br(pct_top1_vs_3)} vs 3º")
            if extras:
                right = f"<div style='font-size:12px;color:var(--sub)'>{' • '.join(extras)}</div>"
        st.markdown(f"""
        <div class="hstack" style="justify-content:space-between;border:1px solid var(--border);
             padding:8px 10px;border-radius:10px;margin-top:8px;">
          <div style="font-weight:700;">{i+1}º — {ag}</div>
          <div style="text-align:right;">
            <div style="font-weight:800;">{fmt_num0_br(pr)}</div>
            {right}
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ======================== TENDÊNCIA POR HORA (4 AGÊNCIAS) =====================
st.subheader("Tendência de Preços por Hora — 4 principais agências")

ag_principais = ["123MILHAS","MAXMILHAS","FLIPMILHAS","CAPOVIAGENS"]
buckets = []
for h in range(24):
    row = {"Hora": f"{h:02d}"}
    subset = df[df["HORA_HH"]==h]
    for ag in ag_principais:
        m = subset.loc[subset["AGENCIA_COMP"]==ag, "PRECO"]
        row[ag] = float(m.min()) if not m.empty else None
    buckets.append(row)
fig_line = go.Figure()
for ag in ag_principais:
    fig_line.add_trace(go.Scatter(x=[b["Hora"] for b in buckets], y=[b[ag] for b in buckets],
                                  mode="lines", name=ag))
fig_line.update_layout(height=340, template="plotly_white",
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig_line, use_container_width=True)

# ====================== COMPETITIVIDADE — PARTICIPAÇÃO CIA ====================
st.subheader("Análise de Competitividade — participação das CIAs nos menores preços")

ids = df["IDPESQUISA"].unique().tolist()
wins = {"AZUL":0,"GOL":0,"LATAM":0}
for idp in ids:
    sub = df[df["IDPESQUISA"]==idp]
    if sub.empty: continue
    best_row = sub.loc[sub["PRECO"].idxmin()]
    cia = best_row["CIA_NORM"]
    if cia in wins: wins[cia]+=1
total = sum(wins.values()) or 1

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(name="AZUL", x=["Share"], y=[wins["AZUL"]/total*100], texttemplate="%{y:.1f}%", textposition="inside"))
fig_bar.add_trace(go.Bar(name="GOL",  x=["Share"], y=[wins["GOL"]/total*100],  texttemplate="%{y:.1f}%", textposition="inside"))
fig_bar.add_trace(go.Bar(name="LATAM",x=["Share"], y=[wins["LATAM"]/total*100],texttemplate="%{y:.1f}%", textposition="inside"))
fig_bar.update_layout(barmode="stack", height=320, template="plotly_white",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      yaxis_title="% de vezes mais barata")
st.plotly_chart(fig_bar, use_container_width=True)
