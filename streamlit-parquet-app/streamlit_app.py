# ui_dashboard.py (Streamlit) — layout animado + visões completas
from __future__ import annotations
import re
from pathlib import Path
from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
import streamlit as st

# ---- Gráficos: Plotly se houver; senão Altair (fallback) ----
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    import altair as alt  # type: ignore
    HAS_PLOTLY = False

# ============================ CONFIG INICIAL ================================
st.set_page_config(page_title="Flight Deal Scanner — Painel", layout="wide")
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "OFERTAS.parquet"

# ======================== ESTILO (CSS + animações) =========================
st.markdown("""
<style>
:root{
  --bg:#0b1220; --card:#0f172a; --muted:#0b1324; --text:#e5e7eb; --sub:#94a3b8;
  --primary:#38bdf8; --primary-glow:#60a5fa; --border:#1f2a44; --ok:#10b981; --bad:#ef4444;
}
html, body, .main { background: var(--bg) !important; color: var(--text); }
.block-container{ padding-top:0 !important; }

/* deixar alertas legíveis no seu tema */
.stAlert, .stAlert p, .stAlert div { color: #111 !important; }

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

# =============================== HELPERS =====================================
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

# ====================== LOAD_DF ROBUSTO (com fallback 0..12) ==================
@st.cache_data(show_spinner=False)
def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Arquivo obrigatório não encontrado: {path.as_posix()}"); st.stop()

    df = pd.read_parquet(path)

    # ---- FALLBACK: colunas 0..12 -> renomeia pelo colmap antigo ----
    try:
        first13 = list(df.columns[:13])
        if all(isinstance(c, (int, np.integer, float)) for c in first13):
            colmap = {
                0: "IDPESQUISA",
                1: "CIA",
                2: "HORA_BUSCA",
                3: "HORA_PARTIDA",
                4: "HORA_CHEGADA",
                5: "TIPO_VOO",
                6: "DATA_EMBARQUE",
                7: "DATAHORA_BUSCA",
                8: "AGENCIA_COMP",
                9: "PRECO",
                10: "TRECHO",
                11: "ADVP",
                12: "RANKING",
            }
            rename = {df.columns[i]: colmap[i] for i in range(min(13, df.shape[1]))}
            df = df.rename(columns=rename)
    except Exception:
        pass
    # ---------------------------------------------------------------

    # Apelidos por coluna canônica
    aliases: dict[str, list[str]] = {
        "IDPESQUISA": ["IDPESQUISA","ID_PESQUISA","ID-BUSCA","IDBUSCA","ID","SEARCH_ID"],
        "CIA": ["CIA","CIA_NORM","CIAEREA","COMPANHIA","CIA_AEREA","AIRLINE"],
        "HORA_BUSCA": ["HORA_BUSCA","HORA DA BUSCA","HORA_COLETA","COLUNA_C","C","HORA"],
        "HORA_PARTIDA": ["HORA_PARTIDA","HORA DE PARTIDA","PARTIDA_HORA"],
        "HORA_CHEGADA": ["HORA_CHEGADA","HORA DE CHEGADA","CHEGADA_HORA"],
        "TIPO_VOO": ["TIPO_VOO","TIPO","CABINE","CLASSE"],
        "DATA_EMBARQUE": ["DATA_EMBARQUE","DATA DE EMBARQUE","EMBARQUE_DATA","DAT_EMB"],
        "DATAHORA_BUSCA": ["DATAHORA_BUSCA","DATA_HORA_BUSCA","TIMESTAMP","DT_BUSCA","DATA_BUSCA","COLETA_DH"],
        "AGENCIA_COMP": ["AGENCIA_COMP","AGENCIA_NORM","AGENCIA","AGENCIA_COMPRA","AGÊNCIA"],
        "PRECO": ["PRECO","PREÇO","PRICE","VALOR","AMOUNT"],
        "TRECHO": ["TRECHO","ROTA","ORIGEM-DESTINO","OD","ORIGEM_DESTINO","ROUTE"],
        "ADVP": ["ADVP","ADVP_CANON","ANTECEDENCIA","ANTECEDENCIA_DIAS","D0_D30"],
        "RANKING": ["RANKING","POSICAO","POSIÇÃO","RANK","PLACE"],
    }

    # Match flexível (case-insensitive)
    col_by_norm = {str(c).strip().lower(): c for c in df.columns}
    selected: dict[str, str] = {}
    missing: list[str] = []

    def pick(cands: list[str]) -> str | None:
        for c in cands:
            norm = c.strip().lower()
            if norm in col_by_norm:
                return col_by_norm[norm]
        return None

    for canon, cands in aliases.items():
        real = pick(cands)
        if real is not None:
            selected[canon] = real
        else:
            missing.append(canon)

    required = ["DATAHORA_BUSCA", "PRECO", "TRECHO"]
    still_missing = [c for c in required if c not in selected]
    if still_missing:
        st.error(
            "Colunas obrigatórias ausentes: "
            + ", ".join(still_missing)
            + ". Veja abaixo as colunas detectadas e ajuste os aliases."
        )
        with st.expander("Colunas detectadas no arquivo"):
            st.write(list(df.columns))
        st.stop()

    # recorta/renomeia
    df2 = df[list(selected.values())].copy()
    df2.columns = list(selected.keys())

    # cria opcionais ausentes
    for opt in missing:
        if opt in required:
            continue
        df2[opt] = np.nan

    # datas/horas
    df2["DATAHORA_BUSCA"] = pd.to_datetime(df2["DATAHORA_BUSCA"], errors="coerce")
    for c in ["HORA_BUSCA","HORA_PARTIDA","HORA_CHEGADA"]:
        if c in df2.columns:
            df2[c] = df2[c].apply(parse_hhmmss)

    # preço
    df2["PRECO"] = (
        df2["PRECO"].astype(str)
        .str.replace(r"[^\d,.-]", "", regex=True)
        .str.replace(",", ".", regex=False)
    )
    df2["PRECO"] = pd.to_numeric(df2["PRECO"], errors="coerce")

    # derivadas
    df2["CIA_NORM"] = df2.get("CIA", pd.Series([None]*len(df2))).apply(std_cia)
    if "ADVP" in df2.columns:
        df2["ADVP_CANON"] = df2["ADVP"].apply(advp_nearest)
    else:
        df2["ADVP_CANON"] = 1

    df2["HORA_HH"] = df2.get("HORA_BUSCA").astype(str).str.slice(0,2)
    df2.loc[df2["HORA_HH"].isin([None,"nan","NaN",""]), "HORA_HH"] = "00"
    df2["HORA_HH"] = pd.to_numeric(df2["HORA_HH"], errors="coerce").fillna(0).astype(int)

    # Se IDPESQUISA não veio, gera um estável por timestamp
    if "IDPESQUISA" not in df2.columns or df2["IDPESQUISA"].isna().all():
        ts = df2["DATAHORA_BUSCA"].astype("int64", errors="ignore")
        df2["IDPESQUISA"] = pd.factorize(ts)[0] + 1

    # limpa
    df2 = df2.dropna(subset=["DATAHORA_BUSCA","PRECO"]).reset_index(drop=True)

    # debug amigável
    with st.expander("Detalhes de mapeamento de colunas", expanded=False):
        ok_map = ", ".join([f"{k} ← {v}" for k,v in selected.items()])
        st.caption(f"Mapeadas: {ok_map}")
        if missing:
            st.caption("Ausentes (criadas vazias): " + ", ".join([m for m in missing if m not in required]))

    return df2

# =============================== CARREGA BASE ================================
df_raw = load_df(DATA_PATH)

# =============================== HERO =======================================
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

# =============================== FILTROS =====================================
min_dt = df_raw["DATAHORA_BUSCA"].min().date()
max_dt = df_raw["DATAHORA_BUSCA"].max().date()

c1,c2,c3,c4,c5 = st.columns([1.4,1,1,1,1])
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

mask = (df_raw["DATAHORA_BUSCA"].dt.date >= dt_ini) & (df_raw["DATAHORA_BUSCA"].dt.date <= dt_fim)
if advp_sel: mask &= df_raw["ADVP_CANON"].isin(advp_sel)
if trecho_sel: mask &= df_raw["TRECHO"].isin(trecho_sel)
if hh_sel: mask &= df_raw["HORA_HH"].isin(hh_sel)
if cia_sel: mask &= df_raw["CIA_NORM"].isin(cia_sel)
df = df_raw[mask].copy()

if df.empty:
    st.info("Sem dados para o recorte atual.")
    st.stop()

# =============================== MÉTRICAS ====================================
def dd_pct(curr: float, prev: float) -> tuple[float,bool] | None:
    if prev is None or prev==0 or not np.isfinite(prev): return None
    pct = (curr - prev)/prev*100
    return pct, (pct>=0)

def slice_day(base: pd.DataFrame, d: datetime) -> pd.DataFrame:
    return base[base["DATAHORA_BUSCA"].dt.date == d.date()]

total_pesquisas = df["IDPESQUISA"].nunique()
total_ofertas   = len(df)
menor_preco     = df["PRECO"].min()

last_dt = df["DATAHORA_BUSCA"].max()
prev_day = last_dt - timedelta(days=1)
cur = slice_day(df, last_dt)
prv = slice_day(df, prev_day)

pesq_dd = dd_pct(cur["IDPESQUISA"].nunique() or 0, prv["IDPESQUISA"].nunique() or 0)
of_dd   = dd_pct(len(cur) or 0, len(prv) or 0)
pre_dd  = dd_pct(cur["PRECO"].min() if not cur.empty else np.nan,
                 prv["PRECO"].min() if not prv.empty else np.nan)

last_row = df.loc[df["DATAHORA_BUSCA"].idxmax()]
last_hh = parse_hhmmss(last_row.get("HORA_BUSCA")) or last_row["DATAHORA_BUSCA"].strftime("%H:%M:%S")
last_label = f"{last_row['DATAHORA_BUSCA'].strftime('%d/%m/%Y')} {last_hh}"

def pill(delta):
    if delta is None: return "<span class='pill' style='opacity:.65'>—</span>"
    pct, up = delta; cls = "ok" if up else "bad"; arrow = "⬆️" if up else "⬇️"
    pct_text = f"{abs(pct):.2f}".replace(".", ",")
    return f"<span class='pill {cls}'>{arrow} {pct_text}%</span>"

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

# =============== RANKING DE AGÊNCIAS — POR CIA (1º/2º/3º %) ==================
st.subheader("Ranking de Agências — por CIA")

def ranking_por_cia(df_in: pd.DataFrame) -> dict[str, pd.DataFrame]:
    base = (df_in.groupby(["IDPESQUISA","CIA_NORM","AGENCIA_COMP"], as_index=False)
                 .agg(PRECO_MIN=("PRECO","min")))
    out = {}
    for cia in ["AZUL","GOL","LATAM"]:
        sub = base[base["CIA_NORM"]==cia].copy()
        if sub.empty:
            out[cia] = pd.DataFrame(columns=["Agência","1º%","2º%","3º%"])
            continue
        pos_rows = []
        for _, g in sub.groupby(["IDPESQUISA"]):
            g = g.sort_values("PRECO_MIN", ascending=True).reset_index(drop=True)
            for i in range(min(3, len(g))):
                pos_rows.append({"Agência": g.loc[i,"AGENCIA_COMP"], "pos": i+1})
        pos_df = pd.DataFrame(pos_rows)
        total_ids = sub["IDPESQUISA"].nunique() or 1
        agg = (pos_df.pivot_table(index="Agência", columns="pos", values="pos", aggfunc="count", fill_value=0)
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
            for c in ["1º%","2º%","3º%"]:
                tbl[c] = tbl[c].map(lambda v: f"{v:.2f}".replace(".", ",") + "%")
            st.dataframe(tbl, use_container_width=True, hide_index=True)

# =================== % GANHO NO MENOR PREÇO POR CIA (EMPILHADO) ==============
st.subheader("% de Ganho no Menor Preço por CIA (empilhado)")

agencias_all = sorted(df["AGENCIA_COMP"].dropna().astype(str).unique().tolist())
stack_rows = []
for cia in ["AZUL","GOL","LATAM"]:
    sub = df[df["CIA_NORM"]==cia]
    ids = sub["IDPESQUISA"].unique().tolist()
    wins_by_ag = {}
    for idp in ids:
        rows = sub[sub["IDPESQUISA"]==idp]
        if rows.empty: continue
        best_row = rows.loc[rows["PRECO"].idxmin()]
        ag = str(best_row["AGENCIA_COMP"])
        wins_by_ag[ag] = wins_by_ag.get(ag, 0) + 1
    total = max(1, sum(wins_by_ag.values()))
    row = {"CIA": cia}
    for ag in agencias_all:
        row[ag] = wins_by_ag.get(ag, 0) / total * 100
    stack_rows.append(row)
stack_df = pd.DataFrame(stack_rows)

if HAS_PLOTLY:
    fig_stack = go.Figure()
    for ag in agencias_all:
        fig_stack.add_trace(go.Bar(x=stack_df["CIA"], y=stack_df[ag], name=ag))
    fig_stack.update_layout(barmode="stack", height=320, template="plotly_white",
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            yaxis_title="% de vezes mais barata")
    st.plotly_chart(fig_stack, use_container_width=True)
else:
    stack_long = stack_df.melt(id_vars="CIA", var_name="Agência", value_name="Share")
    chart = (alt.Chart(stack_long)
             .mark_bar()
             .encode(x="CIA:N", y="Share:Q", color="Agência:N", tooltip=["CIA","Agência","Share"])
             .properties(height=320))
    st.altair_chart(chart, use_container_width=True)

# ============= TOP 3 PREÇOS POR TRECHO — POR ADVP (última pesquisa) ==========
st.subheader("Top 3 Preços por Trecho (última pesquisa de cada ADVP)")

dfp = df.copy()
dfp["PAR"] = dfp["TRECHO"].astype(str) + "||" + dfp["ADVP_CANON"].astype(str)
last_per_pair = dfp.groupby("PAR")["DATAHORA_BUSCA"].transform("max")==dfp["DATAHORA_BUSCA"]
df_last = dfp[last_per_pair].copy()

for par, sub in df_last.groupby("PAR"):
    trecho, advp = par.split("||")
    id_last = sub.loc[sub["DATAHORA_BUSCA"].idxmax(),"IDPESQUISA"]
    sub_id = sub[sub["IDPESQUISA"]==id_last]
    best_by_ag = (sub_id.groupby("AGENCIA_COMP", as_index=False)["PRECO"].min()
                        .sort_values("PRECO", ascending=True).head(3).reset_index(drop=True))
    data_hora = sub_id["DATAHORA_BUSCA"].max()
    hh = parse_hhmmss(sub_id["HORA_BUSCA"].dropna().iloc[0] if not sub_id["HORA_BUSCA"].dropna().empty else None) \
         or data_hora.strftime("%H:%M:%S")
    label = f"{data_hora.strftime('%d/%m/%Y')} {hh}"

    top1 = best_by_ag.iloc[0]["PRECO"] if len(best_by_ag)>=1 else np.nan
    top2 = best_by_ag.iloc[1]["PRECO"] if len(best_by_ag)>=2 else np.nan
    top3 = best_by_ag.iloc[2]["PRECO"] if len(best_by_ag)>=3 else np.nan
    pct_top2_vs_top1 = ((top2-top1)/top1*100) if np.isfinite(top1) and np.isfinite(top2) else None
    pct_top1_vs_2    = ((top1-top2)/top2*100) if np.isfinite(top1) and np.isfinite(top2) else None
    pct_top1_vs_3    = ((top1-top3)/top3*100) if np.isfinite(top1) and np.isfinite(top3) else None

    st.markdown(f"""
    <div class="card card-hover" style="padding:12px 14px; margin-bottom:10px;">
      <div class="hstack" style="justify-content:space-between;">
        <div>
          <div style="font-weight:800;">{trecho} • ADVP {advp}</div>
          <div style="font-size:12px;color:var(--sub)">{label}</div>
        </div>
        <button class="copy-btn" onclick="navigator.clipboard.writeText('{id_last}')" title="Copiar ID da pesquisa">?</button>
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

# =================== TENDÊNCIA POR HORA — 4 PRINCIPAIS AGÊNCIAS ==============
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

if HAS_PLOTLY:
    fig_line = go.Figure()
    for ag in ag_principais:
        fig_line.add_trace(go.Scatter(x=[b["Hora"] for b in buckets], y=[b[ag] for b in buckets],
                                      mode="lines", name=ag))
    fig_line.update_layout(height=340, template="plotly_white",
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_line, use_container_width=True)
else:
    df_hour = pd.DataFrame(buckets).melt(id_vars="Hora", var_name="Agência", value_name="Preço")
    chart = (alt.Chart(df_hour)
             .mark_line()
             .encode(x="Hora:N", y="Preço:Q", color="Agência:N", tooltip=["Hora","Agência","Preço"])
             .properties(height=340))
    st.altair_chart(chart, use_container_width=True)

# =========== COMPETITIVIDADE — PARTICIPAÇÃO DAS CIAS NOS MENORES PREÇOS ======
st.subheader("Análise de Competitividade — participação das CIAs no menor preço")

ids = df["IDPESQUISA"].unique().tolist()
wins = {"AZUL":0,"GOL":0,"LATAM":0}
for idp in ids:
    sub = df[df["IDPESQUISA"]==idp]
    if sub.empty: continue
    best_row = sub.loc[sub["PRECO"].idxmin()]
    cia = str(best_row["CIA_NORM"])
    if cia in wins: wins[cia]+=1
total = sum(wins.values()) or 1
comp_df = pd.DataFrame([
    {"CIA":"AZUL","Share":wins["AZUL"]/total*100},
    {"CIA":"GOL","Share":wins["GOL"]/total*100},
    {"CIA":"LATAM","Share":wins["LATAM"]/total*100},
])

if HAS_PLOTLY:
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="AZUL", x=["Share"], y=[comp_df.loc[0,"Share"]], texttemplate="%{y:.1f}%", textposition="inside"))
    fig_bar.add_trace(go.Bar(name="GOL",  x=["Share"], y=[comp_df.loc[1,"Share"]], texttemplate="%{y:.1f}%", textposition="inside"))
    fig_bar.add_trace(go.Bar(name="LATAM",x=["Share"], y=[comp_df.loc[2,"Share"]], texttemplate="%{y:.1f}%", textposition="inside"))
    fig_bar.update_layout(barmode="stack", height=320, template="plotly_white",
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          yaxis_title="% de vezes mais barata")
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    chart = (alt.Chart(comp_df)
             .mark_bar()
             .encode(x="CIA:N", y=alt.Y("Share:Q", title="% de vezes mais barata"), tooltip=["CIA","Share"])
             .properties(height=320))
    st.altair_chart(chart, use_container_width=True)
