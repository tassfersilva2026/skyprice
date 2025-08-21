# streamlit_app.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime, date
import re
import math
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Skyscanner — Painel", layout="wide", initial_sidebar_state="expanded")

# ===========================
# Config / caminhos fixos
# ===========================
APP_DIR   = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "OFERTAS.parquet"

# ===========================
# Helpers
# ===========================
def _first_existing_image(repo_root: Path) -> Path | None:
    # tenta achar 1ª imagem na pasta principal do repo (irmã do app)
    candidates = list(repo_root.glob("*.png")) + list(repo_root.glob("*.jpg")) + list(repo_root.glob("*.jpeg")) + list(repo_root.glob("*.gif"))
    return candidates[0] if candidates else None

def std_agencia(raw: str) -> str:
    ag = (raw or "").strip().upper()
    if ag == "BOOKINGCOM":            return "BOOKING.COM"
    if ag == "KIWICOM":               return "KIWI.COM"
    if ag.startswith("123MILHAS") or ag == "123":   return "123MILHAS"
    if ag.startswith("MAXMILHAS") or ag == "MAX":   return "MAXMILHAS"
    if ag.startswith("CAPOVIAGENS"):  return "CAPOVIAGENS"
    if ag.startswith("FLIPMILHAS"):   return "FLIPMILHAS"
    if ag.startswith("VAIDEPROMO"):   return "VAIDEPROMO"
    if ag.startswith("KISSANDFLY"):   return "KISSANDFLY"
    if ag.startswith("ZUPPER"):       return "ZUPPER"
    if ag.startswith("MYTRIP"):       return "MYTRIP"
    if ag.startswith("GOTOGATE"):     return "GOTOGATE"
    if ag.startswith("DECOLAR"):      return "DECOLAR"
    if ag.startswith("EXPEDIA"):      return "EXPEDIA"
    if ag.startswith("GOL"):          return "GOL"
    if ag.startswith("LATAM"):        return "LATAM"
    if ag.startswith("TRIPCOM"):      return "TRIP.COM"
    if ag.startswith("VIAJANET"):     return "VIAJANET"
    if ag in ("", "NAN", "NONE", "NULL", "SKYSCANNER"): return "SEM OFERTAS"
    return ag

def advp_nearest(x: float | int | str) -> int:
    if pd.isna(x): return 1
    try: v = float(str(x).replace(",", "."))
    except: return 1
    options = [1,5,11,17,30]
    return min(options, key=lambda k: abs(v-k))

@st.cache_data(show_spinner=True)
def load_base(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Arquivo obrigatório não encontrado: `{path.as_posix()}`"); st.stop()
    df = pd.read_parquet(path)

    # Renomeia por segurança se já vierem corretas (mantém), senão ajusta
    # Posições esperadas:
    # A: IDPESQUISA | B: CIA | C: HORA_BUSCA | D: HORA_PARTIDA | E: HORA_CHEGADA
    # F: TIPO_VOO | G: DATA_EMBARQUE | H: DATAHORA_BUSCA | I: AGENCIA_COMP
    # J: PRECO | K: TRECHO | L: ADVP | M: RANKING
    colmap = {
        0:"IDPESQUISA", 1:"CIA", 2:"HORA_BUSCA", 3:"HORA_PARTIDA", 4:"HORA_CHEGADA",
        5:"TIPO_VOO", 6:"DATA_EMBARQUE", 7:"DATAHORA_BUSCA", 8:"AGENCIA_COMP",
        9:"PRECO", 10:"TRECHO", 11:"ADVP", 12:"RANKING"
    }
    # se colunas vierem sem nomes ou diferentes, tenta mapear por posição
    if list(df.columns[:13]) != list(colmap.values()):
        rename = {df.columns[i]: colmap[i] for i in range(min(13, df.shape[1]))}
        df = df.rename(columns=rename)

    # Tipagens/conversões
    # Horas texto -> datetime.time + HH
    for c in ["HORA_BUSCA", "HORA_PARTIDA", "HORA_CHEGADA"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c].astype(str).str.strip(), errors="coerce").dt.time
    df["HORA_HH"] = pd.to_datetime(df["HORA_BUSCA"].astype(str), errors="coerce").dt.hour

    # Datas
    for c in ["DATA_EMBARQUE", "DATAHORA_BUSCA"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True).dt.date

    # Preço
    if "PRECO" in df.columns:
        df["PRECO"] = pd.to_numeric(df["PRECO"].astype(str).str.replace(r"[^\d,.-]","", regex=True).str.replace(",",".", regex=False), errors="coerce")

    # Ranking
    if "RANKING" in df.columns:
        df["RANKING"] = pd.to_numeric(df["RANKING"], errors="coerce").astype("Int64")

    # Normalização de agência/cia ofertante
    df["AGENCIA_NORM"] = df["AGENCIA_COMP"].apply(std_agencia)
    df["AGENCIA_GRUPO"] = df["AGENCIA_NORM"].replace({"MAXMILHAS":"GRUPO 123","123MILHAS":"GRUPO 123"})

    # ADVP canônico
    df["ADVP_CANON"] = df["ADVP"].apply(advp_nearest)

    return df

def winners_by_position(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna uma tabela por IDPESQUISA com colunas R1, R2, R3 (agência do ranking 1..3).
    Se não houver 2º/3º, preenche 'SEM OFERTAS'."""
    base = pd.DataFrame({"IDPESQUISA": df["IDPESQUISA"].unique()})
    for r in (1,2,3):
        s = (df[df["RANKING"]==r]
             .sort_values(["IDPESQUISA"]) # 1 por pesquisa
             .drop_duplicates(subset=["IDPESQUISA"]))
        base = base.merge(s[["IDPESQUISA","AGENCIA_NORM"]].rename(columns={"AGENCIA_NORM":f"R{r}"}),
                          on="IDPESQUISA", how="left")
    for r in (1,2,3):
        base[f"R{r}"] = base[f"R{r}"].fillna("SEM OFERTAS")
    return base

def top_share(series: pd.Series, topn=20) -> pd.DataFrame:
    ct = series.value_counts(dropna=False).rename("Qtde").to_frame()
    total = int(ct["Qtde"].sum()) or 1
    ct["%"] = (ct["Qtde"] / total * 100).round(2)
    ct = ct.reset_index().rename(columns={"index":"Agência/Cia"})
    return ct.head(topn)

def make_bar(df: pd.DataFrame, x: str, y: str, sort_y_desc=True):
    if sort_y_desc:
        df = df.sort_values(y, ascending=False)
    return alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{x}:Q", title=x),
        y=alt.Y(f"{y}:N", sort="-x", title=y),
        tooltip=list(df.columns)
    ).properties(height=360, use_container_width=True)

def make_line(df: pd.DataFrame, x: str, y: str, color: str|None=None):
    enc = dict(
        x=alt.X(f"{x}:T", title=x),
        y=alt.Y(f"{y}:Q", title=y),
        tooltip=list(df.columns)
    )
    if color:
        enc["color"] = alt.Color(f"{color}:N", title=color)
    return alt.Chart(df).mark_line(point=True).encode(**enc).properties(height=360, use_container_width=True)

# ===========================
# Carregar base
# ===========================
df_raw = load_base(DATA_PATH)

# ===========================
# Banner (pasta principal do repo)
# ===========================
repo_root = APP_DIR  # o app está na raiz do repo; se estiver em subpasta, ajuste aqui
img = _first_existing_image(repo_root)
if img:
    st.image(img.as_posix(), use_container_width=True)

# ===========================
# Abas
# ===========================
tab_labels = [
    "1. Painel",
    "2. Top 3 Agências",
    "3. Top 3 Preços Mais Baratos",
    "4. Ranking por Agências",
    "5. Preço por Período do Dia",
    "6. Qtde de Buscas x Ofertas",
    "7. Comportamento Cias",
    "8. Competitividade",
    "9. Melhor Preço Diário",
    "10. Dados"
]
tabs = st.tabs(tab_labels)

# ===========================
# Filtros (horizontais, valem pra todas as abas)
# DATA usa H = DATAHORA_BUSCA
# HORA usa C (HORA_BUSCA -> HH)
# ADVP usa L (agrupado em 1,5,11,17,30)
# TRECHO usa K
# ===========================
with st.container():
    c1, c2, c3, c4, c5 = st.columns([1.2,1.2,1,2,1.2])

    # Data min/max a partir de H
    data_min = pd.to_datetime(pd.Series(df_raw["DATAHORA_BUSCA"])).min()
    data_max = pd.to_datetime(pd.Series(df_raw["DATAHORA_BUSCA"])).max()
    def d(x): 
        try: return x.date()
        except: return None
    data_min = d(data_min) or date(2000,1,1)
    data_max = d(data_max) or date.today()

    with c1:
        dt_ini = st.date_input("Data inicial (col. H)", value=data_min, min_value=data_min, max_value=data_max)
    with c2:
        dt_fim = st.date_input("Data final (col. H)", value=data_max, min_value=data_min, max_value=data_max)
    with c3:
        advp_opts = [1,5,11,17,30]
        advp_sel = st.multiselect("ADVP (col. L)", advp_opts, default=advp_opts)
    with c4:
        trechos = sorted([t for t in df_raw["TRECHO"].dropna().unique().tolist() if str(t).strip()!=""])
        tr_sel = st.multiselect("Trechos (col. K)", trechos, default=[])
    with c5:
        horas = list(range(0,24))
        hh_sel = st.multiselect("Hora da busca HH (col. C)", horas, default=[])

# aplica filtros
df = df_raw.copy()
mask = pd.Series(True, index=df.index)

# Data (H)
if dt_ini: mask &= (pd.to_datetime(df["DATAHORA_BUSCA"], errors="coerce") >= pd.Timestamp(dt_ini))
if dt_fim: mask &= (pd.to_datetime(df["DATAHORA_BUSCA"], errors="coerce") <= pd.Timestamp(dt_fim))

# ADVP
if advp_sel:
    mask &= df["ADVP_CANON"].isin(advp_sel)

# TRECHO
if tr_sel:
    mask &= df["TRECHO"].isin(tr_sel)

# HORA (HH)
if hh_sel:
    mask &= df["HORA_HH"].isin(hh_sel)

df = df[mask].copy()

# Métricas rápidas no topo
m1, m2, m3, m4 = st.columns(4)
m1.metric("Linhas filtradas", f"{len(df):,}".replace(",", "."))
m2.metric("Pesquisas únicas", f"{df['IDPESQUISA'].nunique():,}".replace(",", "."))
m3.metric("Ofertas (linhas) por pesquisa", f"{(len(df)/(df['IDPESQUISA'].nunique() or 1)):.2f}".replace(".", ","))
m4.metric("Trechos distintos", f"{df['TRECHO'].nunique():,}".replace(",", "."))

# ======================================================================
# 1) PAINEL
# ======================================================================
with tabs[0]:
    st.subheader("Painel")
    if df.empty:
        st.info("Sem dados para os filtros."); st.stop()

    st.caption("• Contagem única de pesquisas (col. A) • Participação por posição de ranking (col. M)")

    # Pesquisas únicas
    st.metric("Pesquisas únicas", f"{df['IDPESQUISA'].nunique():,}".replace(",", "."))

    # Vencedores por posição (R1, R2, R3) com 'SEM OFERTAS' quando inexistente
    W = winners_by_position(df)

    # Separado (agência normal)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Participação TOP1 (agências separadas)**")
        t1 = top_share(W["R1"])
        st.dataframe(t1, use_container_width=True, height=280)
    with col_b:
        st.markdown("**Participação TOP1 (visão GRUPO 123)**")
        Wg = W.copy()
        Wg["R1G"] = Wg["R1"].replace({"MAXMILHAS":"GRUPO 123","123MILHAS":"GRUPO 123"})
        t1g = top_share(Wg["R1G"])
        st.dataframe(t1g, use_container_width=True, height=280)

    # Posição 2 e 3 com "SEM OFERTAS"
    c21, c22 = st.columns(2)
    with c21:
        st.markdown("**Participação TOP2 (com 'SEM OFERTAS' quando não existe)**")
        st.dataframe(top_share(W["R2"]), use_container_width=True, height=280)
    with c22:
        st.markdown("**Participação TOP3 (com 'SEM OFERTAS' quando não existe)**")
        st.dataframe(top_share(W["R3"]), use_container_width=True, height=280)

# ======================================================================
# 2) TOP 3 AGÊNCIAS (mais vitórias em TOP1)
# ======================================================================
with tabs[1]:
    st.subheader("Top 3 Agências (mais TOP1)")
    if df.empty: st.info("Sem dados."); st.stop()
    W = winners_by_position(df)
    t = W["R1"].value_counts().rename_axis("Agência/Cia").reset_index(name="Vitórias TOP1")
    t["%"] = (t["Vitórias TOP1"] / t["Vitórias TOP1"].sum() * 100).round(2)
    t3 = t.head(3)
    st.dataframe(t3, use_container_width=True)
    st.altair_chart(make_bar(t3.rename(columns={"Vitórias TOP1":"Quantidade"}), "Quantidade", "Agência/Cia"))

# ======================================================================
# 3) TOP 3 PREÇOS MAIS BARATOS (geral filtrado)
# ======================================================================
with tabs[2]:
    st.subheader("Top 3 Preços Mais Baratos (geral)")
    if df.empty: st.info("Sem dados."); st.stop()
    t = df.sort_values(["PRECO"]).head(3)[["IDPESQUISA","AGENCIA_NORM","TRECHO","PRECO","DATAHORA_BUSCA"]]
    st.dataframe(t.rename(columns={"AGENCIA_NORM":"Agência/Cia","DATAHORA_BUSCA":"Data Busca"}), use_container_width=True)

# ======================================================================
# 4) RANKING POR AGÊNCIAS (volume e vitórias)
# ======================================================================
with tabs[3]:
    st.subheader("Ranking por Agências")
    if df.empty: st.info("Sem dados."); st.stop()
    # volume de ofertas + vitórias TOP1
    W = winners_by_position(df)
    wins = W["R1"].value_counts().rename_axis("Agência/Cia").reset_index(name="Vitórias TOP1")
    vol  = df["AGENCIA_NORM"].value_counts().rename_axis("Agência/Cia").reset_index(name="Ofertas")
    rt = vol.merge(wins, on="Agência/Cia", how="left").fillna(0)
    rt["Vitórias TOP1"] = rt["Vitórias TOP1"].astype(int)
    rt["Taxa Vitória (%)"] = (rt["Vitórias TOP1"]/rt["Ofertas"]*100).round(2)
    st.dataframe(rt.sort_values(["Vitórias TOP1","Ofertas"], ascending=False), use_container_width=True, height=420)

# ======================================================================
# 5) PREÇO POR PERÍODO DO DIA (HH da busca)
# ======================================================================
with tabs[4]:
    st.subheader("Preço por Período do Dia (HH)")
    if df.empty: st.info("Sem dados."); st.stop()
    t = df.groupby("HORA_HH", as_index=False)["PRECO"].median().rename(columns={"PRECO":"Preço Mediano"})
    st.altair_chart(make_line(t, "HORA_HH", "Preço Mediano"))

# ======================================================================
# 6) QTDE DE BUSCAS x OFERTAS
# ======================================================================
with tabs[5]:
    st.subheader("Quantidade de Buscas x Ofertas")
    if df.empty: st.info("Sem dados."); st.stop()
    searches = df["IDPESQUISA"].nunique()
    offers   = len(df)
    col1, col2 = st.columns(2)
    col1.metric("Pesquisas únicas", f"{searches:,}".replace(",", "."))
    col2.metric("Ofertas (linhas)", f"{offers:,}".replace(",", "."))
    t = pd.DataFrame({"Métrica":["Pesquisas","Ofertas"], "Qtde":[searches, offers]})
    st.altair_chart(make_bar(t.rename(columns={"Qtde":"Valor"}), "Valor", "Métrica"))

# ======================================================================
# 7) COMPORTAMENTO CIAS (share por trecho)
# ======================================================================
with tabs[6]:
    st.subheader("Comportamento Cias (share por Trecho)")
    if df.empty: st.info("Sem dados."); st.stop()
    base = df.groupby(["TRECHO","AGENCIA_NORM"]).size().rename("Qtde").reset_index()
    total_por_trecho = base.groupby("TRECHO")["Qtde"].transform("sum")
    base["%"] = (base["Qtde"]/total_por_trecho*100).round(2)
    st.dataframe(base.sort_values(["TRECHO","%"], ascending=[True, False]), use_container_width=True, height=420)

# ======================================================================
# 8) COMPETITIVIDADE (diferença para o melhor preço da pesquisa)
# ======================================================================
with tabs[7]:
    st.subheader("Competitividade (Δ para o melhor preço na mesma pesquisa)")
    if df.empty: st.info("Sem dados."); st.stop()
    # Δ-preço para o menor da pesquisa
    best = df.groupby("IDPESQUISA")["PRECO"].min().rename("BEST").reset_index()
    t = df.merge(best, on="IDPESQUISA", how="left")
    t["DELTA"] = t["PRECO"] - t["BEST"]
    agg = t.groupby("AGENCIA_NORM", as_index=False)["DELTA"].median().rename(columns={"DELTA":"Δ Mediano"})
    st.dataframe(agg.sort_values("Δ Mediano"), use_container_width=True)
    st.altair_chart(make_bar(agg.rename(columns={"Δ Mediano":"Delta"}), "Delta", "AGENCIA_NORM"))

# ======================================================================
# 9) MELHOR PREÇO DIÁRIO (por DATAHORA_BUSCA)
# ======================================================================
with tabs[8]:
    st.subheader("Melhor Preço Diário (Data da busca — col. H)")
    if df.empty: st.info("Sem dados."); st.stop()
    t = df.groupby("DATAHORA_BUSCA", as_index=False)["PRECO"].min().rename(columns={"PRECO":"Melhor Preço"})
    st.altair_chart(make_line(t, "DATAHORA_BUSCA", "Melhor Preço"))

# ======================================================================
# 10) DADOS (visão bruta filtrada)
# ======================================================================
with tabs[9]:
    st.subheader("Dados (após filtros)")
    st.dataframe(df, use_container_width=True, height=560)
