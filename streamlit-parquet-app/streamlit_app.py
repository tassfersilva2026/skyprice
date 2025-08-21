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
APP_DIR = Path(__file__).resolve().parent
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
    if ag == "BOOKINGCOM":              return "BOOKING.COM"
    if ag == "KIWICOM":                 return "KIWI.COM"
    if ag.startswith("123MILHAS") or ag == "123":    return "123MILHAS"
    if ag.startswith("MAXMILHAS") or ag == "MAX":    return "MAXMILHAS"
    if ag.startswith("CAPOVIAGENS"):    return "CAPOVIAGENS"
    if ag.startswith("FLIPMILHAS"):     return "FLIPMILHAS"
    if ag.startswith("VAIDEPROMO"):     return "VAIDEPROMO"
    if ag.startswith("KISSANDFLY"):     return "KISSANDFLY"
    if ag.startswith("ZUPPER"):         return "ZUPPER"
    if ag.startswith("MYTRIP"):         return "MYTRIP"
    if ag.startswith("GOTOGATE"):       return "GOTOGATE"
    if ag.startswith("DECOLAR"):        return "DECOLAR"
    if ag.startswith("EXPEDIA"):        return "EXPEDIA"
    if ag.startswith("GOL"):            return "GOL"
    if ag.startswith("LATAM"):          return "LATAM"
    if ag.startswith("TRIPCOM"):        return "TRIP.COM"
    if ag.startswith("VIAJANET"):       return "VIAJANET"
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
    colmap = {
        0:"IDPESQUISA", 1:"CIA", 2:"HORA_BUSCA", 3:"HORA_PARTIDA", 4:"HORA_CHEGADA",
        5:"TIPO_VOO", 6:"DATA_EMBARQUE", 7:"DATAHORA_BUSCA", 8:"AGENCIA_COMP",
        9:"PRECO", 10:"TRECHO", 11:"ADVP", 12:"RANKING"
    }
    if list(df.columns[:13]) != list(colmap.values()):
        rename = {df.columns[i]: colmap[i] for i in range(min(13, df.shape[1]))}
        df = df.rename(columns=rename)
    for c in ["HORA_BUSCA", "HORA_PARTIDA", "HORA_CHEGADA"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c].astype(str).str.strip(), errors="coerce").dt.time
    df["HORA_HH"] = pd.to_datetime(df["HORA_BUSCA"].astype(str), errors="coerce").dt.hour
    for c in ["DATA_EMBARQUE", "DATAHORA_BUSCA"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True).dt.date
    if "PRECO" in df.columns:
        df["PRECO"] = pd.to_numeric(df["PRECO"].astype(str).str.replace(r"[^\d,.-]","", regex=True).str.replace(",",".", regex=False), errors="coerce")
    if "RANKING" in df.columns:
        df["RANKING"] = pd.to_numeric(df["RANKING"], errors="coerce").astype("Int64")
    df["AGENCIA_NORM"] = df["AGENCIA_COMP"].apply(std_agencia)
    df["AGENCIA_GRUPO"] = df["AGENCIA_NORM"].replace({"MAXMILHAS":"GRUPO 123","123MILHAS":"GRUPO 123"})
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
repo_root = APP_DIR
img = _first_existing_image(repo_root)
if img:
    st.image(img.as_posix(), use_container_width=True)

# ===========================
# Filtros (horizontais, valem pra todas as abas)
# ===========================
st.markdown("---")
with st.container():
    c1, c2, c3, c4, c5 = st.columns([1.2,1.2,1,2,1.2])
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

df = df_raw.copy()
mask = pd.Series(True, index=df.index)
if dt_ini: mask &= (pd.to_datetime(df["DATAHORA_BUSCA"], errors="coerce") >= pd.Timestamp(dt_ini))
if dt_fim: mask &= (pd.to_datetime(df["DATAHORA_BUSCA"], errors="coerce") <= pd.Timestamp(dt_fim))
if advp_sel:
    mask &= df["ADVP_CANON"].isin(advp_sel)
if tr_sel:
    mask &= df["TRECHO"].isin(tr_sel)
if hh_sel:
    mask &= df["HORA_HH"].isin(hh_sel)
df = df[mask].copy()

if df.empty:
    st.info("Sem dados para os filtros."); st.stop()

st.markdown("---")

c_pesquisas, c_cobertura = st.columns([1, 2])

W = winners_by_position(df)
total_pesquisas = df['IDPESQUISA'].nunique()
rank1_count = W[W['R1'] != 'SEM OFERTAS'].shape[0]
rank2_count = W[W['R2'] != 'SEM OFERTAS'].shape[0]
rank3_count = W[W['R3'] != 'SEM OFERTAS'].shape[0]
rank1_perc = (rank1_count / total_pesquisas * 100) if total_pesquisas > 0 else 0
rank2_perc = (rank2_count / total_pesquisas * 100) if total_pesquisas > 0 else 0
rank3_perc = (rank3_count / total_pesquisas * 100) if total_pesquisas > 0 else 0

with c_pesquisas:
    st.markdown("Pesquisas únicas")
    st.title(f"{total_pesquisas:,}".replace(",", "."))

with c_cobertura:
    st.markdown("Cobertura por Ranking")
    st.subheader(f"1°: {rank1_count:,} ({rank1_perc:.1f}%) • 2°: {rank2_count:,} ({rank2_perc:.1f}%) • 3°: {rank3_count:,} ({rank3_perc:.1f}%)".replace(",", ".").replace(".", ",", 1).replace(".", "", 1))
    
st.markdown("---")

# ======================================================================
# GRUPO
# ======================================================================
st.markdown("## GRUPO")

c_grupo1, c_grupo2, c_grupo3 = st.columns(3)

# Corrigido: `top_r1_grupo` é um DataFrame.
# Use `.loc` para encontrar a linha e a coluna desejada.
with c_grupo1:
    st.markdown("1°")
    top_r1_grupo = top_share(W['R1'].replace({"MAXMILHAS":"GRUPO 123", "123MILHAS":"GRUPO 123"}))
    perc_grupo_r1 = top_r1_grupo.loc[top_r1_grupo['Agência/Cia'] == 'GRUPO 123', '%']
    perc_val = perc_grupo_r1.iloc[0] if not perc_grupo_r1.empty else 0
    st.subheader(f"{perc_val:.2f}%".replace(".", ","))
    st.progress(perc_val / 100)

with c_grupo2:
    st.markdown("2°")
    top_r2_grupo = top_share(W['R2'].replace({"MAXMILHAS":"GRUPO 123", "123MILHAS":"GRUPO 123"}))
    perc_grupo_r2 = top_r2_grupo.loc[top_r2_grupo['Agência/Cia'] == 'GRUPO 123', '%']
    perc_val = perc_grupo_r2.iloc[0] if not perc_grupo_r2.empty else 0
    st.subheader(f"{perc_val:.2f}%".replace(".", ","))
    st.progress(perc_val / 100)

with c_grupo3:
    st.markdown("3°")
    top_r3_grupo = top_share(W['R3'].replace({"MAXMILHAS":"GRUPO 123", "123MILHAS":"GRUPO 123"}))
    perc_grupo_r3 = top_r3_grupo.loc[top_r3_grupo['Agência/Cia'] == 'GRUPO 123', '%']
    perc_val = perc_grupo_r3.iloc[0] if not perc_grupo_r3.empty else 0
    st.subheader(f"{perc_val:.2f}%".replace(".", ","))
    st.progress(perc_val / 100)

st.markdown("---")

# ======================================================================
# FLIPMILHAS
# ======================================================================
st.markdown("## FLIPMILHAS")

c_flip1, c_flip2, c_flip3 = st.columns(3)

with c_flip1:
    st.markdown("1°")
    top_r1_flip = top_share(W['R1'])
    perc_flip_r1 = top_r1_flip.loc[top_r1_flip['Agência/Cia'] == 'FLIPMILHAS', '%']
    perc_val = perc_flip_r1.iloc[0] if not perc_flip_r1.empty else 0
    st.subheader(f"{perc_val:.2f}%".replace(".", ","))
    st.progress(perc_val / 100)

with c_flip2:
    st.markdown("2°")
    top_r2_flip = top_share(W['R2'])
    perc_flip_r2 = top_r2_flip.loc[top_r2_flip['Agência/Cia'] == 'FLIPMILHAS', '%']
    perc_val = perc_flip_r2.iloc[0] if not perc_flip_r2.empty else 0
    st.subheader(f"{perc_val:.2f}%".replace(".", ","))
    st.progress(perc_val / 100)

with c_flip3:
    st.markdown("3°")
    top_r3_flip = top_share(W['R3'])
    perc_flip_r3 = top_r3_flip.loc[top_r3_flip['Agência/Cia'] == 'FLIPMILHAS', '%']
    perc_val = perc_flip_r3.iloc[0] if not perc_flip_r3.empty else 0
    st.subheader(f"{perc_val:.2f}%".replace(".", ","))
    st.progress(perc_val / 100)
