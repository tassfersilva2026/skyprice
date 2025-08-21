# streamlit_app.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime, date, time as dtime
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Skyscanner — Painel", layout="wide", initial_sidebar_state="expanded")

APP_DIR   = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "OFERTAS.parquet"

# =========================================================
# Helpers de normalização e carregamento
# =========================================================
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

def advp_nearest(x) -> int:
    try:
        v = float(str(x).replace(",", "."))
    except Exception:
        v = np.nan
    if np.isnan(v): v = 1
    return min([1,5,11,17,30], key=lambda k: abs(v-k))

@st.cache_data(show_spinner=True)
def load_base(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Arquivo obrigatório não encontrado: `{path.as_posix()}`"); st.stop()
    df = pd.read_parquet(path)

    # Mapa por posição (A..M) caso os nomes venham diferentes
    colmap = {
        0:"IDPESQUISA", 1:"CIA", 2:"HORA_BUSCA", 3:"HORA_PARTIDA", 4:"HORA_CHEGADA",
        5:"TIPO_VOO", 6:"DATA_EMBARQUE", 7:"DATAHORA_BUSCA", 8:"AGENCIA_COMP",
        9:"PRECO", 10:"TRECHO", 11:"ADVP", 12:"RANKING"
    }
    if list(df.columns[:13]) != list(colmap.values()):
        rename = {df.columns[i]: colmap[i] for i in range(min(13, df.shape[1]))}
        df = df.rename(columns=rename)

    # Horas texto -> "HH:MM:SS" + coluna HH (somente hora)
    for c in ["HORA_BUSCA", "HORA_PARTIDA", "HORA_CHEGADA"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c].astype(str).str.strip(), errors="coerce").dt.strftime("%H:%M:%S")
    df["HORA_HH"] = pd.to_datetime(df["HORA_BUSCA"], errors="coerce").dt.hour

    # Datas (dd/mm/aaaa)
    for c in ["DATA_EMBARQUE", "DATAHORA_BUSCA"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)

    # Preço
    if "PRECO" in df.columns:
        df["PRECO"] = (df["PRECO"].astype(str)
                       .str.replace(r"[^\d,.-]", "", regex=True)
                       .str.replace(",", ".", regex=False))
        df["PRECO"] = pd.to_numeric(df["PRECO"], errors="coerce")

    # Ranking
    if "RANKING" in df.columns:
        df["RANKING"] = pd.to_numeric(df["RANKING"], errors="coerce").astype("Int64")

    # Normalizações
    df["AGENCIA_NORM"]  = df["AGENCIA_COMP"].apply(std_agencia)
    df["AGENCIA_GRUPO"] = df["AGENCIA_NORM"].replace({"MAXMILHAS":"GRUPO 123", "123MILHAS":"GRUPO 123"})
    df["ADVP_CANON"]    = df["ADVP"].apply(advp_nearest)
    return df

def winners_by_position(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna por IDPESQUISA a agência vencedora em R1, R2 e R3 (preenche 'SEM OFERTAS' se não houver)."""
    base = pd.DataFrame({"IDPESQUISA": df["IDPESQUISA"].unique()})
    for r in (1,2,3):
        s = (df[df["RANKING"]==r]
             .sort_values(["IDPESQUISA"])
             .drop_duplicates(subset=["IDPESQUISA"]))
        base = base.merge(s[["IDPESQUISA","AGENCIA_NORM"]].rename(columns={"AGENCIA_NORM":f"R{r}"}),
                          on="IDPESQUISA", how="left")
    for r in (1,2,3):
        base[f"R{r}"] = base[f"R{r}"].fillna("SEM OFERTAS")
    return base

def fmt_int(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")

def last_update_from_cols(df: pd.DataFrame) -> str:
    """Última atualização = maior data (H) e, nessa data, maior HORA_BUSCA (C)."""
    if df.empty: return "—"
    max_d = pd.to_datetime(df["DATAHORA_BUSCA"], errors="coerce").max()
    if pd.isna(max_d): return "—"
    same_day = df[pd.to_datetime(df["DATAHORA_BUSCA"], errors="coerce").dt.date == max_d.date()]
    hh = pd.to_datetime(same_day["HORA_BUSCA"], errors="coerce").dt.time
    max_h = max([h for h in hh if pd.notna(h)], default=None)
    if isinstance(max_h, dtime):
        return f"{max_d.strftime('%d/%m/%Y')} - {max_h.strftime('%H:%M:%S')}"
    return f"{max_d.strftime('%d/%m/%Y')}"

# =========================================================
# Helpers de gráfico (com coerção de tipos → evita SchemaValidationError)
# =========================================================
def make_bar(df: pd.DataFrame, x_col: str, y_col: str, sort_y_desc: bool = True):
    # garante tipos: x numérico, y string
    d = df[[y_col, x_col]].copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = d[y_col].astype(str)
    d = d.dropna(subset=[x_col])

    if sort_y_desc:
        d = d.sort_values(x_col, ascending=False)

    if d.empty:
        return alt.Chart(pd.DataFrame({x_col: [], y_col: []})).mark_bar()

    return (
        alt.Chart(d)
        .mark_bar()
        .encode(
            x=alt.X(f"{x_col}:Q", title=x_col),
            y=alt.Y(f"{y_col}:N", sort="-x", title=y_col),
            tooltip=[
                alt.Tooltip(f"{y_col}:N", title=y_col),
                alt.Tooltip(f"{x_col}:Q", title=x_col),
            ],
        )
        .properties(height=320, use_container_width=True)
    )

def make_line(df: pd.DataFrame, x_col: str, y_col: str, color: str | None = None):
    cols = [x_col, y_col] + ([color] if color else [])
    d = df[cols].copy()

    # X temporal se possível; senão numérico
    try:
        d[x_col] = pd.to_datetime(d[x_col], errors="raise")
        x_enc = alt.X(f"{x_col}:T", title=x_col)
    except Exception:
        d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
        x_enc = alt.X(f"{x_col}:Q", title=x_col)

    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    if color:
        d[color] = d[color].astype(str)

    d = d.dropna(subset=[x_col, y_col])
    if d.empty:
        return alt.Chart(pd.DataFrame({x_col: [], y_col: []})).mark_line()

    enc = dict(
        x=x_enc,
        y=alt.Y(f"{y_col}:Q", title=y_col),
        tooltip=[alt.Tooltip(f"{x_col}", title=x_col), alt.Tooltip(f"{y_col}:Q", title=y_col)],
    )
    if color:
        enc["color"] = alt.Color(f"{color}:N", title=color)

    return alt.Chart(d).mark_line(point=True).encode(**enc).properties(height=320, use_container_width=True)

# =========================================================
# Load base
# =========================================================
