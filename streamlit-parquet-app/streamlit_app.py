# streamlit_app.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime, date, time as dtime
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Skyscanner — Painel", layout="wide", initial_sidebar_state="expanded")

APP_DIR   = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "OFERTAS.parquet"

# ============================== UTILIDADES GERAIS ==============================
import re
def _norm_hhmmss(v: object) -> str | None:
    s = str(v or "").strip()
    m = re.search(r"(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?", s)
    if not m:
        return None
    hh = max(0, min(23, int(m.group(1))))
    mm = max(0, min(59, int(m.group(2))))
    ss = max(0, min(59, int(m.group(3) or 0)))
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

def std_agencia(raw: str) -> str:
    ag = (raw or "").strip().upper()
    if ag == "BOOKINGCOM":            return "BOOKING.COM"
    if ag == "KIWICOM":               return "KIWI.COM"
    if ag.startswith("123MILHAS") or ag == "123":   return "123MILHAS"
    if ag.startswith("MAXMILHAS") or ag == "MAX":     return "MAXMILHAS"
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

def std_cia(raw: str) -> str:
    s = (str(raw) or "").strip().upper()
    s_simple = "".join(ch for ch in s if ch.isalnum() or ch.isspace())
    if s in {"AD", "AZU"} or s.startswith("AZUL") or "AZUL" in s_simple: 
        return "AZUL"
    if s in {"G3"} or s.startswith("GOL") or "GOL" in s_simple:          
        return "GOL"
    if s in {"LA", "JJ"} or s.startswith("TAM") or s.startswith("LATAM") or "LATAM" in s_simple or "TAM" in s_simple:
        return "LATAM"
    if s in {"AZUL", "GOL", "LATAM"}: 
        return s
    return s

def advp_nearest(x) -> int:
    try:
        v = float(str(x).replace(",", "."))
    except Exception:
        v = np.nan
    if np.isnan(v):
        v = 1
    return min([1, 5, 11, 17, 30], key=lambda k: abs(v - k))

@st.cache_data(show_spinner=True)
def load_base(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Arquivo obrigatório não encontrado: {path.as_posix()}")
        st.stop()
    df = pd.read_parquet(path)

    # Renomeia as colunas fixas (0..12)
    colmap = {
        0:"IDPESQUISA",1:"CIA",2:"HORA_BUSCA",3:"HORA_PARTIDA",4:"HORA_CHEGADA",
        5:"TIPO_VOO",6:"DATA_EMBARQUE",7:"DATAHORA_BUSCA",8:"AGENCIA_COMP",9:"PRECO",
        10:"TRECHO",11:"ADVP",12:"RANKING"
    }
    if list(df.columns[:13]) != list(colmap.values()):
        rename = {df.columns[i]: colmap[i] for i in range(min(13, df.shape[1]))}
        df = df.rename(columns=rename)

    # Usa o campo __DTKEY__ do parquet, se existir, garantindo que seja datetime
    if "__DTKEY__" in df.columns:
        df["__DTKEY__"] = pd.to_datetime(df["__DTKEY__"], errors="coerce")

    # Normaliza as horas (formata a string como HH:MM:SS)
    for c in ["HORA_BUSCA", "HORA_PARTIDA", "HORA_CHEGADA"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c].astype(str).str.strip(), errors="coerce").dt.strftime("%H:%M:%S")
    df["HORA_HH"] = pd.to_datetime(df["HORA_BUSCA"], errors="coerce").dt.hour

    # Converte datas
    for c in ["DATA_EMBARQUE", "DATAHORA_BUSCA"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)

    # Converte e formata o preço
    if "PRECO" in df.columns:
        df["PRECO"] = (
            df["PRECO"].astype(str)
            .str.replace(r"[^\d,.-]", "", regex=True)
            .str.replace(",", ".", regex=False)
        )
        df["PRECO"] = pd.to_numeric(df["PRECO"], errors="coerce")

    if "RANKING" in df.columns:
        df["RANKING"] = pd.to_numeric(df["RANKING"], errors="coerce").astype("Int64")

    df["AGENCIA_NORM"] = df["AGENCIA_COMP"].apply(std_agencia)
    df["ADVP_CANON"]   = df["ADVP"].apply(advp_nearest)
    df["CIA_NORM"]     = df.get("CIA", pd.Series([None]*len(df))).apply(std_cia)
    return df

def winners_by_position(df: pd.DataFrame) -> pd.DataFrame:
    base = pd.DataFrame({"IDPESQUISA": df["IDPESQUISA"].unique()})
    for r in (1, 2, 3):
        s = (df[df["RANKING"] == r]
             .sort_values(["IDPESQUISA"])
             .drop_duplicates(subset=["IDPESQUISA"]))
        base = base.merge(
            s[["IDPESQUISA", "AGENCIA_NORM"]].rename(columns={"AGENCIA_NORM": f"R{r}"}),
            on="IDPESQUISA", how="left"
        )
    for r in (1, 2, 3):
        base[f"R{r}"] = base[f"R{r}"].fillna("SEM OFERTAS")
    return base

def fmt_int(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")

def last_update_from_cols(df: pd.DataFrame) -> str:
    """
    Retorna a data/hora da última pesquisa:
      - Se existir o __DTKEY__ (do parquet), utiliza-o;
      - Caso contrário, utiliza DATAEMBARQUE e HORA_BUSCA.
    """
    if df.empty:
        return "—"

    if "__DTKEY__" in df.columns:
        max_dt = pd.to_datetime(df["__DTKEY__"], errors="coerce").max()
        if pd.isna(max_dt):
            return "—"
        return max_dt.strftime("%d/%m/%Y - %H:%M:%S")

    # Fallback: utiliza DATAEMBARQUE e HORA_BUSCA
    max_d = pd.to_datetime(df["DATAEMBARQUE"], errors="coerce").max()
    if pd.isna(max_d):
        return "—"
    same_day = df[
        pd.to_datetime(df["DATAEMBARQUE"], errors="coerce").dt.date == max_d.date()
    ]
    hh = pd.to_datetime(same_day["HORA_BUSCA"], errors="coerce").dt.time
    max_h = max([h for h in hh if pd.notna(h)], default=None)
    if isinstance(max_h, dtime):
        return f"{max_d.strftime('%d/%m/%Y')} - {max_h.strftime('%H:%M:%S')}"
    return f"{max_d.strftime('%d/%m/%Y')}"

# ---- CSS Global
GLOBAL_TABLE_CSS = """
<style>
table { width:100% !important; }
.dataframe { width:100% !important; }
</style>
"""
st.markdown(GLOBAL_TABLE_CSS, unsafe_allow_html=True)

# ---- Estilos dos Cards (Painel)
CARD_CSS = """
<style>
  .cards-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
  @media (max-width: 1100px) { .cards-grid { grid-template-columns: repeat(2, minmax(0,1fr)); } }
  @media (max-width: 700px) { .cards-grid { grid-template-columns: 1fr; } }
  .card { border:1px solid #e9e9ee; border-radius:14px; padding:10px 12px;
          background:#fff; box-shadow:0 1px 2px rgba(0,0,0,0.04); }
  .card .title { font-weight:650; font-size:15px; margin-bottom:8px; }
  .goldcard  { background:#FFF9E5; border-color:#D4AF37; }
  .silvercard{ background:#F7F7FA; border-color:#C0C0C0; }
  .bronzecard{ background:#FFF1E8; border-color:#CD7F32; }
  .row { display:flex; gap:8px; }
  .item { flex:1; display:flex; align-items:center;
         justify-content:space-between; gap:8px; padding:8px 10px;
         border-radius:10px; border:1px solid #e3e3e8; background:#fafbfc; }
  .pos { font-weight:700; font-size:12px; opacity:0.85; }
  .pct { font-size:16px; font-weight:650; }
</style>
"""

CARDS_STACK_CSS = """
<style>
  .cards-stack { display:flex; flex-direction:column; gap:10px; }
  .cards-stack .card { width:100%; }
  .stack-title { font-weight:800; padding:8px 10px; margin:6px 0 10px 0;
                 border-radius:10px; border:1px solid #e9e9ee;
                 background:#f8fafc; color:#0A2A6B; }
</style>
"""

def card_html(nome: str, p1: float, p2: float, p3: float, rank_cls: str = "") -> str:
    p1 = max(0.0, min(100.0, float(p1 or 0.0)))
    p2 = max(0.0, min(100.0, float(p2 or 0.0)))
    p3 = max(0.0, min(100.0, float(p3 or 0.0)))
    cls = f"card {rank_cls}".strip()
    return (
        f"<div class='{cls}'>"
        f"  <div class='title'>{nome}</div>"
        f"  <div class='row'>"
        f"    <div class='item'><span class='pos'>1º</span><span class='pct'>{p1:.2f}%</span></div>"
        f"    <div class='item'><span class='pos'>2º</span><span class='pct'>{p2:.2f}%</span></div>"
        f"    <div class='item'><span class='pos'>3º</span><span class='pct'>{p3:.2f}%</span></div>"
        f"  </div>"
        f"</div>"
    )

# ---- Gráficos Utilitários
def make_bar(df: pd.DataFrame, x_col: str, y_col: str, sort_y_desc: bool = True):
    d = df[[y_col, x_col]].copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = d[y_col].astype(str)
    d = d.dropna(subset=[x_col])
    if sort_y_desc:
        d = d.sort_values(x_col, ascending=False)
    if d.empty:
        return alt.Chart(pd.DataFrame({x_col: [], y_col: []})).mark_bar()
    return alt.Chart(d).mark_bar().encode(
        x=alt.X(f"{x_col}:Q", title=x_col),
        y=alt.Y(f"{y_col}:N", sort="-x", title=y_col),
        tooltip=[f"{y_col}:N", f"{x_col}:Q"],
    ).properties(height=300)

def make_line(df: pd.DataFrame, x_col: str, y_col: str, color: str | None = None):
    cols = [x_col, y_col] + ([color] if color else [])
    d = df[cols].copy()
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
    enc = dict(x=x_enc, y=alt.Y(f"{y_col}:Q", title=y_col), tooltip=[f"{x_col}", f"{y_col}:Q"])
    if color:
        enc["color"] = alt.Color(f"{color}:N", title=color)
    return alt.Chart(d).mark_line(point=True).encode(**enc).properties(height=300)

# ---- Registro de Abas
TAB_REGISTRY: List[Tuple[str, Callable]] = []
def register_tab(label: str):
    def _wrap(fn: Callable):
        TAB_REGISTRY.append((label, fn))
        return fn
    return _wrap

# Se nenhuma aba foi registrada, adiciona uma aba padrão
if not TAB_REGISTRY:
    @register_tab("Painel")
    def default_tab(df_raw: pd.DataFrame):
        st.subheader("Painel")
        st.write("Conteúdo padrão do Painel.")

# ---- Filtros
def _init_filter_state(df_raw: pd.DataFrame):
    if "flt" in st.session_state:
        return
    dmin = pd.to_datetime(df_raw["DATAEMBARQUE"], errors="coerce").min()
    dmax = pd.to_datetime(df_raw["DATAEMBARQUE"], errors="coerce").max()
    st.session_state["flt"] = {
        "dt_ini": (dmin.date() if pd.notna(dmin) else date(2000, 1, 1)),
        "dt_fim": (dmax.date() if pd.notna(dmax) else date.today()),
        "advp": [], "trechos": [], "hh": [], "cia": [],
    }

def render_filters(df_raw: pd.DataFrame, key_prefix: str = "flt"):
    _init_filter_state(df_raw)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns([1.1, 1.1, 1, 2, 1, 1.4])

    dmin_abs = pd.to_datetime(df_raw["DATAEMBARQUE"], errors="coerce").min()
    dmax_abs = pd.to_datetime(df_raw["DATAEMBARQUE"], errors="coerce").max()
    dmin_abs = dmin_abs.date() if pd.notna(dmin_abs) else date(2000, 1, 1)
    dmax_abs = dmax_abs.date() if pd.notna(dmax_abs) else date.today()

    with c1:
        dt_ini = st.date_input("Data inicial", key=f"{key_prefix}_dtini",
                               value=st.session_state["flt"]["dt_ini"],
                               min_value=dmin_abs, max_value=dmax_abs, format="DD/MM/YYYY")
    with c2:
        dt_fim = st.date_input("Data final", key=f"{key_prefix}_dtfim",
                               value=st.session_state["flt"]["dt_fim"],
                               min_value=dmin_abs, max_value=dmax_abs, format="DD/MM/YYYY")
    with c3:
        advp_all = sorted(set(pd.to_numeric(df_raw["ADVP_CANON"], errors="coerce")
                              .dropna().astype(int).tolist()))
        advp_sel = st.multiselect("ADVP", options=advp_all,
                                  default=st.session_state["flt"]["advp"],
                                  key=f"{key_prefix}_advp")
    with c4:
        trechos_all = sorted([t for t in df_raw["TRECHO"]
                              .dropna().unique().tolist()
                              if str(t).strip() != ""])
        tr_sel = st.multiselect("Trechos", options=trechos_all,
                                default=st.session_state["flt"]["trechos"],
                                key=f"{key_prefix}_trechos")
    with c5:
        hh_sel = st.multiselect("Hora da busca", options=list(range(24)),
                                 default=st.session_state["flt"]["hh"],
                                 key=f"{key_prefix}_hh")
    with c6:
        cia_presentes = set(str(x).upper()
                            for x in df_raw.get("CIA_NORM", pd.Series([], dtype=str))
                            .dropna().unique())
        ordem = ["AZUL", "GOL", "LATAM"]
        cia_opts = [c for c in ordem if (not cia_presentes or c in cia_presentes)] or ordem
        cia_default = [c for c in st.session_state["flt"]["cia"] if c in cia_opts]
        cia_sel = st.multiselect("Cia (Azul/Gol/Latam)", options=cia_opts,
                                 default=cia_default, key=f"{key_prefix}_cia")

    st.session_state["flt"] = {
        "dt_ini": dt_ini, "dt_fim": dt_fim,
        "advp": advp_sel or [],
        "trechos": tr_sel or [],
        "hh": hh_sel or [],
        "cia": cia_sel or []
    }

    mask = pd.Series(True, index=df_raw.index)
    mask &= (pd.to_datetime(df_raw["DATAEMBARQUE"], errors="coerce")
             >= pd.Timestamp(dt_ini))
    mask &= (pd.to_datetime(df_raw["DATAEMBARQUE"], errors="coerce")
             <= pd.Timestamp(dt_fim))
    if advp_sel:
        mask &= df_raw["ADVP_CANON"].isin(advp_sel)
    if tr_sel:
        mask &= df_raw["TRECHO"].isin(tr_sel)
    if hh_sel:
        mask &= df_raw["HORA_HH"].isin(hh_sel)
    if st.session_state["flt"]["cia"]:
        mask &= df_raw["CIA_NORM"].astype(str).str.upper() \
                .isin(st.session_state["flt"]["cia"])

    df = df_raw[mask].copy()
    st.caption(
        f"Linhas após filtros: {fmt_int(len(df))} • Última atualização: {last_update_from_cols(df)}"
    )
    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
    return df

# ---- MAIN
def main():
    df_raw = load_base(DATA_PATH)
    for ext in ("*.png","*.jpg","*.jpeg","*.gif","*.webp"):
        imgs = list(APP_DIR.glob(ext))
        if imgs:
            st.image(imgs[0].as_posix(), use_container_width=True)
            break

    labels = [label for label, _ in TAB_REGISTRY]
    if not labels:
        st.info("Nenhuma aba registrada.")
        return
    tabs = st.tabs(labels)
    for i, (label, fn) in enumerate(TAB_REGISTRY):
        with tabs[i]:
            try:
                fn(df_raw)
            except Exception as e:
                st.error(f"Erro na aba {label}")
                st.exception(e)

if __name__ == "__main__":
    main()
