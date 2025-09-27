# streamlit_app.py — versão com botão ↻ de atualização discreto ao final dos filtros
from __future__ import annotations
from pathlib import Path
from datetime import date
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import re

# ─────────────────────────── CONFIG ───────────────────────────
st.set_page_config(page_title="Skyscanner — Painel", layout="wide", initial_sidebar_state="expanded")
APP_DIR   = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "OFERTASCONSOLIDADO_OFERTAS.parquet"   # ajuste se necessário

# ─────────────────────────── HELPERS ──────────────────────────
def _norm_hhmmss(v: object) -> str | None:
    s = str(v or "").strip()
    m = re.search(r"(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?", s)
    if not m: return None
    hh = max(0, min(23, int(m.group(1))))
    mm = max(0, min(59, int(m.group(2))))
    ss = max(0, min(59, int(m.group(3) or 0)))
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

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

def std_cia(raw: str) -> str:
    s = (str(raw) or "").strip().upper()
    s_simple = "".join(ch for ch in s if ch.isalnum() or ch.isspace())
    if s in {"AD", "AZU"} or s.startswith("AZUL") or "AZUL" in s_simple: return "AZUL"
    if s in {"G3"} or s.startswith("GOL") or "GOL" in s_simple:          return "GOL"
    if s in {"LA", "JJ"} or s.startswith("TAM") or s.startswith("LATAM") or "LATAM" in s_simple or "TAM" in s_simple:
        return "LATAM"
    if s in {"AZUL", "GOL", "LATAM"}: return s
    return s

def advp_nearest(x) -> int:
    try: v = float(str(x).replace(",", "."))
    except Exception: v = np.nan
    if np.isnan(v): v = 1
    return min([1, 5, 11, 17, 30], key=lambda k: abs(v - k))

def fmt_int(n: int) -> str:
    try: return f"{int(n):,}".replace(",", ".")
    except Exception: return "0"

def fmt_num0_br(x):
    try:
        v = float(x)
        if not np.isfinite(v): return "-"
        return f"{v:,.0f}".replace(",", ".")
    except Exception:
        return "-"

def fmt_pct2_br(v):
    try:
        x = float(v)
        if not np.isfinite(x): return "-"
        return f"{x:.2f}%".replace(".", ",")
    except Exception:
        return "-"

# ─────────────────────── CARREGAMENTO (com cache) ───────────────────────
@st.cache_data(show_spinner=True)
def load_base(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Arquivo obrigatório não encontrado: {path.as_posix()}"); st.stop()

    df = pd.read_parquet(path)

    # Padroniza primeiras 13 colunas se necessário (compat)
    colmap = {0:"IDPESQUISA",1:"CIA",2:"HORA_BUSCA",3:"HORA_PARTIDA",4:"HORA_CHEGADA",
              5:"TIPO_VOO",6:"DATA_EMBARQUE",7:"DATAHORA_BUSCA",8:"AGENCIA_COMP",9:"PRECO",
              10:"TRECHO",11:"ADVP",12:"RANKING"}
    if list(df.columns[:13]) != list(colmap.values()):
        rename = {df.columns[i]: colmap[i] for i in range(min(13, df.shape[1]))}
        df = df.rename(columns=rename)

    # Normalizações
    for c in ["HORA_BUSCA","HORA_PARTIDA","HORA_CHEGADA"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c].astype(str).str.strip(), errors="coerce").dt.strftime("%H:%M:%S")

    df["HORA_HH"] = pd.to_datetime(df.get("HORA_BUSCA"), errors="coerce").dt.hour

    for c in ["DATA_EMBARQUE","DATAHORA_BUSCA"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)

    if "PRECO" in df.columns:
        df["PRECO"] = (
            df["PRECO"].astype(str)
            .str.replace(r"[^\d,.-]", "", regex=True)
            .str.replace(",", ".", regex=False)
        )
        df["PRECO"] = pd.to_numeric(df["PRECO"], errors="coerce")

    if "RANKING" in df.columns:
        df["RANKING"] = pd.to_numeric(df["RANKING"], errors="coerce").astype("Int64")

    df["AGENCIA_NORM"] = df.get("AGENCIA_COMP", pd.Series([], dtype=str)).apply(std_agencia)
    df["ADVP_CANON"]   = df.get("ADVP", pd.Series([], dtype=str)).apply(lambda x: advp_nearest(x) if pd.notna(x) else np.nan)
    df["CIA_NORM"]     = df.get("CIA", pd.Series([None]*len(df))).apply(std_cia)

    # __DTKEY__ robusto
    dt_base = pd.to_datetime(df.get("DATAHORA_BUSCA"), errors="coerce", dayfirst=True)
    def _hh_to_sec(hs: object) -> float:
        s = _norm_hhmmss(hs)
        if not s: return np.nan
        hh, mm, ss = [int(x) for x in s.split(":")]
        return hh*3600 + mm*60 + ss
    hora_sec = pd.to_numeric(df.get("HORA_BUSCA", pd.Series([np.nan]*len(df))).map(_hh_to_sec), errors="coerce")
    dt_norm = pd.to_datetime(dt_base.dt.date, errors="coerce")
    dtkey = dt_norm + pd.to_timedelta(hora_sec.fillna(0), unit="s")
    mask_dt_ok = pd.notna(dt_base); mask_h_ok  = pd.notna(hora_sec)
    dtkey = dtkey.where(~mask_dt_ok, dt_base.dt.normalize() + pd.to_timedelta(hora_sec.fillna(0), unit="s"))
    dtkey = dtkey.where(mask_dt_ok | mask_h_ok, pd.NaT)
    df["__DTKEY__"] = dtkey

    return df

# ─────────────────────────── ESTILOS/TABELAS ───────────────────────────
GLOBAL_TABLE_CSS = """
<style>
table { width:100% !important; }
.dataframe { width:100% !important; }
</style>
"""
st.markdown(GLOBAL_TABLE_CSS, unsafe_allow_html=True)

CARD_CSS = """
<style>
  .cards-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
  @media (max-width: 1100px) { .cards-grid { grid-template-columns: repeat(2, minmax(0,1fr)); } }
  @media (max-width: 700px) { .cards-grid { grid-template-columns: 1fr; } }
  .card { border:1px solid #e9e9ee; border-radius:14px; padding:10px 12px; background:#fff; box-shadow:0 1px 2px rgba(0,0,0,.04); }
  .card .title { font-weight:650; font-size:15px; margin-bottom:8px; }
  .goldcard{background:#FFF9E5;border-color:#D4AF37;}
  .silvercard{background:#F7F7FA;border-color:#C0C0C0;}
  .bronzecard{background:#FFF1E8;border-color:#CD7F32;}
  .row{display:flex;gap:8px;}
  .item{flex:1;display:flex;align-items:center;justify-content:space-between;gap:8px;padding:8px 10px;border-radius:10px;border:1px solid #e3e3e8;background:#fafbfc;}
  .pos{font-weight:700;font-size:12px;opacity:.85;}
  .pct{font-size:16px;font-weight:650;}
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)

CARDS_STACK_CSS = """
<style>
  .cards-stack { display:flex; flex-direction:column; gap:10px; }
  .cards-stack .card { width:100%; }
  .stack-title { font-weight:800; padding:8px 10px; margin:6px 0 10px 0; border-radius:10px; border:1px solid #e9e9ee; background:#f8fafc; color:#0A2A6B; }
</style>
"""
st.markdown(CARDS_STACK_CSS, unsafe_allow_html=True)

def card_html(nome: str, p1: float, p2: float, p3: float, rank_cls: str = "") -> str:
    p1 = max(0.0, min(100.0, float(p1 or 0.0)))
    p2 = max(0.0, min(100.0, float(p2 or 0.0)))
    p3 = max(0.0, min(100.0, float(p3 or 0.0)))
    cls = f"card {rank_cls}".strip()
    return (
        f"<div class='{cls}'>"
        f"<div class='title'>{nome}</div>"
        f"<div class='row'>"
        f"<div class='item'><span class='pos'>1º</span><span class='pct'>{p1:.2f}%</span></div>"
        f"<div class='item'><span class='pos'>2º</span><span class='pct'>{p2:.2f}%</span></div>"
        f"<div class='item'><span class='pos'>3º</span><span class='pct'>{p3:.2f}%</span></div>"
        f"</div></div>"
    )

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

# Heatmap helpers
BLUE  = "#cfe3ff"; ORANGE= "#fdd0a2"; GREEN = "#c7e9c0"; YELLOW= "#fee391"; PINK  = "#f1b6da"
def _hex_to_rgb(h): return tuple(int(h[i:i+2], 16) for i in (1,3,5))
def _rgb_to_hex(t): return f"#{t[0]:02x}{t[1]:02x}{t[2]:02x}"
def _blend(c_from, c_to, t):
    f, to = _hex_to_rgb(c_from), _hex_to_rgb(c_to)
    return _rgb_to_hex(tuple(int(round(f[i] + (to[i]-f[i])*t)) for i in range(3)))
def make_scale(base_hex, steps=5): return [_blend("#ffffff", base_hex, k/(steps-1)) for k in range(steps)]
SCALE_BLUE   = make_scale(BLUE)
SCALE_ORANGE = make_scale(ORANGE)
SCALE_GREEN  = make_scale(GREEN)
SCALE_YELLOW = make_scale(YELLOW)
SCALE_PINK   = make_scale(PINK)

def _pick_scale(colname: str):
    u = str(colname).upper()
    if "MAXMILHAS" in u:   return SCALE_GREEN
    if "123" in u:         return SCALE_ORANGE
    if "FLIP" in u:        return SCALE_YELLOW
    if "CAPO" in u:        return SCALE_PINK
    return SCALE_BLUE

def _is_null_like(v) -> bool:
    if v is None: return True
    if isinstance(v, float) and np.isnan(v): return True
    if isinstance(v, str) and v.strip().lower() in {"none", "nan", ""}: return True
    return False

def style_heatmap_discrete(styler: pd.io.formats.style.Styler, col: str, scale_colors: list[str]):
    s = pd.to_numeric(styler.data[col], errors="coerce")
    if s.notna().sum() == 0:
        return styler
    try:
        bins = pd.qcut(s.rank(method="average"), q=5, labels=False, duplicates="drop")
    except Exception:
        bins = pd.cut(s.rank(method="average"), bins=5, labels=False)
    bins = bins.fillna(-1).astype(int)
    def _fmt(val, idx):
        if pd.isna(val) or bins.iloc[idx] == -1: return "background-color:#ffffff;color:#111111"
        color = scale_colors[int(bins.iloc[idx])]
        return f"background-color:{color};color:#111111"
    styler = styler.apply(lambda col_vals: [_fmt(v, i) for i, v in enumerate(col_vals)], subset=[col])
    return styler

def style_smart_colwise(df_show: pd.DataFrame, fmt_map: dict, grad_cols: list[str]):
    sty = (df_show.style
           .set_properties(**{"background-color": "#FFFFFF", "color": "#111111"})
           .set_table_attributes('style="width:100%; table-layout:fixed"'))
    if fmt_map:
        sty = sty.format(fmt_map, na_rep="-")
    for c in grad_cols:
        if c in df_show.columns:
            sty = style_heatmap_discrete(sty, c, _pick_scale(c))
    try:
        sty = sty.hide(axis="index")
    except Exception:
        try:
            sty = sty.hide_index()
        except Exception:
            pass
    sty = sty.applymap(lambda v: "background-color: #FFFFFF; color: #111111" if _is_null_like(v) else "")
    sty = sty.set_table_styles([{"selector":"tbody td, th","props":[("border","1px solid #EEE")]}])
    return sty

def show_table(df: pd.DataFrame, styler: pd.io.formats.style.Styler | None = None, caption: str | None = None):
    if caption:
        st.markdown(f"**{caption}**")
    try:
        if styler is not None:
            st.markdown(styler.to_html(), unsafe_allow_html=True)
        else:
            st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.warning(f"Falha ao aplicar estilo ({e}). Exibindo tabela simples.")
        st.dataframe(df, use_container_width=True)

# ─────────────────────────── REGISTRO DE ABAS ───────────────────────────
TAB_REGISTRY: List[Tuple[str, Callable]] = []
def register_tab(label: str):
    def _wrap(fn: Callable):
        TAB_REGISTRY.append((label, fn))
        return fn
    return _wrap

# ─────────────────────────── FILTROS + BOTÃO ↻ ───────────────────────────
def last_update_from_cols(df: pd.DataFrame) -> str:
    ts = pd.to_datetime(df.get("__DTKEY__"), errors="coerce").max()
    if pd.isna(ts):
        ts = pd.to_datetime(df.get("DATAHORA_BUSCA"), errors="coerce").max()
    return ts.strftime("%d/%m/%Y - %H:%M:%S") if pd.notna(ts) else "—"

def _init_filter_state(df_raw: pd.DataFrame):
    if "flt" in st.session_state: return
    base_dt = pd.to_datetime(df_raw.get("__DTKEY__"), errors="coerce")
    if base_dt.isna().all():
        base_dt = pd.to_datetime(df_raw.get("DATAHORA_BUSCA"), errors="coerce")
    dmin = base_dt.min(); dmax = base_dt.max()
    st.session_state["flt"] = {
        "dt_ini": (dmin.date() if pd.notna(dmin) else date(2000, 1, 1)),
        "dt_fim": (dmax.date() if pd.notna(dmax) else date.today()),
        "advp": [], "trechos": [], "hh": [], "cia": [],
    }

def render_filters(df_raw: pd.DataFrame, key_prefix: str = "flt"):
    _init_filter_state(df_raw)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # 7 colunas — a última (c7) é o botão ↻ pequeno no fim da linha
    c1, c2, c3, c4, c5, c6, c7 = st.columns([1.1, 1.1, 1.0, 2.0, 1.0, 1.4, 0.32])

    base_dt = pd.to_datetime(df_raw.get("__DTKEY__"), errors="coerce")
    if base_dt.isna().all():
        base_dt = pd.to_datetime(df_raw.get("DATAHORA_BUSCA"), errors="coerce", dayfirst=True)
    dmin_abs = base_dt.min()
    dmax_abs = base_dt.max()
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
        advp_all = sorted(set(pd.to_numeric(df_raw.get("ADVP_CANON"), errors="coerce").dropna().astype(int).tolist()))
        advp_sel = st.multiselect("ADVP", options=advp_all,
                                  default=st.session_state["flt"]["advp"], key=f"{key_prefix}_advp")
    with c4:
        trechos_all = sorted([t for t in df_raw.get("TRECHO", pd.Series([], dtype=str)).dropna().unique().tolist() if str(t).strip() != ""])
        tr_sel = st.multiselect("Trechos", options=trechos_all,
                                default=st.session_state["flt"]["trechos"], key=f"{key_prefix}_trechos")
    with c5:
        hh_sel = st.multiselect("Hora da busca", options=list(range(24)),
                                default=st.session_state["flt"]["hh"], key=f"{key_prefix}_hh")
    with c6:
        cia_presentes = set(str(x).upper() for x in df_raw.get("CIA_NORM", pd.Series([], dtype=str)).dropna().unique())
        ordem = ["AZUL", "GOL", "LATAM"]
        cia_opts = [c for c in ordem if (not cia_presentes or c in cia_presentes)] or ordem
        cia_default = [c for c in st.session_state["flt"]["cia"] if c in cia_opts]
        cia_sel = st.multiselect("Cia (Azul/Gol/Latam)", options=cia_opts,
                                 default=cia_default, key=f"{key_prefix}_cia")

    # Botão redondo e discreto, alinhado ao final dos filtros (mesma linha)
    with c7:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)  # alinhar verticalmente
        clicked = st.button("↻", key=f"{key_prefix}_refresh", type="secondary", help="Recarregar base")
        if clicked:
            # limpa cache de dados e força re-run
            st.cache_data.clear()
            st.session_state.pop("flt", None)  # reseta seleção para evitar conflitos de range
            st.rerun()

    st.session_state["flt"] = {
        "dt_ini": dt_ini, "dt_fim": dt_fim, "advp": advp_sel or [],
        "trechos": tr_sel or [], "hh": hh_sel or [], "cia": cia_sel or []
    }

    # Filtro por data (inclusivo)
    base_date_series = pd.to_datetime(df_raw.get("__DTKEY__"), errors="coerce").dt.date
    if base_date_series.isna().all():
        base_date_series = pd.to_datetime(df_raw.get("DATAHORA_BUSCA"), errors="coerce").dt.date

    mask = pd.Series(True, index=df_raw.index)
    mask &= (base_date_series >= dt_ini) & (base_date_series <= dt_fim)
    if advp_sel:
        mask &= df_raw.get("ADVP_CANON").isin(advp_sel)
    if tr_sel:
        mask &= df_raw.get("TRECHO").astype(str).isin([str(x) for x in tr_sel])
    if hh_sel:
        hh_series = df_raw.get("HORA_HH")
        if hh_series is None or (isinstance(hh_series, pd.Series) and hh_series.isna().all()):
            hh_series = pd.to_datetime(df_raw.get("__DTKEY__"), errors="coerce").dt.hour
        mask &= hh_series.isin(hh_sel)
    if st.session_state["flt"]["cia"]:
        mask &= df_raw.get("CIA_NORM").astype(str).str.upper().isin(st.session_state["flt"]["cia"])

    df = df_raw[mask].copy()
    st.caption(f"Linhas após filtros: {fmt_int(len(df))} • Última atualização: {last_update_from_cols(df)}")
    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
    return df

# ─────────────────────────── LÓGICA (painéis/tabelas) ───────────────────────────
def winners_by_position(df: pd.DataFrame) -> pd.DataFrame:
    base = pd.DataFrame({"IDPESQUISA": df["IDPESQUISA"].unique()})
    for r in (1, 2, 3):
        s = (df[df["RANKING"] == r]
             .sort_values(["IDPESQUISA"])
             .drop_duplicates(subset=["IDPESQUISA"]))
        base = base.merge(
            s[["IDPESQUISA","AGENCIA_NORM"]].rename(columns={"AGENCIA_NORM": f"R{r}"}),
            on="IDPESQUISA", how="left"
        )
    for r in (1,2,3): base[f"R{r}"] = base[f"R{r}"].fillna("SEM OFERTAS")
    return base

# ─────────────────────────── ABAS ───────────────────────────
@register_tab("Painel")
def tab1_painel(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t1")
    st.subheader("Painel")
    total_pesq = df["IDPESQUISA"].nunique() or 1
    cov = {r: df.loc[df["RANKING"].eq(r), "IDPESQUISA"].nunique() for r in (1, 2, 3)}
    st.markdown(
        f"<div style='font-size:13px;opacity:.85;margin-top:-6px;'>"
        f"Pesquisas únicas: <b>{fmt_int(total_pesq)}</b> • "
        f"Cobertura 1º: {cov[1]/total_pesq*100:.1f}% • "
        f"2º: {cov[2]/total_pesq*100:.1f}% • "
        f"3º: {cov[3]/total_pesq*100:.1f}%</div>",
        unsafe_allow_html=True
    )

    st.markdown(CARD_CSS, unsafe_allow_html=True)
    st.markdown("<hr style='margin:6px 0'>", unsafe_allow_html=True)

    W = winners_by_position(df)
    Wg = W.replace({
        "R1": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
        "R2": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
        "R3": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
    })

    agencias_all = sorted(set(df["AGENCIA_NORM"].dropna().astype(str)))
    targets_base = list(agencias_all)
    if "GRUPO 123" not in targets_base: targets_base.insert(0, "GRUPO 123")
    if "SEM OFERTAS" not in targets_base: targets_base.append("SEM OFERTAS")

    def pcts_for_target(base_df: pd.DataFrame, tgt: str, agrupado: bool) -> tuple[float,float,float]:
        base = (base_df.replace({
            "R1": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
            "R2": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
            "R3": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
        }) if agrupado else base_df)
        p1 = float((base["R1"] == tgt).mean())*100
        p2 = float((base["R2"] == tgt).mean())*100
        p3 = float((base["R3"] == tgt).mean())*100
        return p1, p2, p3

    targets_sorted = sorted(
        targets_base,
        key=lambda t: pcts_for_target(Wg if t=="GRUPO 123" else W, t, t=="GRUPO 123")[0],
        reverse=True
    )
    cards = []
    for idx, tgt in enumerate(targets_sorted):
        p1, p2, p3 = pcts_for_target(Wg if tgt=="GRUPO 123" else W, tgt, tgt=="GRUPO 123")
        rank_cls = "goldcard" if idx == 0 else "silvercard" if idx == 1 else "bronzecard" if idx == 2 else ""
        cards.append(card_html(tgt, p1, p2, p3, rank_cls))
    st.markdown(f"<div class='cards-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)

    st.markdown("<hr style='margin:14px 0 8px 0'>", unsafe_allow_html=True)
    st.subheader("Painel por Cia")
    st.caption("Cada coluna mostra o ranking das agências para a CIA correspondente (cards um abaixo do outro).")
    st.markdown(CARDS_STACK_CSS, unsafe_allow_html=True)

    if "CIA_NORM" not in df.columns:
        st.info("Coluna 'CIA_NORM' não encontrada nos dados filtrados."); return

    c1, c2, c3 = st.columns(3)
    def render_por_cia(container, df_in: pd.DataFrame, cia_name: str):
        with container:
            st.markdown(f"<div class='stack-title'>Ranking {cia_name.title()}</div>", unsafe_allow_html=True)
            sub = df_in[df_in["CIA_NORM"].astype(str).str.upper() == cia_name]
            if sub.empty: st.info("Sem dados para os filtros atuais."); return
            Wc = winners_by_position(sub)
            Wc_g = Wc.replace({
                "R1": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
                "R2": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
                "R3": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
            })
            ags = sorted(set(sub["AGENCIA_NORM"].dropna().astype(str)))
            targets = [a for a in ags if a != "SEM OFERTAS"]
            if "GRUPO 123" not in targets: targets.insert(0, "GRUPO 123")
            def pct_target(tgt: str):
                base = Wc_g if tgt == "GRUPO 123" else Wc
                p1 = float((base["R1"] == tgt).mean())*100
                p2 = float((base["R2"] == tgt).mean())*100
                p3 = float((base["R3"] == tgt).mean())*100
                return p1, p2, p3
            targets_sorted_local = sorted(targets, key=lambda t: pct_target(t)[0], reverse=True)
            cards_local = [card_html(t, *pct_target(t)) for t in targets_sorted_local]
            st.markdown(f"<div class='cards-stack'>{''.join(cards_local)}</div>", unsafe_allow_html=True)
    render_por_cia(c1, df, "AZUL"); render_por_cia(c2, df, "GOL"); render_por_cia(c3, df, "LATAM")

@register_tab("Top 3 Agências")
def tab2_top3_agencias(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t2")
    st.subheader("Top 3 Agências (por menor preço no trecho)")
    if df.empty:
        st.info("Sem resultados para os filtros selecionados.")
        return

    # 1) Última pesquisa por trecho
    df2 = df.copy()
    df2["DT"] = pd.to_datetime(df2["DATAHORA_BUSCA"], errors="coerce")
    g = (df2.dropna(subset=["TRECHO","IDPESQUISA","DT"])
              .groupby(["TRECHO","IDPESQUISA"], as_index=False)["DT"].max())
    last_idx = g.groupby("TRECHO")["DT"].idxmax()
    last_ids = {r["TRECHO"]: r["IDPESQUISA"] for _, r in g.loc[last_idx].iterrows()}
    df2["__ID_TARGET__"] = df2["TRECHO"].map(last_ids)
    df_last = df2[df2["IDPESQUISA"].astype(str) == df2["__ID_TARGET__"].astype(str)].copy()

    # 2) Data/Hora
    def _compose_dt_hora(sub: pd.DataFrame) -> str:
        d = pd.to_datetime(sub["DATAHORA_BUSCA"], errors="coerce").max()
        hh = None
        for v in sub["HORA_BUSCA"].tolist():
            hh = _norm_hhmmss(v)
            if hh: break
        if pd.isna(d) and not hh:
            return "-"
        if not hh:
            hh = pd.to_datetime(d, errors="coerce").strftime("%H:%M:%S")
        return f"{d.strftime('%d/%m/%Y')} {hh}"
    dt_by_trecho = {trecho: _compose_dt_hora(sub) for trecho, sub in df_last.groupby("TRECHO")}

    # 3) Ranking Top-3 por trecho
    PRICE_COL, TRECHO_COL, AGENCIA_COL = "PRECO", "TRECHO", "AGENCIA_NORM"
    by_ag = (
        df_last.groupby([TRECHO_COL, AGENCIA_COL], as_index=False)
               .agg(PRECO_MIN=(PRICE_COL, "min"))
               .rename(columns={TRECHO_COL:"TRECHO_STD", AGENCIA_COL:"AGENCIA_UP"})
    )
    def _row_top3(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("PRECO_MIN", ascending=True).reset_index(drop=True)
        trecho = g["TRECHO_STD"].iloc[0] if len(g) else "-"
        def name(i):  return g.loc[i, "AGENCIA_UP"] if i < len(g) else "-"
        def price(i): return g.loc[i, "PRECO_MIN"]  if i < len(g) else np.nan
        return pd.Series({
            "Data/Hora Busca": dt_by_trecho.get(trecho, "-"),
            "Trecho": trecho,
            "Agencia Top 1": name(0), "Preço Top 1": price(0),
            "Agencia Top 2": name(1), "Preço Top 2": price(1),
            "Agencia Top 3": name(2), "Preço Top 3": price(2),
        })
    t1 = by_ag.groupby("TRECHO_STD").apply(_row_top3).reset_index(drop=True)
    for c in ["Preço Top 1","Preço Top 2","Preço Top 3"]:
        t1[c] = pd.to_numeric(t1[c], errors="coerce")
    sty1 = style_smart_colwise(t1, {c: fmt_num0_br for c in ["Preço Top 1","Preço Top 2","Preço Top 3"]},
                               grad_cols=["Preço Top 1","Preço Top 2","Preço Top 3"])
    show_table(t1, sty1, caption="Ranking Top 3 (Agências) — por trecho")

    # 4) % Dif vs Top1
    def pct_diff(base, other):
        if pd.isna(base) or base == 0 or pd.isna(other): return np.nan
        return (other - base) / base * 100
    rows2 = []
    for _, r in t1.iterrows():
        basep = r["Preço Top 1"]
        rows2.append({
            "Data/Hora Busca": r["Data/Hora Busca"],
            "Trecho": r["Trecho"],
            "Agencia Top 1": r["Agencia Top 1"], "Preço Top 1": basep,
            "Agencia Top 2": r["Agencia Top 2"], "% Dif Top2 vs Top1": pct_diff(basep, r["Preço Top 2"]),
            "Agencia Top 3": r["Agencia Top 3"], "% Dif Top3 vs Top1": pct_diff(basep, r["Preço Top 3"]),
        })
    t2 = pd.DataFrame(rows2).reset_index(drop=True)
    sty2 = style_smart_colwise(
        t2,
        {"Preço Top 1": fmt_num0_br, "% Dif Top2 vs Top1": fmt_pct2_br, "% Dif Top3 vs Top1": fmt_pct2_br},
        grad_cols=["Preço Top 1", "% Dif Top2 vs Top1", "% Dif Top3 vs Top1"]
    )
    show_table(t2, sty2, caption="% Diferença entre Agências (base: TOP1)")

    # 5) Comparativo Cia × Milhas
    A_123, A_MAX, A_FLIP, A_CAPO = "123MILHAS", "MAXMILHAS", "FLIPMILHAS", "CAPOVIAGENS"
    by_air = (df_last.groupby(["TRECHO","CIA_NORM"], as_index=False)
                    .agg(PRECO_AIR_MIN=("PRECO","min"))
                    .rename(columns={"TRECHO":"TRECHO_STD","CIA_NORM":"Cia Menor Preço"}))
    idx = by_air.groupby("TRECHO_STD")["PRECO_AIR_MIN"].idxmin()
    base_min = by_air.loc[idx, ["TRECHO_STD","Cia Menor Preço","PRECO_AIR_MIN"]] \
                     .rename(columns={"PRECO_AIR_MIN":"Preço Menor Valor"})
    def _best_price(sub: pd.DataFrame, ag: str) -> float:
        m = sub[sub["AGENCIA_NORM"] == ag]
        return float(m["PRECO"].min()) if not m.empty else np.nan
    rows3 = []
    for trecho, sub in df_last.groupby("TRECHO"):
        rows3.append({
            "Data/Hora Busca": dt_by_trecho.get(trecho, "-"),
            "Trecho": trecho,
            "Cia Menor Preço": base_min.loc[base_min["TRECHO_STD"].eq(trecho), "Cia Menor Preço"].squeeze() if (base_min["TRECHO_STD"]==trecho).any() else "-",
            "Preço Menor Valor": base_min.loc[base_min["TRECHO_STD"].eq(trecho), "Preço Menor Valor"].squeeze() if (base_min["TRECHO_STD"]==trecho).any() else np.nan,
            "123milhas": _best_price(sub, A_123),
            "Maxmilhas": _best_price(sub, A_MAX),
            "FlipMilhas": _best_price(sub, A_FLIP),
            "Capo Viagens": _best_price(sub, A_CAPO),
        })
    t3 = pd.DataFrame(rows3).reset_index(drop=True)
    fmt3 = {c: fmt_num0_br for c in ["Preço Menor Valor","123milhas","Maxmilhas","FlipMilhas","Capo Viagens"]}
    sty3 = style_smart_colwise(t3, fmt3, grad_cols=list(fmt3.keys()))
    show_table(t3, sty3, caption="Comparativo Menor Preço Cia × Agências de Milhas")

    # 6) % Dif. vs menor valor por Cia
    def pct_vs_base(b, x):
        if pd.isna(b) or b == 0 or pd.isna(x): return np.nan
        return (x - b) / b * 100
    t4 = t3[["Data/Hora Busca","Trecho","Cia Menor Preço","Preço Menor Valor"]].copy()
    for label in ["123milhas","Maxmilhas","FlipMilhas","Capo Viagens"]:
        t4[f"% Dif {label}"] = [pct_vs_base(b, x) for b, x in zip(t3["Preço Menor Valor"], t3[label])]
    fmt4 = {"Preço Menor Valor": fmt_num0_br} | {c: fmt_pct2_br for c in t4.columns if c.startswith("% Dif ")}
    sty4 = style_smart_colwise(t4.reset_index(drop=True), fmt4, grad_cols=list(fmt4.keys()))
    show_table(t4, sty4, caption="%Comparativo Menor Preço Cia × Agências de Milhas")

@register_tab("Top 3 Preços Mais Baratos")
def tab3_top3_precos(df_raw: pd.DataFrame):
    import re
    df = render_filters(df_raw, key_prefix="t3")
    st.subheader("Top 3 Preços Mais Baratos")

    top_row = st.container()
    with top_row:
        c1, c2, _ = st.columns([0.28, 0.18, 0.54])
        agencia_foco = c1.selectbox("Agência alvo", ["Todos", "123MILHAS", "MAXMILHAS"], index=0)
        posicao_foco = c2.selectbox("Ranking", ["Todas", 1, 2, 3], index=0)

    if df.empty:
        st.info("Sem resultados para os filtros selecionados."); return

    def fmt_moeda_br(x) -> str:
        try:
            xv = float(x)
            if not np.isfinite(xv): return "R$ -"
            return "R$ " + f"{xv:,.0f}".replace(",", ".")
        except Exception:
            return "R$ -"

    def _find_id_col(df_: pd.DataFrame) -> str | None:
        cands = ["IDPESQUISA","ID_PESQUISA","ID BUSCA","IDBUSCA","ID","NOME_ARQUIVO_STD","NOME_ARQUIVO","NOME DO ARQUIVO","ARQUIVO"]
        norm = { re.sub(r"[^A-Z0-9]+","", c.upper()): c for c in df_.columns }
        for nm in cands:
            key = re.sub(r"[^A-Z0-9]+","", nm.upper())
            if key in norm: return norm[key]
        return df_.columns[0] if len(df_.columns) else None

    GRID_STYLE    = "display:grid;grid-auto-flow:column;grid-auto-columns:260px;gap:10px;overflow-x:auto;padding:6px 2px 10px 2px;scrollbar-width:thin;"
    BOX_STYLE     = "border:1px solid #e5e7eb;border-radius:12px;background:#fff;"
    HEAD_STYLE    = "padding:8px 10px;border-bottom:1px solid #f1f5f9;font-weight:700;color:#111827;"
    STACK_STYLE   = "display:grid;gap:8px;padding:8px;"
    CARD_BASE     = "position:relative;border:1px solid #e5e7eb;border-radius:10px;padding:8px;background:#fff;min-height:78px;"
    DT_WRAP_STYLE = "position:absolute;right:8px;top:6px;display:flex;align-items:center;gap:6px;"
    DT_TXT_STYLE  = "font-size:10px;color:#94a3b8;font-weight:800;"
    RANK_STYLE    = "font-weight:900;font-size:11px;color:#6b7280;letter-spacing:.3px;text-transform:uppercase;"
    AG_STYLE      = "font-weight:800;font-size:15px;color:#111827;margin-top:2px;"
    PR_STYLE      = "font-weight:900;font-size:18px;color:#111827;margin-top:2px;"
    SUB_STYLE     = "font-weight:700;font-size:12px;color:#374151;"
    NO_STYLE      = "padding:22px 12px;color:#6b7280;font-weight:800;text-align:center;border:1px dashed #e5e7eb;border-radius:10px;background:#fafafa;"
    TRE_HDR_STYLE = "margin:14px 0 10px 0;padding:10px 12px;border-left:4px solid #0B5FFF;background:#ECF3FF;border-radius:8px;font-weight:800;color:#0A2A6B;"

    BADGE_POP_CSS = """
    <style>
    .idp-wrap{position:relative; display:inline-flex; align-items:center;}
    .idp-badge{display:inline-flex; align-items:center; justify-content:center;width:16px; height:16px; border:1px solid #cbd5e1; border-radius:50%;font-size:11px; font-weight:900; color:#64748b; background:#fff;user-select:none; cursor:default; line-height:1;}
    .idp-pop{position:absolute; top:18px; right:0;background:#fff; color:#0f172a; border:1px solid #e5e7eb;border-radius:8px; padding:6px 8px; font-size:12px; font-weight:700;box-shadow:0 6px 16px rgba(0,0,0,.08); display:none; z-index:9999; white-space:nowrap;}
    .idp-wrap:hover .idp-pop{ display:block; }
    .idp-idbox{border:1px solid #e5e7eb; background:#f8fafc; border-radius:6px;padding:2px 6px; font-weight:800; font-size:12px; min-width:60px;font-family: ui-monospace;user-select:text; cursor:text;}
    .tag-extra{font-size:12px; color:#4b5563; margin-top:2px;}
    .extra-wrap{padding:6px 8px 10px 8px; border-top:1px dashed #e5e7eb; margin-top:6px;}
    .extra-title{font-size:12px; font-weight:700; color:#374151; margin-bottom:2px;}
    </style>
    """
    st.markdown(BADGE_POP_CSS, unsafe_allow_html=True)

    dfp = df.copy()
    dfp["TRECHO_STD"] = dfp.get("TRECHO", "").astype(str)
    dfp["AGENCIA_UP"] = dfp.get("AGENCIA_NORM", "").astype(str)
    dfp["ADVP"]       = (dfp.get("ADVP_CANON").fillna(dfp.get("ADVP"))).astype(str)
    dfp["__PRECO__"]  = pd.to_numeric(dfp.get("PRECO"), errors="coerce")
    dfp["__DTKEY__"]  = pd.to_datetime(dfp.get("DATAHORA_BUSCA"), errors="coerce")

    def _normalize_id(val):
        if val is None or (isinstance(val, float) and np.isnan(val)): return None
        s = str(val)
        try:
            f = float(s.replace(",", "."))
            if f.is_integer(): return str(int(f))
        except Exception:
            pass
        return s

    def dt_and_id_for(sub_rows: pd.DataFrame) -> tuple[str, str | None]:
        if sub_rows.empty: return "", None
        r = sub_rows.loc[sub_rows["__DTKEY__"].idxmax()]
        date_part = pd.to_datetime(r["DATAHORA_BUSCA"], errors="coerce")
        date_txt  = date_part.strftime("%d/%m") if pd.notna(date_part) else ""
        htxt_raw  = str(r.get("HORA_BUSCA","")).strip()
        htxt = _norm_hhmmss(htxt_raw) or (pd.to_datetime(r["__DTKEY__"], errors="coerce").strftime("%H:%M:%S") if pd.notna(r["__DTKEY__"]) else "")
        id_val = _normalize_id(r.get("IDPESQUISA"))
        lbl = f"{date_txt} {htxt}".strip()
        return lbl, id_val

    ID_COL = "IDPESQUISA"
    dfp = dfp[dfp["__PRECO__"].notna()].copy()
    if dfp.empty: st.info("Sem preços válidos no recorte atual."); return

    # última pesquisa por Trecho×ADVP
    pesq_por_ta = {}
    tmp = dfp.dropna(subset=["TRECHO_STD","ADVP",ID_COL,"__DTKEY__"]).copy()
    g = tmp.groupby(["TRECHO_STD","ADVP",ID_COL], as_index=False)["__DTKEY__"].max()
    if not g.empty:
        idx = g.groupby(["TRECHO_STD","ADVP"])["__DTKEY__"].idxmax()
        last_by_ta = g.loc[idx]
        pesq_por_ta = {(str(r["TRECHO_STD"]), str(r["ADVP"])): str(r[ID_COL]) for _, r in last_by_ta.iterrows()}

    trechos_sorted = sorted(dfp["TRECHO_STD"].dropna().astype(str).unique(), key=lambda x: str(x))
    for trecho in trechos_sorted:
        df_t = dfp[dfp["TRECHO_STD"] == trecho]
        advps = sorted(df_t["ADVP"].dropna().astype(str).unique(),
                       key=lambda v: (0, int("".join([d for d in v if d.isdigit()]) or 9999), str(v)))

        boxes = []
        for advp in advps:
            df_ta = df_t[df_t["ADVP"].astype(str) == str(advp)].copy()
            pesq_id = pesq_por_ta.get((trecho, advp))
            if pesq_id:
                all_rows = df_ta[df_ta[ID_COL].astype(str) == pesq_id]
            else:
                all_rows = df_ta.iloc[0:0]

            base_rank = (all_rows.groupby("AGENCIA_UP", as_index=False)["__PRECO__"]
                                   .min().sort_values("__PRECO__").reset_index(drop=True))

            GRID_STYLE    = "display:grid;grid-auto-flow:column;grid-auto-columns:260px;gap:10px;overflow-x:auto;padding:6px 2px 10px 2px;scrollbar-width:thin;"
            BOX_STYLE     = "border:1px solid #e5e7eb;border-radius:12px;background:#fff;"
            HEAD_STYLE    = "padding:8px 10px;border-bottom:1px solid #f1f5f9;font-weight:700;color:#111827;"
            STACK_STYLE   = "display:grid;gap:8px;padding:8px;"
            CARD_BASE     = "position:relative;border:1px solid #e5e7eb;border-radius:10px;padding:8px;background:#fff;min-height:78px;"
            DT_WRAP_STYLE = "position:absolute;right:8px;top:6px;display:flex;align-items:center;gap:6px;"
            DT_TXT_STYLE  = "font-size:10px;color:#94a3b8;font-weight:800;"
            RANK_STYLE    = "font-weight:900;font-size:11px;color:#6b7280;letter-spacing:.3px;text-transform:uppercase;"
            AG_STYLE      = "font-weight:800;font-size:15px;color:#111827;margin-top:2px;"
            PR_STYLE      = "font-weight:900;font-size:18px;color:#111827;margin-top:2px;"
            SUB_STYLE     = "font-weight:700;font-size:12px;color:#374151;"
            NO_STYLE      = "padding:22px 12px;color:#6b7280;font-weight:800;text-align:center;border:1px dashed #e5e7eb;border-radius:10px;background:#fafafa;"

            box_content = []
            box_content.append(f"<div style='{BOX_STYLE}'>")
            box_content.append(f"<div style='{HEAD_STYLE}'>ADVP: <b>{advp}</b></div>")

            if base_rank.empty:
                box_content.append(f"<div style='{NO_STYLE}'>Sem ofertas</div>")
                box_content.append("</div>"); boxes.append("".join(box_content)); continue

            # Top3 cards
            box_content.append(f"<div style='{STACK_STYLE}'>")
            for i in range(min(3, len(base_rank))):
                row_i = base_rank.iloc[i]
                preco_i = float(row_i["__PRECO__"])
                sub_rows = all_rows[(all_rows["AGENCIA_UP"] == row_i["AGENCIA_UP"]) & (np.isclose(all_rows["__PRECO__"], preco_i, atol=1))]
                dt_lbl, id_val = dt_and_id_for(sub_rows)

                if i == 0:
                    subtxt = "—"
                    if len(base_rank) >= 2:
                        p1 = preco_i
                        p2 = float(base_rank.iloc[1]["__PRECO__"])
                        if np.isfinite(p2) and p2 != 0:
                            pct_below = int(round((p2 - p1) / p2 * 100))
                            subtxt = f"-{pct_below}% vs 2º"
                else:
                    p1 = float(base_rank.iloc[0]["__PRECO__"])
                    subtxt = "—"
                    if np.isfinite(p1) and p1 != 0:
                        subtxt = f"+{int(round((preco_i - p1)/p1*100))}% vs 1º"

                stripe = "#D4AF37" if i==0 else "#9CA3AF" if i==1 else "#CD7F32"
                box_content.append(
                    f"<div style='{CARD_BASE}'>"
                    f"<div style='position:absolute;left:0;top:0;bottom:0;width:6px;border-radius:10px 0 0 10px;background:{stripe};'></div>"
                    f"<div style='{DT_WRAP_STYLE}'><span style='{DT_TXT_STYLE}'>{dt_lbl}</span>"
                    f"</div>"
                    f"<div style='{RANK_STYLE}'>{i+1}º</div>"
                    f"<div style='{AG_STYLE}'>{row_i['AGENCIA_UP']}</div>"
                    f"<div style='{PR_STYLE}'>{fmt_moeda_br(preco_i)}</div>"
                    f"<div style='{SUB_STYLE}'>{subtxt}</div>"
                    f"</div>"
                )
            box_content.append("</div>")
            box_content.append("</div>")
            boxes.append("".join(box_content))

        if boxes:
            st.markdown(f"<div style='{TRE_HDR_STYLE}'>Trecho: <b>{trecho}</b></div>", unsafe_allow_html=True)
            st.markdown("<div style='" + GRID_STYLE + "'>" + "".join(boxes) + "</div>", unsafe_allow_html=True)

@register_tab("Ranking por Agências")
def tab4_ranking_agencias(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t4")
    if df.empty:
        st.subheader("Ranking por Agências")
        st.info("Sem dados para os filtros.")
        return

    work = df.copy()
    work["AGENCIA_UP"] = work.get("AGENCIA_UP", work.get("AGENCIA_NORM", work.get("AGENCIA_COMP", work.get("AGENCIA", "")))).astype(str)
    if "RANKING" not in work.columns:
        st.warning("Coluna 'RANKING' não encontrada."); return
    work["Ranking"] = pd.to_numeric(work["RANKING"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["AGENCIA_UP", "Ranking"])
    work["Ranking"] = work["Ranking"].astype(int)

    def _fmt_pct(v):
        try: x = float(v); return "-" if not np.isfinite(x) else f"{x:.2f}%".replace(".", ",")
        except Exception: return "-"

    def style_heatmap(df_show: pd.DataFrame, percent_cols=None, int_cols=None,
                      highlight_total_row=False, highlight_total_col=None,
                      highlight_rows_map=None, height=440):
        percent_cols = set(percent_cols or [])
        num_cols = [c for c in df_show.columns if pd.api.types.is_numeric_dtype(df_show[c])]
        fmt_map = {c: _fmt_pct for c in percent_cols}
        sty = (df_show.style.format(fmt_map, na_rep="-"))
        if num_cols:
            try:
                sty = sty.background_gradient(cmap="Blues", subset=num_cols)
            except Exception:
                pass
        if highlight_total_col and highlight_total_col in df_show.columns:
            def _hl_total_col(col):
                return ["background-color:#E6F0FF; font-weight:bold; color:#0A2A6B;" for _ in col]
            sty = sty.apply(_hl_total_col, subset=[highlight_total_col])
        if highlight_rows_map:
            first_col = df_show.columns[0]
            def _hl_special_rows(row):
                key = str(row[first_col]).upper()
                if key in highlight_rows_map:
                    color = highlight_rows_map[key]
                    return [f"background-color:{color}; color:#0A2A6B; font-weight:bold;" for _ in row]
                return ["" for _ in row]
            sty = sty.apply(_hl_special_rows, axis=1)
        if highlight_total_row and len(df_show) > 0:
            def _hl_last_row(row):
                return ["background-color:#E6F0FF; color:#0A2A6B; font-weight:bold;"
                        if row.name == df_show.index.max() else "" for _ in row]
            sty = sty.apply(_hl_last_row, axis=1)
        return sty

    def show_table_rank(df_in: pd.DataFrame, percent_cols=None, highlight_total_row=False,
                        highlight_total_col=None, highlight_rows_map=None, height=440):
        df_disp = df_in.reset_index(drop=True)
        df_disp.index = np.arange(1, len(df_disp) + 1)
        st.dataframe(
            style_heatmap(df_disp, percent_cols=percent_cols, highlight_total_row=highlight_total_row,
                          highlight_total_col=highlight_total_col, highlight_rows_map=highlight_rows_map, height=height),
            use_container_width=True, height=height
        )

    RANKS = list(range(1, 16))
    counts = (work.groupby(["AGENCIA_UP", "Ranking"], as_index=False)
                   .agg(OFERTAS=("AGENCIA_UP", "size")))
    pv = (counts.pivot(index="AGENCIA_UP", columns="Ranking", values="OFERTAS")
                 .reindex(columns=RANKS, fill_value=0)
                 .fillna(0).astype(int))
    pv.index.name = "Agência/Companhia"
    if 1 not in pv.columns:
        pv[1] = 0
    pv["Total"] = pv.sum(axis=1)
    pv = pv.sort_values(by=1, ascending=False)

    total_row = pv.sum(axis=0).to_frame().T
    total_row.index = ["Total"]
    total_row.index.name = "Agência/Companhia"
    pv2 = pd.concat([pv, total_row], axis=0)

    HL_MAP = {"123MILHAS": "#FFD8A8", "MAXMILHAS": "#D3F9D8"}

    st.subheader("Quantidade de Ofertas por Ranking (Ofertas)")
    t_qtd = pv2.reset_index()
    if t_qtd.columns[0] != "Agência/Companhia":
        t_qtd = t_qtd.rename(columns={t_qtd.columns[0]: "Agência/Companhia"})
    show_table_rank(
        t_qtd[["Agência/Companhia"] + RANKS + ["Total"]],
        highlight_total_row=True, highlight_total_col="Total",
        highlight_rows_map=HL_MAP, height=480
    )

    mat = pv[RANKS].copy()
    row_sum = mat.sum(axis=1).replace(0, np.nan)
    pct_linha = (mat.div(row_sum, axis=0) * 100).fillna(0)
    pct_linha = pct_linha.sort_values(by=1, ascending=False)
    t_pct_linha = pct_linha.reset_index()
    if t_pct_linha.columns[0] != "Agência/Companhia":
        t_pct_linha = t_pct_linha.rename(columns={t_pct_linha.columns[0]: "Agência/Companhia"})
    st.subheader("Participação Ranking dentro da Agência")
    show_table_rank(t_pct_linha[["Agência/Companhia"] + RANKS],
                    percent_cols=set(RANKS),
                    highlight_rows_map=HL_MAP, height=440)

    col_sum = mat.sum(axis=0).replace(0, np.nan)
    pct_coluna = (mat.div(col_sum, axis=1) * 100).fillna(0)
    pct_coluna = pct_coluna.sort_values(by=1, ascending=False)
    t_pct_coluna = pct_coluna.reset_index()
    if t_pct_coluna.columns[0] != "Agência/Companhia":
        t_pct_coluna = t_pct_coluna.rename(columns={t_pct_coluna.columns[0]: "Agência/Companhia"})
    st.subheader("Participação Ranking Geral")
    show_table_rank(t_pct_coluna[["Agência/Companhia"] + RANKS],
                    percent_cols=set(RANKS),
                    highlight_rows_map=HL_MAP, height=440)

@register_tab("Competitividade Cia x Trecho")
def tab5_competitividade(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t5")
    st.subheader("Competitividade Cia × Trecho")
    if df.empty:
        st.info("Sem resultados para os filtros atuais."); return
    need = {"RANKING","CIA_NORM","TRECHO","AGENCIA_NORM","IDPESQUISA"}
    if not need.issubset(df.columns):
        st.warning(f"Colunas ausentes: {sorted(list(need - set(df.columns)))}"); return

    # Vencedores
    df1 = df[df["RANKING"].astype("Int64") == 1].copy()
    if df1.empty:
        st.info("Nenhum 1º lugar no recorte."); return
    df1["CIA_UP"]     = df1["CIA_NORM"].astype(str).str.upper()
    df1["TRECHO_STD"] = df1["TRECHO"].astype(str)
    df1["AG_UP"]      = df1["AGENCIA_NORM"].astype(str)

    # Top 11 trechos por nº de pesquisas
    top_trechos = (df1.groupby("TRECHO_STD")["IDPESQUISA"]
                     .nunique().sort_values(ascending=False).head(11).index.tolist())
    if len(top_trechos) < 11:
        restantes = [t for t in df1["TRECHO_STD"].unique().tolist() if t not in top_trechos]
        top_trechos += restantes[: max(0, 11 - len(top_trechos))]

    # Totais e vitórias por Cia×Trecho
    tot = (df1.groupby(["CIA_UP","TRECHO_STD"])["IDPESQUISA"]
              .nunique().reset_index(name="TotalPesq"))
    win = (df1.groupby(["CIA_UP","TRECHO_STD","AG_UP"])["IDPESQUISA"]
              .nunique().reset_index(name="QtdTop1"))
    base = win.merge(tot, on=["CIA_UP","TRECHO_STD"], how="right")
    base["QtdTop1"] = base["QtdTop1"].fillna(0)
    base["Pct"] = (base["QtdTop1"] / base["TotalPesq"].replace(0, np.nan) * 100).fillna(0.0)

    def _tot_for(cia: str, trecho: str) -> int:
        m = tot.loc[(tot["CIA_UP"] == cia) & (tot["TRECHO_STD"] == trecho), "TotalPesq"]
        if m.empty: return 0
        v = m.iloc[0]
        try: return int(v) if pd.notna(v) else 0
        except: 
            try: return int(float(v))
            except: return 0

    def lideres_por_cia(cia: str) -> pd.DataFrame:
        sub_all = base[base["CIA_UP"] == cia].copy()
        if sub_all.empty:
            rows = [{"TRECHO": t, "AGENCIA": "SEM OFERTAS", "PCT": 0.0, "N": _tot_for(cia, t)} for t in top_trechos]
            return pd.DataFrame(rows).sort_values("PCT", ascending=False, kind="mergesort").reset_index(drop=True)
        sub_all = sub_all.sort_values(["TRECHO_STD", "Pct"], ascending=[True, False])
        idx = sub_all.groupby("TRECHO_STD")["Pct"].idxmax()
        lideres = sub_all.loc[idx, ["TRECHO_STD", "AG_UP", "Pct", "TotalPesq"]].set_index("TRECHO_STD")
        rows = []
        for t in top_trechos:
            if t in lideres.index:
                ag  = str(lideres.loc[t, "AG_UP"]) if pd.notna(lideres.loc[t, "AG_UP"]) else "SEM OFERTAS"
                pct = float(lideres.loc[t, "Pct"])  if pd.notna(lideres.loc[t, "Pct"])  else 0.0
                n   = int(lideres.loc[t, "TotalPesq"]) if pd.notna(lideres.loc[t, "TotalPesq"]) else _tot_for(cia, t)
            else:
                ag, pct, n = "SEM OFERTAS", 0.0, _tot_for(cia, t)
            rows.append({"TRECHO": t, "AGENCIA": ag, "PCT": pct, "N": n})
        return pd.DataFrame(rows).sort_values("PCT", ascending=False, kind="mergesort").reset_index(drop=True)

    cia_colors = {"AZUL":("#2D6CDF","#FFFFFF"), "GOL":("#E67E22","#FFFFFF"), "LATAM":("#C0392B","#FFFFFF")}
    st.markdown("""
    <style>
      .comp-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;}
      @media (max-width:1100px){.comp-grid{grid-template-columns:repeat(2,minmax(0,1fr));}}
      @media (max-width:700px){.comp-grid{grid-template-columns:1fr;}}
      .comp-card{border:1px solid #e5e7eb;border-radius:12px;background:#fff;overflow:hidden;box-shadow:0 1px 2px rgba(0,0,0,.06);}
      .comp-head{padding:10px 12px;font-weight:900;letter-spacing:.2px;}
      .tbl,.tblc{width:100%;border-collapse:collapse;}
      .tbl th,.tbl td,.tblc th,.tblc td{border:1px solid #e5e7eb;padding:6px 8px;font-size:13px; text-align:center;}
      .tbl td:nth-child(3){font-weight:800;}
      .muted{font-size:10px;color:#94a3b8;font-weight:800;display:block;line-height:1;margin-top:2px;}
      .row0{background:#ffffff;}
      .row1{background:#fcfcfc;}
      .g123{color:#0B6B2B;font-weight:900;}
      .sec-sep{height:12px;}
    </style>
    """, unsafe_allow_html=True)

    def fmt_pct(v: float) -> str:
        try: return f"{float(v):.2f}%".replace(".", ",")
        except: return "0,00%"

    def quadro_trecho_html(cia: str, dfq: pd.DataFrame) -> str:
        bg, fg = cia_colors.get(cia, ("#0B5FFF","#FFFFFF"))
        head = f"<div class='comp-head' style='background:{bg};color:{fg};'>{cia}</div>"
        if dfq.empty:
            body = "<div style='padding:14px;color:#64748b;font-weight:700;text-align:center;'>Sem dados</div>"
            return f"<div class='comp-card'>{head}{body}</div>"
        rows = ["<table class='tbl'><thead><tr><th>TRECHO</th><th>AGENCIA</th><th>% DE GANHO</th></tr></thead><tbody>"]
        for i, r in dfq.iterrows():
            alt = "row1" if i % 2 else "row0"
            ag  = str(r["AGENCIA"])
            ag_cls = "g123" if ag in {"MAXMILHAS","123MILHAS"} else ""
            pct_txt = fmt_pct(r["PCT"])
            n_txt = f"<span class='muted'>{int(r['N'])} pesq.</span>"
            rows.append(
                f"<tr class='{alt}'>"
                f"<td>{r['TRECHO']}</td>"
                f"<td><span class='{ag_cls}'>{ag}</span></td>"
                f"<td><div>{pct_txt}{n_txt}</div></td>"
                f"</tr>"
            )
        rows.append("</tbody></table>")
        return f"<div class='comp-card'>{head}{''.join(rows)}</div>"

    items = [quadro_trecho_html(cia, lideres_por_cia(cia)) for cia in ["AZUL","GOL","LATAM"]]
    st.markdown("<div class='comp-grid'>" + "".join(items) + "</div>", unsafe_allow_html=True)

    # ───── Seção: Competitividade x ADVP ─────
    st.markdown("<div class='sec-sep'></div>", unsafe_allow_html=True)
    st.subheader("Competitividade x ADVP")

    advp_series = pd.to_numeric(df1.get("ADVP_CANON"), errors="coerce")
    df1_advp = df1[advp_series.notna()].copy()
    if df1_advp.empty:
        st.info("Sem ADVPs válidos para os filtros atuais."); return
    df1_advp["ADVP_BKT"] = advp_series.loc[df1_advp.index].astype(int)

    totA = (df1_advp.groupby(["CIA_UP","ADVP_BKT"])["IDPESQUISA"]
              .nunique().reset_index(name="TotAdvp"))
    winA = (df1_advp.groupby(["CIA_UP","ADVP_BKT","AG_UP"])["IDPESQUISA"]
              .nunique().reset_index(name="QtdTop1"))
    baseA = winA.merge(totA, on=["CIA_UP","ADVP_BKT"], how="right").fillna({"QtdTop1":0})
    baseA["Pct"] = (baseA["QtdTop1"] / baseA["TotAdvp"].replace(0, np.nan) * 100).fillna(0.0)

    advp_buckets = [1, 5, 11, 17, 30]
    def lideres_por_cia_advp(cia: str) -> pd.DataFrame:
        sub_all = baseA[baseA["CIA_UP"] == cia].copy()
        rows = []
        if not sub_all.empty:
            sub_all = sub_all.sort_values(["ADVP_BKT","Pct"], ascending=[True, False])
            idx = sub_all.groupby("ADVP_BKT")["Pct"].idxmax()
            lid = sub_all.loc[idx, ["ADVP_BKT","AG_UP","Pct","TotAdvp"]].set_index("ADVP_BKT")
        else:
            lid = pd.DataFrame(columns=["AG_UP","Pct","TotAdvp"])
        for a in advp_buckets:
            if a in lid.index:
                ag  = str(lid.loc[a, "AG_UP"]) if pd.notna(lid.loc[a, "AG_UP"]) else "SEM OFERTAS"
                pct = float(lid.loc[a, "Pct"])  if pd.notna(lid.loc[a, "Pct"])  else 0.0
                n   = int(lid.loc[a, "TotAdvp"]) if pd.notna(lid.loc[a, "TotAdvp"]) else 0
            else:
                ag, pct, n = "SEM OFERTAS", 0.0, 0
            rows.append({"ADVP": a, "AGENCIA": ag, "PCT": pct, "N": n})
        return pd.DataFrame(rows)

    def quadro_advp_html(cia: str, dfq: pd.DataFrame) -> str:
        bg, fg = cia_colors.get(cia, ("#0B5FFF","#FFFFFF"))
        head = f"<div class='comp-head' style='background:{bg};color:{fg};'>{cia}</div>"
        if dfq.empty:
            body = "<div style='padding:14px;color:#64748b;font-weight:700;text-align:center;'>Sem dados</div>"
            return f"<div class='comp-card'>{head}{body}</div>"
        rows = ["<table class='tblc'><thead><tr><th>ADVP</th><th>AGENCIA</th><th>% DE GANHO</th><th>PESQ.</th></tr></thead><tbody>"]
        for i, r in dfq.iterrows():
            alt = "row1" if i % 2 else "row0"
            ag  = str(r["AGENCIA"])
            ag_cls = "g123" if ag in {"MAXMILHAS","123MILHAS"} else ""
            rows.append(
                f"<tr class='{alt}'>"
                f"<td>{r['ADVP']}</td>"
                f"<td><span class='{ag_cls}'>{ag}</span></td>"
                f"<td>{fmt_pct(r['PCT'])}</td>"
                f"<td>{fmt_int(r['N'])}</td>"
                f"</tr>"
            )
        rows.append("</tbody></table>")
        return f"<div class='comp-card'>{head}{''.join(rows)}</div>"

    items_advp = [quadro_advp_html(cia, lideres_por_cia_advp(cia)) for cia in ["AZUL","GOL","LATAM"]]
    st.markdown("<div class='comp-grid'>" + "".join(items_advp) + "</div>", unsafe_allow_html=True)

@register_tab("Competitividade Cia x Trecho x ADVPs Agrupados")
def tab6_compet_tabelas(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t6")
    if df.empty:
        st.subheader("Competividade  Cia x Trecho x ADVPs Agrupados")
        st.info("Sem resultados para os filtros atuais."); 
        return
    need = {"RANKING","CIA_NORM","TRECHO","AGENCIA_NORM","IDPESQUISA"}
    if not need.issubset(df.columns):
        st.subheader("Competividade  Cia x Trecho x ADVPs Agrupados")
        st.warning(f"Colunas ausentes: {sorted(list(need - set(df.columns)))}"); 
        return

    d1 = df[df["RANKING"].astype("Int64") == 1].copy()
    d1["CIA_UP"]     = d1["CIA_NORM"].astype(str).str.upper()
    d1["TRECHO_STD"] = d1["TRECHO"].astype(str)
    d1["AG_UP"]      = d1["AGENCIA_NORM"].astype(str)

    trechos_all = sorted(df["TRECHO"].dropna().astype(str).unique().tolist())

    tot_t = (df.assign(CIA_UP=df["CIA_NORM"].astype(str).str.upper(),
                       TRECHO_STD=df["TRECHO"].astype(str))
               .groupby(["CIA_UP","TRECHO_STD"])["IDPESQUISA"]
               .nunique().reset_index(name="TotPesq"))
    win_t = (d1.groupby(["CIA_UP","TRECHO_STD","AG_UP"])["IDPESQUISA"]
               .nunique().reset_index(name="QtdTop1"))
    base_t = win_t.merge(tot_t, on=["CIA_UP","TRECHO_STD"], how="right").fillna({"QtdTop1":0})
    base_t["Pct"] = (base_t["QtdTop1"] / base_t["TotPesq"].replace(0, np.nan) * 100).fillna(0.0)

    def pick_leader(cia:str, trecho:str):
        sub = base_t[(base_t["CIA_UP"]==cia) & (base_t["TRECHO_STD"]==trecho)]
        if sub.empty or (sub["TotPesq"].sum()==0):
            return {"CIA": cia, "TRECHO": trecho, "AGENCIA":"SEM OFERTAS", "PCT":0.0, "N":0}
        top = sub.sort_values(["Pct","QtdTop1","TotPesq"], ascending=False).iloc[0]
        return {"CIA": cia, "TRECHO": trecho, "AGENCIA": str(top["AG_UP"]), "PCT": float(top["Pct"]), "N": int(top["TotPesq"])}

    st.markdown("""
    <style>
      .t6{width:100%; border-collapse:collapse; table-layout:fixed; border:3px solid #94a3b8;}
      .t6 th,.t6 td{border:1px solid #e5e7eb; padding:5px 6px; font-size:15px; line-height:1.25; text-align:center;}
      .t6 th{background:#f3f4f6; font-weight:800;}
      .t6 .l{text-align:left;} .t6 .clip{white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
      .t6 th.cia{width:58px;} .t6 th.trc{width:64px;} .t6 th.ag{width:96px;}
      .t6 th.pct,.t6 td.pct{width:200px; white-space:nowrap;}
      .sep td{padding:0!important; border:0!important; border-top:3px solid #94a3b8!important; background:#fff;}
      .pct-val{font-weight:900; font-size:16px; color:#111827;}
      .pesq{opacity:.55; font-weight:800; margin-left:6px; font-size:13px;}
      .chip{display:inline-flex; align-items:center; gap:6px; font-weight:900; font-size:15px;}
      .dot{width:10px; height:10px; border-radius:2px; display:inline-block;}
      .az{background:#2D6CDF;} .go{background:#F7C948;} .la{background:#C0392B;} .g123{color:#0B6B2B; font-weight:900;}
      .alt{background:#fcfcfc;}
      .cards-mini{display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:8px 0 0 0;}
      @media (max-width:1100px){.cards-mini{grid-template-columns:repeat(2,minmax(0,1fr));}}
      @media (max-width:700px){.cards-mini{grid-template-columns:1fr;}}
      .mini{border:2px solid #94a3b8; border-radius:12px; background:#fff; padding:10px 12px; box-shadow:0 1px 2px rgba(0,0,0,.06);}
      .mini-title{font-weight:900; font-size:13px; letter-spacing:.2px; color:#0A2A6B; margin:0 0 4px 0;}
      .mini-name{font-weight:900; font-size:18px; color:#111827; margin:0;}
      .mini-pct{font-weight:1000; font-size:24px; color:#111827; line-height:1; margin-top:4px;}
      .mini-note{opacity:.6; font-weight:800; font-size:13px; margin-top:2px;}
      .mini-line{display:flex; align-items:baseline; justify-content:space-between; gap:8px; margin-top:4px;}
      .group-title{margin:10px 0 4px 0; font-weight:900; font-size:12px; color:#0A2A6B; letter-spacing:.3px;}
    </style>
    """, unsafe_allow_html=True)

    def cia_chip(cia:str) -> str:
        cls = "az" if cia=="AZUL" else "go" if cia=="GOL" else "la"
        return f"<span class='chip'><span class='dot {cls}'></span>{cia}</span>"

    def pct_cell(pct, n) -> str:
        try: p = int(round(float(pct)))
        except: p = 0
        try: k = int(n)
        except: k = 0
        return f"<span class='pct-val'>{p}%</span><span class='pesq'>( {k} pesq )</span>"

    def ag_fmt(ag:str) -> str:
        return f"<span class='g123'>{ag}</span>" if ag in {"123MILHAS","MAXMILHAS"} else ag

    def compute_summary(win_df: pd.DataFrame, total_base: int):
        if win_df.empty or total_base == 0:
            return ("SEM OFERTAS", 0, 0, "SEM OFERTAS", 0, 0, 0)
        cia_s = win_df.groupby("CIA_UP")["IDPESQUISA"].nunique().sort_values(ascending=False)
        ag_s  = win_df.groupby("AG_UP")["IDPESQUISA"].nunique().sort_values(ascending=False)
        cia_nome, cia_qtd = (str(cia_s.index[0]), int(cia_s.iloc[0])) if not cia_s.empty else ("SEM OFERTAS", 0)
        ag_nome, ag_qtd   = (str(ag_s.index[0]),  int(ag_s.iloc[0]))  if not ag_s.empty  else ("SEM OFERTAS", 0)
        cia_pct = int(round(cia_qtd/total_base*100)) if total_base else 0
        ag_pct  = int(round(ag_qtd/total_base*100))  if total_base else 0
        return (cia_nome, cia_qtd, cia_pct, ag_nome, ag_qtd, ag_pct, total_base)

    def cards_block(title: str, cia_nome, cia_qtd, cia_pct, ag_nome, ag_qtd, ag_pct, total_base):
        return f"""
        <div class='group-title'>{title}</div>
        <div class='cards-mini'>
          <div class='mini'>
            <div class='mini-title'>CIA + BARATA</div>
            <div class='mini-name'>{cia_nome}</div>
            <div class='mini-pct'>{cia_pct}%</div>
            <div class='mini-note'>( {fmt_int(cia_qtd)} pesq )</div>
          </div>
          <div class='mini'>
            <div class='mini-title'>Agência Vencedora</div>
            <div class='mini-name'>{ag_nome}</div>
            <div class='mini-pct'>{ag_pct}%</div>
            <div class='mini-note'>( {fmt_int(ag_qtd)} pesq )</div>
          </div>
          <div class='mini'>
            <div class='mini-title'>Nº de Pesquisas</div>
            <div class='mini-name'>{fmt_int(ag_qtd)} pesquisas</div>
            <div class='mini-pct'>{ag_pct}%</div>
            <div class='mini-note'>Base: {fmt_int(total_base)} pesquisas</div>
          </div>
        </div>
        """

    # Render Tabela Trecho
    def render_tbl_trecho():
        html = ["<table class='t6'>",
                "<thead><tr><th class='cia'>CIA</th><th class='trc l'>TRECHO</th><th class='ag l'>AGENCIA</th><th class='pct'>% DE GANHO</th></tr></thead><tbody>"]
        for t in trechos_all:
            rows = [pick_leader(cia, t) for cia in ["AZUL","GOL","LATAM"]]
            for i, r in enumerate(rows):
                alt = " class='alt'" if i % 2 else ""
                html.append(
                    f"<tr{alt}><td>{cia_chip(r['CIA'])}</td>"
                    f"<td class='trc l clip'>{t}</td>"
                    f"<td class='ag l clip'>{ag_fmt(r['AGENCIA'])}</td>"
                    f"<td class='pct'>{pct_cell(r['PCT'], r['N'])}</td></tr>"
                )
            html.append("<tr class='sep'><td colspan='4'></td></tr>")
        html.append("</tbody></table>")
        st.markdown("".join(html), unsafe_allow_html=True)

    # ADVP agrupado + cards
    buckets = [1,5,11,17,30]
    advp_series = pd.to_numeric(df.get("ADVP_CANON"), errors="coerce")
    df_advp = df[advp_series.notna()].copy()
    df_advp["ADVP_BKT"] = advp_series.loc[df_advp.index].astype(int)

    d1a = d1.copy()
    d1a["ADVP_BKT"] = pd.to_numeric(df.get("ADVP_CANON"), errors="coerce")
    d1a = d1a.dropna(subset=["ADVP_BKT"])

    tot_a = (df_advp.assign(CIA_UP=df_advp["CIA_NORM"].astype(str).str.upper())
                    .groupby(["CIA_UP","ADVP_BKT"])["IDPESQUISA"]
                    .nunique().reset_index(name="TotAdvp"))
    win_a = (d1a.groupby(["CIA_UP","ADVP_BKT","AG_UP"])["IDPESQUISA"]
                .nunique().reset_index(name="QtdTop1"))
    base_a = win_a.merge(tot_a, on=["CIA_UP","ADVP_BKT"], how="right").fillna({"QtdTop1":0})
    base_a["Pct"] = (base_a["QtdTop1"] / base_a["TotAdvp"].replace(0, np.nan) * 100).fillna(0.0)

    def pick_leader_advp(cia:str, advp:int):
        sub = base_a[(base_a["CIA_UP"]==cia) & (base_a["ADVP_BKT"]==advp)]
        if sub.empty or (sub["TotAdvp"].sum()==0):
            return {"CIA": cia, "ADVP": advp, "AGENCIA":"SEM OFERTAS", "PCT":0.0, "N":0}
        top = sub.sort_values(["Pct","QtdTop1","TotAdvp"], ascending=False).iloc[0]
        return {"CIA": cia, "ADVP": advp, "AGENCIA": str(top["AG_UP"]), "PCT": float(top["Pct"]), "N": int(top["TotAdvp"])}

    def render_tbl_advp_and_cards():
        html = ["<table class='t6'>",
                "<thead><tr><th class='cia'>CIA</th><th style='width:56px'>ADVP</th><th class='ag l'>AGENCIA</th><th class='pct'>% DE GANHO</th></tr></thead><tbody>"]
        for a in buckets:
            rows = [pick_leader_advp(cia, a) for cia in ["AZUL","GOL","LATAM"]]
            for i, r in enumerate(rows):
                alt = " class='alt'" if i % 2 else ""
                html.append(
                    f"<tr{alt}><td>{cia_chip(r['CIA'])}</td>"
                    f"<td>{a}</td>"
                    f"<td class='ag l clip'>{ag_fmt(r['AGENCIA'])}</td>"
                    f"<td class='pct'>{pct_cell(r['PCT'], r['N'])}</td></tr>"
                )
            html.append("<tr class='sep'><td colspan='4'></td></tr>")
        html.append("</tbody></table>")
        st.markdown("".join(html), unsafe_allow_html=True)

        total_base_trechos = int(df["IDPESQUISA"].nunique() or 0)
        c_nome, c_qtd, c_pct, a_nome, a_qtd, a_pct, base = compute_summary(d1, total_base_trechos)
        st.markdown(cards_block("Resumo do Vencedor", c_nome, c_qtd, c_pct, a_nome, a_qtd, a_pct, base),
                    unsafe_allow_html=True)

    st.subheader("Competividade  Cia x Trecho x ADVPs Agrupados")
    c1, c2 = st.columns(2, gap="small")
    with c1:
        st.caption("Cia × Trecho")
        render_tbl_trecho()
    with c2:
        st.caption("Cia × ADVP")
        render_tbl_advp_and_cards()

# ─────────────────────────── MAIN ───────────────────────────
def main():
    df_raw = load_base(DATA_PATH)
    for ext in ("*.png","*.jpg","*.jpeg","*.gif","*.webp"):
        imgs = list(APP_DIR.glob(ext))
        if imgs:
            st.image(imgs[0].as_posix(), use_container_width=True); break
    labels = [label for label, _ in TAB_REGISTRY]
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
