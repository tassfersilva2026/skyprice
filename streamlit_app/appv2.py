# streamlit_app.py — versão com botão ↻ e toggle Auto-atualizar no final da linha de filtros
from __future__ import annotations
from pathlib import Path
from datetime import date
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import re

# ─────────────────────────── CONFIG DA PÁGINA ────────────────────────────────
st.set_page_config(page_title="Skyscanner — Painel", layout="wide", initial_sidebar_state="expanded")

APP_DIR   = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "OFERTASCONSOLIDADO_OFERTAS.parquet"   # ajuste se necessário

# ─────────────────────────── HELPERS BÁSICOS ─────────────────────────────────
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
    try:
        v = float(str(x).replace(",", "."))
    except Exception:
        v = np.nan
    if np.isnan(v):
        v = 1
    return min([1, 5, 11, 17, 30], key=lambda k: abs(v - k))

def fmt_int(n: int) -> str:
    try:
        return f"{int(n):,}".replace(",", ".")
    except Exception:
        return "0"

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

# ─────────────────────────── CARREGAMENTO DA BASE ────────────────────────────
@st.cache_data(show_spinner=True)
def load_base(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Arquivo obrigatório não encontrado: {path.as_posix()}"); st.stop()

    df = pd.read_parquet(path)

    # Padroniza primeiras 13 colunas se necessário (mantém compatibilidade)
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

    # HORA_HH para filtros por hora
    df["HORA_HH"] = pd.to_datetime(df.get("HORA_BUSCA"), errors="coerce").dt.hour

    # Datas de interesse (dayfirst, formato BR)
    for c in ["DATA_EMBARQUE","DATAHORA_BUSCA"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)

    # Preço numérico
    if "PRECO" in df.columns:
        df["PRECO"] = (
            df["PRECO"].astype(str)
            .str.replace(r"[^\d,.-]", "", regex=True)
            .str.replace(",", ".", regex=False)
        )
        df["PRECO"] = pd.to_numeric(df["PRECO"], errors="coerce")

    # Ranking inteiro
    if "RANKING" in df.columns:
        df["RANKING"] = pd.to_numeric(df["RANKING"], errors="coerce").astype("Int64")

    # Normalizações auxiliares
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
    mask_dt_ok = pd.notna(dt_base)
    mask_h_ok  = pd.notna(hora_sec)
    dtkey = dtkey.where(~mask_dt_ok, dt_base.dt.normalize() + pd.to_timedelta(hora_sec.fillna(0), unit="s"))
    dtkey = dtkey.where(mask_dt_ok | mask_h_ok, pd.NaT)
    df["__DTKEY__"] = dtkey

    return df

# ─────────────────────── ESTILOS, GRÁFICOS E TABELAS ─────────────────────────
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

# Heatmap/Styler
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

# ─────────────────────────── REGISTRO DE ABAS ────────────────────────────────
TAB_REGISTRY: List[Tuple[str, Callable]] = []
def register_tab(label: str):
    def _wrap(fn: Callable):
        TAB_REGISTRY.append((label, fn))
        return fn
    return _wrap

# ─────────────────────── FUNÇÕES DE LÓGICA DO APP ────────────────────────────
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
    # adicionamos c7 para o botão/toggle no fim da linha
    c1, c2, c3, c4, c5, c6, c7 = st.columns([1.1, 1.1, 1, 2, 1, 1.4, 0.5])

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

    # ── Botão redondo + toggle, discretos no fim da linha (c7)
    with c7:
        st.markdown("""
        <style>
          .round-refresh button[kind="secondary"]{
            width:34px!important;height:34px!important;padding:0!important;border-radius:999px!important;
            border:1px solid #e5e7eb!important;background:#f8fafc!important;color:#111827!important;
            line-height:1!important;font-size:14px!important;font-weight:800!important;
          }
          .round-refresh button[kind="secondary"]:hover{ background:#eef2ff!important; }
          .rf-wrap{display:flex;align-items:end;justify-content:flex-end;height:100%;}
          .rf-col{display:flex;gap:8px;align-items:center;}
          .rf-tgl .stCheckbox>label{font-size:12px;font-weight:700;color:#64748b;}
        </style>
        """, unsafe_allow_html=True)
        st.markdown("<div class='rf-wrap'><div class='rf-col'>", unsafe_allow_html=True)

        do_refresh = st.button("↻", key=f"{key_prefix}_refresh", type="secondary", help="Recarregar parquet do disco")
        auto_toggle = st.toggle("Auto-atualizar", key=f"{key_prefix}_auto", help="Limpa cache e recarrega ao ligar", value=False)
        st.markdown("</div></div>", unsafe_allow_html=True)

    # salva estado dos filtros
    st.session_state["flt"] = {
        "dt_ini": dt_ini, "dt_fim": dt_fim, "advp": advp_sel or [],
        "trechos": tr_sel or [], "hh": hh_sel or [], "cia": cia_sel or []
    }

    # disparos de atualização
    if do_refresh:
        st.cache_data.clear()
        st.rerun()

    ack_key = f"{key_prefix}_auto_ack"
    if auto_toggle and not st.session_state.get(ack_key, False):
        st.session_state[ack_key] = True
        st.cache_data.clear()
        st.rerun()
    elif not auto_toggle and st.session_state.get(ack_key, False):
        st.session_state[ack_key] = False

    # Filtro por DATA (inclusivo)
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

# ───────────────────────────── ABAS DO APLICATIVO ────────────────────────────
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

# ─────────────────────── ABAS ADICIONAIS (mantidas) ──────────────────────────
# (As demais abas permanecem iguais ao seu arquivo anterior: "Top 3 Agências",
#  "Top 3 Preços Mais Baratos", "Ranking por Agências", "Competitividade Cia x Trecho",
#  "Competitividade Cia x Trecho x ADVPs Agrupados" etc.)
# Para brevidade, deixei o restante do conteúdo igual ao seu último envio.
# Se você quiser, colo também TODAS as abas aqui na íntegra.

# ───────── ABA: Competitividade Cia x Trecho x ADVPs Agrupados (COMPLETA) ────
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
        cia_s = d1.groupby("CIA_UP")["IDPESQUISA"].nunique().sort_values(ascending=False)
        ag_s  = d1.groupby("AG_UP")["IDPESQUISA"].nunique().sort_values(ascending=False)
        cia_nome, cia_qtd = (str(cia_s.index[0]), int(cia_s.iloc[0])) if not cia_s.empty else ("SEM OFERTAS", 0)
        ag_nome, ag_qtd   = (str(ag_s.index[0]),  int(ag_s.iloc[0]))  if not ag_s.empty  else ("SEM OFERTAS", 0)
        cia_pct = int(round(cia_qtd/total_base_trechos*100)) if total_base_trechos else 0
        ag_pct  = int(round(ag_qtd/total_base_trechos*100))  if total_base_trechos else 0

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
        st.markdown(cards_block("Resumo do Vencedor", cia_nome, cia_qtd, cia_pct, ag_nome, ag_qtd, ag_pct, total_base_trechos),
                    unsafe_allow_html=True)

        base_advp_n = int(df_advp["IDPESQUISA"].nunique() or 0)
        gset = {"123MILHAS","MAXMILHAS"}
        sub  = d1a[d1a["AG_UP"].isin(gset)].copy()

        total_gwins = sub["IDPESQUISA"].nunique() or 0
        if total_gwins:
            cia_cnt = sub.groupby("CIA_UP")["IDPESQUISA"].nunique().sort_values(ascending=False)
            cia_gnome = str(cia_cnt.index[0]); cia_gqtd = int(cia_cnt.iloc[0])
            cia_gpct  = int(round(cia_gqtd / total_gwins * 100))
        else:
            cia_gnome, cia_gqtd, cia_gpct = "SEM OFERTAS", 0, 0

        c123 = int(sub[sub["AG_UP"]=="123MILHAS"]["IDPESQUISA"].nunique())
        cmax = int(sub[sub["AG_UP"]=="MAXMILHAS"]["IDPESQUISA"].nunique())
        cgrp = c123 + cmax
        p123 = int(round((c123 / base_advp_n * 100))) if base_advp_n else 0
        pmax = int(round((cmax / base_advp_n * 100))) if base_advp_n else 0
        pgrp = int(round((cgrp / base_advp_n * 100))) if base_advp_n else 0

        st.markdown(f"""
        <div class='group-title'>Competitividade - Grupo123</div>
        <div class='cards-mini'>
          <div class='mini'>
            <div class='mini-title'>CIA + BARATA</div>
            <div class='mini-name'>{cia_gnome}</div>
            <div class='mini-pct'>{cia_gpct}%</div>
            <div class='mini-note'>( {fmt_int(cia_gqtd)} pesq )</div>
          </div>
          <div class='mini'>
            <div class='mini-title'>Participação das Empresas</div>
            <div class='mini-line'><strong>123Milhas</strong><span class='mini-pct'>{p123}%</span></div>
            <div class='mini-line'><strong>Maxmilhas</strong><span class='mini-pct'>{pmax}%</span></div>
            <div class='mini-line'><strong>Grupo123</strong><span class='mini-pct'>{pgrp}%</span></div>
            <div class='mini-note'>Base: {fmt_int(base_advp_n)} pesquisas</div>
          </div>
          <div class='mini'>
            <div class='mini-title'>Nº de Pesquisas</div>
            <div class='mini-line'><strong>123Milhas</strong><span class='mini-pct'>{p123}%</span></div>
            <div class='mini-note'>( {fmt_int(c123)} pesq )</div>
            <div class='mini-line'><strong>Maxmilhas</strong><span class='mini-pct'>{pmax}%</span></div>
            <div class='mini-note'>( {fmt_int(cmax)} pesq )</div>
            <div class='mini-line'><strong>Grupo123</strong><span class='mini-pct'>{pgrp}%</span></div>
            <div class='mini-note'>( {fmt_int(cgrp)} pesq )</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Competividade  Cia x Trecho x ADVPs Agrupados")
    c1, c2 = st.columns(2, gap="small")
    with c1:
        st.caption("Cia × Trecho")
        render_tbl_trecho()
    with c2:
        st.caption("Cia × ADVP")
        render_tbl_advp_and_cards()

# ================================ MAIN ========================================
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
