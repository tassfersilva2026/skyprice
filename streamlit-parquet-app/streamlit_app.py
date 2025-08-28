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
    return s  # mantém como veio

def advp_nearest(x) -> int:
    try: v = float(str(x).replace(",", "."))
    except Exception: v = np.nan
    if np.isnan(v): v = 1
    return min([1, 5, 11, 17, 30], key=lambda k: abs(v - k))

@st.cache_data(show_spinner=True)
def load_base(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Arquivo obrigatório não encontrado: {path.as_posix()}"); st.stop()
    df = pd.read_parquet(path)

    colmap = {0:"IDPESQUISA",1:"CIA",2:"HORA_BUSCA",3:"HORA_PARTIDA",4:"HORA_CHEGADA",
              5:"TIPO_VOO",6:"DATA_EMBARQUE",7:"DATAHORA_BUSCA",8:"AGENCIA_COMP",9:"PRECO",
              10:"TRECHO",11:"ADVP",12:"RANKING"}
    if list(df.columns[:13]) != list(colmap.values()):
        rename = {df.columns[i]: colmap[i] for i in range(min(13, df.shape[1]))}
        df = df.rename(columns=rename)

    # Normaliza horas como texto HH:MM:SS (inclusive HORA_BUSCA – coluna C)
    for c in ["HORA_BUSCA","HORA_PARTIDA","HORA_CHEGADA"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c].astype(str).str.strip(), errors="coerce").dt.strftime("%H:%M:%S")

    df["HORA_HH"] = pd.to_datetime(df["HORA_BUSCA"], errors="coerce").dt.hour

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
            s[["IDPESQUISA","AGENCIA_NORM"]].rename(columns={"AGENCIA_NORM": f"R{r}"}),
            on="IDPESQUISA", how="left"
        )
    for r in (1,2,3): base[f"R{r}"] = base[f"R{r}"].fillna("SEM OFERTAS")
    return base

def fmt_int(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")

def last_update_from_cols(df: pd.DataFrame) -> str:
    if df.empty: return "—"
    max_d = pd.to_datetime(df["DATAHORA_BUSCA"], errors="coerce").max()
    if pd.isna(max_d): return "—"
    same_day = df[pd.to_datetime(df["DATAHORA_BUSCA"], errors="coerce").dt.date == max_d.date()]
    hh = pd.to_datetime(same_day["HORA_BUSCA"], errors="coerce").dt.time
    max_h = max([h for h in hh if pd.notna(h)], default=None)
    if isinstance(max_h, dtime):
        return f"{max_d.strftime('%d/%m/%Y')} - {max_h.strftime('%H:%M:%S')}"
    return f"{max_d.strftime('%d/%m/%Y')}"

# ---- CSS global: largura total
GLOBAL_TABLE_CSS = """
<style>
table { width:100% !important; }
.dataframe { width:100% !important; }
</style>
"""
st.markdown(GLOBAL_TABLE_CSS, unsafe_allow_html=True)

# ---- Estilos dos cards (Painel)
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

# ---- Estilo para cards empilhados por Cia
CARDS_STACK_CSS = """
<style>
  .cards-stack { display:flex; flex-direction:column; gap:10px; }
  .cards-stack .card { width:100%; }
  .stack-title { font-weight:800; padding:8px 10px; margin:6px 0 10px 0; border-radius:10px; border:1px solid #e9e9ee; background:#f8fafc; color:#0A2A6B; }
</style>
"""

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

# ---- Gráficos utilitários
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

# ===================== FORMATAÇÃO & HEATMAP (SEM MATPLOTLIB) ==================

BLUE  = "#cfe3ff"
ORANGE= "#fdd0a2"
GREEN = "#c7e9c0"
YELLOW= "#fee391"
PINK  = "#f1b6da"

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

def style_smart_colwise(df_show: pd.DataFrame, fmt_map: dict, grad_cols: list[str]):
    sty = (df_show.style
           .set_properties(**{"background-color": "#FFFFFF", "color": "#111111"})
           .set_table_attributes('style="width:100%; table-layout:fixed"'))
    if fmt_map:
        sty = sty.format(fmt_map, na_rep="-")
    for c in grad_cols:
        if c in df_show.columns:
            sty = style_heatmap_discrete(sty, c, _pick_scale(c))
    # esconder índice (adeus "#")
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

# ============================ REGISTRO DE ABAS ================================
TAB_REGISTRY: List[Tuple[str, Callable]] = []
def register_tab(label: str):
    def _wrap(fn: Callable):
        TAB_REGISTRY.append((label, fn))
        return fn
    return _wrap

# ============================ FILTROS =========================================
def _init_filter_state(df_raw: pd.DataFrame):
    if "flt" in st.session_state: return
    dmin = pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce").min()
    dmax = pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce").max()
    st.session_state["flt"] = {
        "dt_ini": (dmin.date() if pd.notna(dmin) else date(2000, 1, 1)),
        "dt_fim": (dmax.date() if pd.notna(dmax) else date.today()),
        "advp": [], "trechos": [], "hh": [], "cia": [],
    }

def render_filters(df_raw: pd.DataFrame, key_prefix: str = "flt"):
    _init_filter_state(df_raw)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns([1.1, 1.1, 1, 2, 1, 1.4])

    dmin_abs = pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce").min()
    dmax_abs = pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce").max()
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
        advp_all = sorted(set(pd.to_numeric(df_raw["ADVP_CANON"], errors="coerce").dropna().astype(int).tolist()))
        advp_sel = st.multiselect("ADVP", options=advp_all,
                                  default=st.session_state["flt"]["advp"], key=f"{key_prefix}_advp")
    with c4:
        trechos_all = sorted([t for t in df_raw["TRECHO"].dropna().unique().tolist() if str(t).strip() != ""])
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

    st.session_state["flt"] = {
        "dt_ini": dt_ini, "dt_fim": dt_fim, "advp": advp_sel or [],
        "trechos": tr_sel or [], "hh": hh_sel or [], "cia": cia_sel or []
    }

    mask = pd.Series(True, index=df_raw.index)
    mask &= (pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce") >= pd.Timestamp(dt_ini))
    mask &= (pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce") <= pd.Timestamp(dt_fim))
    if advp_sel: mask &= df_raw["ADVP_CANON"].isin(advp_sel)
    if tr_sel:  mask &= df_raw["TRECHO"].isin(tr_sel)
    if hh_sel:  mask &= df_raw["HORA_HH"].isin(hh_sel)
    if st.session_state["flt"]["cia"]:
        mask &= df_raw["CIA_NORM"].astype(str).str.upper().isin(st.session_state["flt"]["cia"])

    df = df_raw[mask].copy()
    st.caption(f"Linhas após filtros: {fmt_int(len(df))} • Última atualização: {last_update_from_cols(df)}")
    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
    return df

# =============================== ABAS (INÍCIO) ===============================
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

# ──────────────────────── ABA: Top 3 Agências (START) ────────────────────────

@register_tab("Top 3 Agências")
def tab2_top3_agencias(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t2")
    if df.empty:
        st.info("Sem resultados para os filtros selecionados.")
        return

    # 1) Garantir mesma pesquisa (pega a última por Trecho)
    df2 = df.copy()
    df2["DT"] = pd.to_datetime(df2["DATAHORA_BUSCA"], errors="coerce")
    g = (df2.dropna(subset=["TRECHO","IDPESQUISA","DT"])
              .groupby(["TRECHO","IDPESQUISA"], as_index=False)["DT"].max())
    last_idx = g.groupby("TRECHO")["DT"].idxmax()
    last_ids = {r["TRECHO"]: r["IDPESQUISA"] for _, r in g.loc[last_idx].iterrows()}
    df2["__ID_TARGET__"] = df2["TRECHO"].map(last_ids)
    df_last = df2[df2["IDPESQUISA"].astype(str) == df2["__ID_TARGET__"].astype(str)].copy()

    # 2) Coluna "Data/Hora Busca" (DATAHORA + hora da coluna C/HORA_BUSCA)
    def _compose_dt_hora(sub: pd.DataFrame) -> str:
        d = pd.to_datetime(sub["DATAHORA_BUSCA"], errors="coerce").max()
        # tenta extrair HH:MM:SS válidos da coluna C
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

    # 3) Ranking Top-3 por Trecho (mesma pesquisa)
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
    t1 = t1.reset_index(drop=True)  # remove a coluna "#"
    for c in ["Preço Top 1","Preço Top 2","Preço Top 3"]:
        t1[c] = pd.to_numeric(t1[c], errors="coerce")
    sty1 = style_smart_colwise(t1, {c: fmt_num0_br for c in ["Preço Top 1","Preço Top 2","Preço Top 3"]},
                               grad_cols=["Preço Top 1","Preço Top 2","Preço Top 3"])
    show_table(t1, sty1, caption="Ranking Top 3 (Agências) — por trecho")

    # 4) % Diferença vs Top1 (mesma pesquisa)
    def pct_diff(base, other):
        if pd.isna(base) or base == 0 or pd.isna(other):
            return np.nan
        return (other - base) / base * 100

    rows2 = []
    for _, r in t1.iterrows():
        base = r["Preço Top 1"]
        rows2.append({
            "Data/Hora Busca": r["Data/Hora Busca"],
            "Trecho": r["Trecho"],
            "Agencia Top 1": r["Agencia Top 1"], "Preço Top 1": base,
            "Agencia Top 2": r["Agencia Top 2"], "% Dif Top2 vs Top1": pct_diff(base, r["Preço Top 2"]),
            "Agencia Top 3": r["Agencia Top 3"], "% Dif Top3 vs Top1": pct_diff(base, r["Preço Top 3"]),
        })
    t2 = pd.DataFrame(rows2).reset_index(drop=True)
    sty2 = style_smart_colwise(
        t2,
        {"Preço Top 1": fmt_num0_br, "% Dif Top2 vs Top1": fmt_pct2_br, "% Dif Top3 vs Top1": fmt_pct2_br},
        grad_cols=["Preço Top 1", "% Dif Top2 vs Top1", "% Dif Top3 vs Top1"]
    )
    show_table(t2, sty2, caption="% Diferença entre Agências (base: TOP1)")

    # 5) Comparativo Cia × Agências de milhas (mesma pesquisa) – mantém largura e sem '#'
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


# ──────────────────── ABA: Top 3 Preços Mais Baratos (START) ─────────────────
@register_tab("Top 3 Preços Mais Baratos")
def tab3_top3_precos(df_raw: pd.DataFrame):
    """
    Pódio por Trecho → ADVP (última pesquisa de cada par).
    Horário do badge vem de HORA_BUSCA — HH:MM:SS.
    Inclui 'chips' extras para 123MILHAS e MAXMILHAS quando não estiverem no Top 3:
      - mostra posição, preço e % vs 1º
      - se não existir ocorrência: 'Não apareceu'
    """
    import re

    df = render_filters(df_raw, key_prefix="t3")
    st.subheader("Pódio por Trecho → ADVP (última pesquisa de cada par)")

    top_row = st.container()
    with top_row:
        c1, c2, c3 = st.columns([0.28, 0.18, 0.54])
        agencia_foco = c1.selectbox("Agência alvo", ["Todos", "123MILHAS", "MAXMILHAS"], index=0)
        posicao_foco = c2.selectbox("Ranking", ["Todas", 1, 2, 3], index=0)
        por_pesquisa = c3.checkbox("Isolar última pesquisa por Trecho×ADVP", value=True)

    if df.empty:
        st.info("Sem resultados para os filtros selecionados."); return

    # ---------- Helpers ----------
    def fmt_moeda_br(x) -> str:
        try:
            xv = float(x)
            if not np.isfinite(xv): return "R$ -"
            return "R$ " + f"{xv:,.0f}".replace(",", ".")
        except Exception:
            return "R$ -"

    def fmt_pct_plus(x) -> str:
        try:
            v = float(x)
            if not np.isfinite(v): return "—"
            return f"+{round(v):.0f}%"
        except Exception:
            return "—"

    def _canon(s: str) -> str:
        return re.sub(r"[^A-Z0-9]+", "", str(s).upper())

    def _brand_tag(s: str) -> str | None:
        cs = _canon(s)
        if cs.startswith("123MILHAS") or cs == "123": return "123MILHAS"
        if cs.startswith("MAXMILHAS") or cs == "MAX":  return "MAXMILHAS"
        return None

    def _find_id_col(df_: pd.DataFrame) -> str | None:
        cands = ["IDPESQUISA","ID_PESQUISA","ID BUSCA","IDBUSCA","ID","NOME_ARQUIVO_STD","NOME_ARQUIVO","NOME DO ARQUIVO","ARQUIVO"]
        norm = { re.sub(r"[^A-Z0-9]+","", c.upper()): c for c in df_.columns }
        for nm in cands:
            key = re.sub(r"[^A-Z0-9]+","", nm.upper())
            if key in norm: return norm[key]
        return df_.columns[0] if len(df_.columns) else None

    def _normalize_id(val):
        if val is None or (isinstance(val, float) and np.isnan(val)): return None
        s = str(val)
        try:
            f = float(s.replace(",", "."))
            if f.is_integer(): return str(int(f))
        except Exception:
            pass
        return s

    # ---------- Estilos ----------
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

    # Chips extras para 123/Max
    EXTRAS_STYLE  = "border-top:1px dashed #e5e7eb;margin:6px 8px 8px 8px;padding-top:6px;display:flex;gap:6px;flex-wrap:wrap;"
    CHIP_STYLE    = "background:#f3f4f6;border:1px solid #e5e7eb;border-radius:10px;padding:2px 6px;font-weight:700;font-size:11px;color:#111827;white-space:nowrap;line-height:1.1;"

    BADGE_POP_CSS = """
    <style>
    .idp-wrap{position:relative; display:inline-flex; align-items:center;}
    .idp-badge{
      display:inline-flex; align-items:center; justify-content:center;
      width:16px; height:16px; border:1px solid #cbd5e1; border-radius:50%;
      font-size:11px; font-weight:900; color:#64748b; background:#fff;
      user-select:none; cursor:default; line-height:1;
    }
    .idp-pop{
      position:absolute; top:18px; right:0;
      background:#fff; color:#0f172a; border:1px solid #e5e7eb;
      border-radius:8px; padding:6px 8px; font-size:12px; font-weight:700;
      box-shadow:0 6px 16px rgba(0,0,0,.08); display:none; z-index:9999; white-space:nowrap;
    }
    .idp-wrap:hover .idp-pop{ display:block; }
    .idp-idbox{
      border:1px solid #e5e7eb; background:#f8fafc; border-radius:6px;
      padding:2px 6px; font-weight:800; font-size:12px; min-width:60px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      user-select:text; cursor:text;
    }
    </style>
    """
    st.markdown(BADGE_POP_CSS, unsafe_allow_html=True)

    def _card_html(rank: int, agencia: str, preco: float, subtxt: str, dt: str, id_to_copy: str | None) -> str:
        stripe = "#D4AF37" if rank == 1 else ("#9CA3AF" if rank == 2 else "#CD7F32")
        stripe_div = f"<div style='position:absolute;left:0;top:0;bottom:0;width:6px;border-radius:10px 0 0 10px;background:{stripe};'></div>"
        badge = ""
        if id_to_copy:
            sid = str(id_to_copy).replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")
            badge = (f"<span class='idp-wrap'><span class='idp-badge' title='Duplo clique no ID para selecionar'>?</span>"
                     f"<span class='idp-pop'>ID:&nbsp;<input class='idp-idbox' type='text' value='{sid}' readonly></span></span>")
        return (
            f"<div style='{CARD_BASE}'>"
            f"{stripe_div}"
            f"<div style='{DT_WRAP_STYLE}'><span style='{DT_TXT_STYLE}'>{dt}</span>{badge}</div>"
            f"<div style='{RANK_STYLE}'>{rank}º</div>"
            f"<div style='{AG_STYLE}'>{agencia}</div>"
            f"<div style='{PR_STYLE}'>{fmt_moeda_br(preco)}</div>"
            f"<div style='{SUB_STYLE}'>{subtxt}</div>"
            f"</div>"
        )

    # ---------- Base preparada ----------
    dfp = df.copy()
    # Colunas canônicas esperadas já devem existir via seus passos anteriores:
    # TRECHO_STD, AGENCIA_UP, ADVP, __PRECO__, __DTKEY__, DATAHORA_BUSCA, HORA_BUSCA
    dfp["__PRECO__"] = pd.to_numeric(dfp.get("__PRECO__", dfp.get("PRECO")), errors="coerce")
    dfp = dfp[dfp["__PRECO__"].notna()].copy()

    ID_COL = "IDPESQUISA" if "IDPESQUISA" in dfp.columns else _find_id_col(dfp)
    PESQ_COL = "NOME_ARQUIVO_STD" if "NOME_ARQUIVO_STD" in dfp.columns else None
    PESQ_OK = PESQ_COL is not None and PESQ_COL in dfp.columns

    if dfp.empty:
        st.info("Sem preços válidos no recorte atual."); return

    # Última pesquisa por Trecho×ADVP (quando marcado)
    pesq_por_ta = {}
    if por_pesquisa and PESQ_OK:
        tmp = dfp.dropna(subset=["TRECHO_STD","ADVP",PESQ_COL,"__DTKEY__"]).copy()
        g = tmp.groupby(["TRECHO_STD","ADVP",PESQ_COL], as_index=False)["__DTKEY__"].max()
        if not g.empty:
            idx = g.groupby(["TRECHO_STD","ADVP"])["__DTKEY__"].idxmax()
            idx = idx[idx.notna()].astype(int)
            if len(idx):
                last_by_ta = g.loc[idx.values]
                pesq_por_ta = {(str(r["TRECHO_STD"]), str(r["ADVP"])): str(r[PESQ_COL]) for _, r in last_by_ta.iterrows()}

    def presence_flags(df_all_rows: pd.DataFrame, label: str) -> dict:
        # Marca presença da marca (123 ou Max), e se tem preço válido
        sub = df_all_rows[df_all_rows["AGENCIA_UP"].astype(str).apply(_brand_tag) == _brand_tag(label)]
        present_any = not sub.empty
        present_with_price = False
        if present_any:
            if np.isfinite(sub["__PRECO__"]).any():
                present_with_price = True
        return {"present_any": present_any, "present_with_price": present_with_price}

    def dt_and_id_for_rows(sub_rows: pd.DataFrame) -> tuple[str, str | None]:
        if sub_rows.empty: return "", None
        r = sub_rows.loc[sub_rows["__DTKEY__"].idxmax()]
        date_part = pd.to_datetime(r.get("DATAHORA_BUSCA"), errors="coerce")
        date_txt  = date_part.strftime("%d/%m") if pd.notna(date_part) else ""
        htxt_raw  = str(r.get("HORA_BUSCA","")).strip()
        htxt = htxt_raw if htxt_raw else (pd.to_datetime(r["__DTKEY__"], errors="coerce").strftime("%H:%M:%S") if pd.notna(r["__DTKEY__"]) else "")
        id_val = _normalize_id(r.get(ID_COL))
        return f"{date_txt} {htxt}".strip(), id_val

    def build_rank(df_subset: pd.DataFrame) -> pd.DataFrame:
        tmp = df_subset.copy()
        tmp["AGENCIA_UP"] = tmp["AGENCIA_UP"].astype(str)
        rank = (
            tmp.groupby("AGENCIA_UP", as_index=False)["__PRECO__"].min()
               .sort_values("__PRECO__")
               .reset_index(drop=True)
        )
        if not rank.empty:
            rank["_CAN"] = rank["AGENCIA_UP"].apply(_canon)
            rank["_BRAND"] = rank["AGENCIA_UP"].apply(_brand_tag)
        return rank

    trechos_sorted = sorted(dfp["TRECHO_STD"].dropna().astype(str).unique(), key=lambda x: str(x))
    for trecho in trechos_sorted:
        df_t = dfp[dfp["TRECHO_STD"] == trecho]
        advps = sorted(
            df_t["ADVP"].dropna().astype(str).unique(),
            key=lambda v: (0, int(re.search(r"\d+", str(v)).group())) if re.search(r"\d+", str(v)) else (1, str(v)),
        )
        boxes = []

        for advp in advps:
            df_ta = df_t[df_t["ADVP"].astype(str) == str(advp)].copy()
            if por_pesquisa and PESQ_OK and pesq_por_ta:
                pesq_id = pesq_por_ta.get((trecho, advp))
                all_rows = df_ta[df_ta[PESQ_COL].astype(str) == pesq_id] if pesq_id is not None else df_ta.iloc[0:0]
            else:
                all_rows = df_ta
                pesq_id = None

            base_rank = build_rank(all_rows)

            # Filtro "agência alvo" / "ranking"
            if not base_rank.empty and agencia_foco != "Todos":
                rk_map = {row["AGENCIA_UP"]: i+1 for i, row in base_rank.head(3).iterrows()}
                found_target = False
                for ag_up, rank_val in rk_map.items():
                    if _canon(ag_up) == _canon(agencia_foco):
                        if posicao_foco == "Todas" or rank_val == int(posicao_foco):
                            found_target = True
                            break
                if not found_target:
                    continue

            box_content = []
            box_content.append(f"<div style='{BOX_STYLE}'>")
            box_content.append(f"<div style='{HEAD_STYLE}'>ADVP: <b>{advp}</b></div>")

            if base_rank.empty:
                box_content.append(f"<div style='{NO_STYLE}'>Sem ofertas</div>")
                extras = []
                for label in ["123MILHAS", "MAXMILHAS"]:
                    pres = presence_flags(all_rows, label)
                    if not pres["present_any"]:
                        extras.append(f"<span style='{CHIP_STYLE}'>{label}: Não apareceu</span>")
                    elif not pres["present_with_price"]:
                        extras.append(f"<span style='{CHIP_STYLE}'>{label}: Sem ofertas</span>")
                if extras:
                    box_content.append(f"<div style='{EXTRAS_STYLE}'>" + "".join(extras) + "</div>")
                box_content.append("</div>")
                boxes.append("".join(box_content))
                continue

            podium = base_rank.head(3).copy()

            # --- Cartões Top 3 ---
            box_content.append(f"<div style='{STACK_STYLE}'>")
            for i in range(len(podium)):
                current = podium.iloc[i]
                preco_i = float(current["__PRECO__"])
                ag_i    = current["AGENCIA_UP"]
                sub_rows = all_rows[(all_rows["AGENCIA_UP"].astype(str).apply(_canon) == _canon(ag_i)) &
                                    (np.isclose(all_rows["__PRECO__"], preco_i, atol=1))]
                dt_lbl, id_val = dt_and_id_for_rows(sub_rows)

                if i == 0 and len(podium) >= 2:
                    p2 = float(podium.iloc[1]["__PRECO__"])
                    subtxt = "—" if not (np.isfinite(p2) and p2 != 0) else f"−{int(round((p2 - preco_i)/p2*100.0))}% vs 2º"
                else:
                    p1 = float(podium.iloc[0]["__PRECO__"])
                    subtxt = "—" if not (np.isfinite(p1) and p1 != 0) else f"{fmt_pct_plus((preco_i - p1)/p1*100.0)} vs 1º"

                box_content.append(_card_html(i+1, ag_i, preco_i, subtxt, dt_lbl, id_val))
            box_content.append("</div>")  # stack

            # --- Extras: 123MILHAS / MAXMILHAS quando fora do pódio ---
            extras_chips = []
            p1 = float(podium.iloc[0]["__PRECO__"])
            podium_brands = {_brand_tag(x) for x in podium["AGENCIA_UP"]}

            for target_label in ["123MILHAS", "MAXMILHAS"]:
                if _brand_tag(target_label) in podium_brands:
                    continue  # já está no Top 3

                match = base_rank[base_rank["_BRAND"] == _brand_tag(target_label)]
                if match.empty:
                    pres = presence_flags(all_rows, target_label)
                    if not pres["present_any"]:
                        extras_chips.append(f"<span style='{CHIP_STYLE}'>{target_label}: Não apareceu</span>")
                    else:
                        extras_chips.append(f"<span style='{CHIP_STYLE}'>{target_label}: Sem ofertas</span>")
                else:
                    pos = int(match.index[0]) + 1
                    preco_val = float(match.iloc[0]["__PRECO__"])
                    delta = None if not (np.isfinite(preco_val) and np.isfinite(p1) and p1 != 0) else ((preco_val - p1) / p1 * 100.0)

                    # timestamp do registro com este preço
                    sub_rows = all_rows[(all_rows["AGENCIA_UP"].astype(str).apply(_brand_tag) == _brand_tag(target_label)) &
                                        (np.isclose(all_rows["__PRECO__"], preco_val, atol=1))]
                    ts_lbl, _ = dt_and_id_for_rows(sub_rows)
                    pct_str = f" {fmt_pct_plus(delta)}" if delta is not None else ""
                    ts_part = f" | {ts_lbl}" if ts_lbl else ""
                    extras_chips.append(f"<span style='{CHIP_STYLE}'>{pos}º {target_label}: {fmt_moeda_br(preco_val)}{pct_str}{ts_part}</span>")

            if extras_chips:
                box_content.append(f"<div style='{EXTRAS_STYLE}'>" + "".join(extras_chips) + "</div>")

            box_content.append("</div>")  # box
            boxes.append("".join(box_content))

        if boxes:
            st.markdown(f"<div style='{TRE_HDR_STYLE}'>Trecho: <b>{trecho}</b></div>", unsafe_allow_html=True)
            st.markdown("<div style='" + GRID_STYLE + "'>" + "".join(boxes) + "</div>", unsafe_allow_html=True)


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


# ─────────────────────── ABA: Ranking por Agências (START) ────────────────────
@register_tab("Ranking por Agências")
def tab4_ranking_agencias(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t4")
    st.subheader("Ranking por Agências (1º ao 15º)")
    if df.empty: st.info("Sem dados para os filtros."); return

    wins = (df[df["RANKING"].eq(1)].groupby("AGENCIA_NORM", as_index=False).size().rename(columns={"size":"Top1 Wins"}))
    wins = wins.sort_values("Top1 Wins", ascending=False)
    top15 = wins.head(15).reset_index(drop=True)
    fmt_map = {"Top1 Wins": fmt_num0_br}
    sty = style_smart_colwise(top15, fmt_map, grad_cols=["Top1 Wins"])
    show_table(top15, sty, caption="Top 15 — Contagem de 1º lugar por Agência")
    st.altair_chart(make_bar(top15, "Top1 Wins", "AGENCIA_NORM"), use_container_width=True)

@register_tab("Qtde de Buscas x Ofertas")
def tab6_buscas_vs_ofertas(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t6")
    st.subheader("Quantidade de Buscas x Ofertas")
    searches = df["IDPESQUISA"].nunique(); offers = len(df)
    c1, c2 = st.columns(2)
    c1.metric("Pesquisas únicas", fmt_int(searches))
    c2.metric("Ofertas (linhas)", fmt_int(offers))
    t = pd.DataFrame({"Métrica": ["Pesquisas", "Ofertas"], "Valor": [searches, offers]})
    st.altair_chart(make_bar(t, "Valor", "Métrica"), use_container_width=True)

@register_tab("Comportamento Cias")
def tab7_comportamento_cias(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t7")
    st.subheader("Comportamento Cias (share por Trecho)")
    base = df.groupby(["TRECHO","AGENCIA_NORM"]).size().rename("Qtde").reset_index()
    if base.empty: st.info("Sem dados."); return
    top_trechos = base.groupby("TRECHO")["Qtde"].sum().sort_values(ascending=False).head(10).index.tolist()
    base = base[base["TRECHO"].isin(top_trechos)]
    total_trecho = base.groupby("TRECHO")["Qtde"].transform("sum")
    base["Share"] = (base["Qtde"]/total_trecho*100).round(2)
    chart = alt.Chart(base).mark_bar().encode(
        x=alt.X("Share:Q", stack="normalize", axis=alt.Axis(format="%")),
        y=alt.Y("TRECHO:N", sort="-x"),
        color=alt.Color("AGENCIA_NORM:N"),
        tooltip=["TRECHO","AGENCIA_NORM","Share"]
    ).properties(height=320)
    st.altair_chart(chart, use_container_width=True)

@register_tab("Competitividade")
def tab8_competitividade(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t8")
    st.subheader("Competitividade (Δ mediano vs melhor preço por pesquisa)")
    best = df.groupby("IDPESQUISA")["PRECO"].min().rename("BEST").reset_index()
    t = df.merge(best, on="IDPESQUISA", how="left")
    t["DELTA"] = t["PRECO"] - t["BEST"]
    agg = t.groupby("AGENCIA_NORM", as_index=False)["DELTA"].median().rename(columns={"DELTA":"Δ Mediano"})
    st.altair_chart(make_bar(agg, "Δ Mediano", "AGENCIA_NORM"), use_container_width=True)

@register_tab("Melhor Preço Diário")
def tab9_melhor_preco_diario(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t9")
    st.subheader("Melhor Preço Diário (col. H - Data da busca)")
    t = df.groupby(df["DATAHORA_BUSCA"].dt.date, as_index=False)["PRECO"].min().rename(
        columns={"DATAHORA_BUSCA":"Data","PRECO":"Melhor Preço"}
    )
    if t.empty: st.info("Sem dados."); return
    t["Data"] = pd.to_datetime(t["Data"], dayfirst=True)
    st.altair_chart(make_line(t, "Data", "Melhor Preço"), use_container_width=True)

@register_tab("Exportar")
def tab10_exportar(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t10")
    st.subheader("Exportar dados filtrados")
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Baixar CSV (filtro aplicado)", data=csv_bytes, file_name="OFERTAS_filtrado.csv", mime="text/csv")

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
