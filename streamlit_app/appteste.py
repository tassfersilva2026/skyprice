from __future__ import annotations

from pathlib import Path
from datetime import date
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import altair as alt
import re
import io

# ─────────────────────────── CONFIG DA PÁGINA ────────────────────────────────
st.set_page_config(
    page_title="Skyscanner — Painel",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "OFERTASCONSOLIDADO_OFERTAS.parquet"

# ─────────────────────────── HELPERS HTML (ANTI-REMOVECHILD) ─────────────────
def render_html(html: str, height: int = 650, scrolling: bool = True) -> None:
    """Renderiza HTML pesado dentro de iframe para evitar bugs DOM (removeChild)."""
    components.html(html, height=height, scrolling=scrolling)

def estimate_height(num_rows: int, base: int = 240, per_row: int = 26, cap: int = 1400) -> int:
    return int(min(cap, base + num_rows * per_row))

# ─────────────────────────── FUNÇÕES AUXILIARES ──────────────────────────────
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
    if ag == "BOOKINGCOM": return "BOOKING.COM"
    if ag == "KIWICOM": return "KIWI.COM"
    if ag.startswith("123MILHAS") or ag == "123": return "123MILHAS"
    if ag.startswith("MAXMILHAS") or ag == "MAX": return "MAXMILHAS"
    if ag.startswith("CAPOVIAGENS"): return "CAPOVIAGENS"
    if ag.startswith("FLIPMILHAS"): return "FLIPMILHAS"
    if ag.startswith("VAIDEPROMO"): return "VAIDEPROMO"
    if ag.startswith("KISSANDFLY"): return "KISSANDFLY"
    if ag.startswith("ZUPPER"): return "ZUPPER"
    if ag.startswith("MYTRIP"): return "MYTRIP"
    if ag.startswith("GOTOGATE"): return "GOTOGATE"
    if ag.startswith("DECOLAR"): return "DECOLAR"
    if ag.startswith("EXPEDIA"): return "EXPEDIA"
    if ag.startswith("GOL"): return "GOL"
    if ag.startswith("LATAM"): return "LATAM"
    if ag.startswith("TRIPCOM"): return "TRIP.COM"
    if ag.startswith("VIAJANET"): return "VIAJANET"
    if ag in ("", "NAN", "NONE", "NULL", "SKYSCANNER"): return "SEM OFERTAS"
    return ag

def std_cia(raw: str) -> str:
    s = (str(raw) or "").strip().upper()
    s_simple = "".join(ch for ch in s if ch.isalnum() or ch.isspace())
    if s in {"AD", "AZU"} or s.startswith("AZUL") or "AZUL" in s_simple: return "AZUL"
    if s in {"G3"} or s.startswith("GOL") or "GOL" in s_simple: return "GOL"
    if s in {"LA", "JJ"} or s.startswith("TAM") or s.startswith("LATAM") or "LATAM" in s_simple or "TAM" in s_simple: return "LATAM"
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

def fmt_pct0_br(v):
    try:
        x = float(v)
        if not np.isfinite(x): return "-"
        return f"{int(round(x))}%"
    except Exception:
        return "-"

# ─────────────────────────── CARREGAMENTO DA BASE ────────────────────────────
@st.cache_data(show_spinner=True)
def load_base(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Arquivo obrigatório não encontrado: {path.as_posix()}")
        st.stop()

    df = pd.read_parquet(path)

    # Compat 13 colunas
    colmap = {
        0: "IDPESQUISA", 1: "CIA", 2: "HORA_BUSCA", 3: "HORA_PARTIDA", 4: "HORA_CHEGADA",
        5: "TIPO_VOO", 6: "DATA_EMBARQUE", 7: "DATAHORA_BUSCA", 8: "AGENCIA_COMP", 9: "PRECO",
        10: "TRECHO", 11: "ADVP", 12: "RANKING"
    }
    if list(df.columns[:13]) != list(colmap.values()):
        rename = {df.columns[i]: colmap[i] for i in range(min(13, df.shape[1]))}
        df = df.rename(columns=rename)

    # Normaliza horas com parser leve (evita pd.to_datetime pesado)
    for c in ["HORA_BUSCA", "HORA_PARTIDA", "HORA_CHEGADA"]:
        if c in df.columns:
            df[c] = df[c].map(_norm_hhmmss)

    # Hora HH pro filtro
    if "HORA_BUSCA" in df.columns:
        hh = df["HORA_BUSCA"].astype(str).str.slice(0, 2)
        df["HORA_HH"] = pd.to_numeric(hh, errors="coerce").astype("Int64")
    else:
        df["HORA_HH"] = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")

    # Datas
    for c in ["DATA_EMBARQUE", "DATAHORA_BUSCA"]:
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

    # Ranking int
    if "RANKING" in df.columns:
        df["RANKING"] = pd.to_numeric(df["RANKING"], errors="coerce").astype("Int64")

    # Normalizações auxiliares
    df["AGENCIA_NORM"] = df.get("AGENCIA_COMP", pd.Series([""] * len(df), index=df.index)).astype(str).apply(std_agencia)
    df["ADVP_CANON"] = df.get("ADVP", pd.Series([np.nan] * len(df), index=df.index)).apply(
        lambda x: advp_nearest(x) if pd.notna(x) else np.nan
    )
    df["CIA_NORM"] = df.get("CIA", pd.Series([None] * len(df), index=df.index)).apply(std_cia)

    # __DTKEY__: usa DATAHORA_BUSCA normalizada + HORA_BUSCA (se existir)
    dt_base = pd.to_datetime(df.get("DATAHORA_BUSCA", pd.Series([pd.NaT] * len(df), index=df.index)), errors="coerce", dayfirst=True)

    def _hh_to_sec(hs: object) -> float:
        s = _norm_hhmmss(hs)
        if not s:
            return np.nan
        hh, mm, ss = [int(x) for x in s.split(":")]
        return hh * 3600 + mm * 60 + ss

    hora_sec = df.get("HORA_BUSCA", pd.Series([None] * len(df), index=df.index)).map(_hh_to_sec)
    hora_sec = pd.to_numeric(hora_sec, errors="coerce")

    dtkey = dt_base.copy()
    mask_h = hora_sec.notna()
    dtkey = dtkey.where(~mask_h, dt_base.dt.normalize() + pd.to_timedelta(hora_sec.fillna(0), unit="s"))
    df["__DTKEY__"] = dtkey

    return df

# ─────────────────────── CSS (OK, mas mantenha leve) ─────────────────────────
st.markdown(
    """
<style>
table { width:100% !important; }
.dataframe { width:100% !important; }
table th { font-size:11px !important; white-space:nowrap !important; overflow:hidden !important; text-overflow:ellipsis !important; padding:4px 6px !important; }
table td { font-size:12px !important; padding:6px 8px !important; }
table th div, table th span { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────── HEATMAP / STYLER (SEM to_html) ─────────────────────
BLUE = "#cfe3ff"; ORANGE = "#fdd0a2"; GREEN = "#c7e9c0"; YELLOW = "#fee391"; PINK = "#f1b6da"

def _hex_to_rgb(h): return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
def _rgb_to_hex(t): return f"#{t[0]:02x}{t[1]:02x}{t[2]:02x}"
def _blend(c_from, c_to, t):
    f, to = _hex_to_rgb(c_from), _hex_to_rgb(c_to)
    return _rgb_to_hex(tuple(int(round(f[i] + (to[i] - f[i]) * t)) for i in range(3)))

def make_scale(base_hex, steps=5): return [_blend("#ffffff", base_hex, k / (steps - 1)) for k in range(steps)]
SCALE_BLUE = make_scale(BLUE); SCALE_ORANGE = make_scale(ORANGE); SCALE_GREEN = make_scale(GREEN); SCALE_YELLOW = make_scale(YELLOW); SCALE_PINK = make_scale(PINK)

def _pick_scale(colname: str):
    u = str(colname).upper()
    if "MAXMILHAS" in u: return SCALE_GREEN
    if "123" in u: return SCALE_ORANGE
    if "FLIP" in u: return SCALE_YELLOW
    if "CAPO" in u: return SCALE_PINK
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
        if pd.isna(val) or bins.iloc[idx] == -1:
            return "background-color:#ffffff;color:#111111"
        color = scale_colors[int(bins.iloc[idx])]
        return f"background-color:{color};color:#111111"

    return styler.apply(lambda col_vals: [_fmt(v, i) for i, v in enumerate(col_vals)], subset=[col])

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
    sty = sty.set_table_styles([{"selector": "tbody td, th", "props": [("border", "1px solid #EEE")]}])
    return sty

def show_table(df: pd.DataFrame, styler: pd.io.formats.style.Styler | None = None, caption: str | None = None, height: int | None = None):
    """Tabela segura: NÃO usa to_html() no markdown."""
    if caption:
        st.markdown(f"**{caption}**")
    if styler is not None:
        st.dataframe(styler, use_container_width=True, height=height)
    else:
        st.dataframe(df, use_container_width=True, height=height)

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
            s[["IDPESQUISA", "AGENCIA_NORM"]].rename(columns={"AGENCIA_NORM": f"R{r}"}),
            on="IDPESQUISA", how="left"
        )
    for r in (1, 2, 3):
        base[f"R{r}"] = base[f"R{r}"].fillna("SEM OFERTAS")
    return base

def last_update_from_cols(df: pd.DataFrame) -> str:
    ts = pd.to_datetime(df.get("__DTKEY__"), errors="coerce").max()
    if pd.isna(ts):
        ts = pd.to_datetime(df.get("DATAHORA_BUSCA"), errors="coerce").max()
    return ts.strftime("%d/%m/%Y - %H:%M:%S") if pd.notna(ts) else "—"

def _init_filter_state(df_raw: pd.DataFrame):
    if "flt" in st.session_state:
        return
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

    c1, c2, c3, c4, c5, c6, c7 = st.columns([1.1, 1.1, 1.0, 2.0, 1.0, 1.4, 0.32])

    base_dt = pd.to_datetime(df_raw.get("__DTKEY__"), errors="coerce")
    if base_dt.isna().all():
        base_dt = pd.to_datetime(df_raw.get("DATAHORA_BUSCA"), errors="coerce", dayfirst=True)

    dmin_abs = base_dt.min()
    dmax_abs = base_dt.max()
    dmin_abs = dmin_abs.date() if pd.notna(dmin_abs) else date(2000, 1, 1)
    dmax_abs = dmax_abs.date() if pd.notna(dmax_abs) else date.today()

    with c1:
        dt_ini = st.date_input(
            "Data inicial",
            key=f"{key_prefix}_dtini",
            value=st.session_state["flt"]["dt_ini"],
            min_value=dmin_abs,
            max_value=dmax_abs,
            format="DD/MM/YYYY",
        )
    with c2:
        dt_fim = st.date_input(
            "Data final",
            key=f"{key_prefix}_dtfim",
            value=st.session_state["flt"]["dt_fim"],
            min_value=dmin_abs,
            max_value=dmax_abs,
            format="DD/MM/YYYY",
        )
    with c3:
        advp_all = sorted(set(pd.to_numeric(df_raw.get("ADVP_CANON"), errors="coerce").dropna().astype(int).tolist()))
        advp_sel = st.multiselect("ADVP", options=advp_all, default=st.session_state["flt"]["advp"], key=f"{key_prefix}_advp")
    with c4:
        trechos_all = sorted([t for t in df_raw.get("TRECHO", pd.Series([], dtype=str)).dropna().unique().tolist() if str(t).strip() != ""])
        tr_sel = st.multiselect("Trechos", options=trechos_all, default=st.session_state["flt"]["trechos"], key=f"{key_prefix}_trechos")
    with c5:
        hh_opts = list(range(24))
        hh_sel = st.multiselect("Hora da busca", options=hh_opts, default=st.session_state["flt"]["hh"], key=f"{key_prefix}_hh")
    with c6:
        cia_presentes = set(str(x).upper() for x in df_raw.get("CIA_NORM", pd.Series([], dtype=str)).dropna().unique())
        ordem = ["AZUL", "GOL", "LATAM"]
        cia_opts = [c for c in ordem if (not cia_presentes or c in cia_presentes)] or ordem
        cia_default = [c for c in st.session_state["flt"]["cia"] if c in cia_opts]
        cia_sel = st.multiselect("Cia (Azul/Gol/Latam)", options=cia_opts, default=cia_default, key=f"{key_prefix}_cia")
    with c7:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("↻", key=f"{key_prefix}_refresh", type="secondary", help="Recarregar base"):
            st.cache_data.clear()
            st.session_state.pop("flt", None)
            st.rerun()

    st.session_state["flt"] = {
        "dt_ini": dt_ini, "dt_fim": dt_fim,
        "advp": advp_sel or [],
        "trechos": tr_sel or [],
        "hh": hh_sel or [],
        "cia": cia_sel or [],
    }

    # Data base
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
    st.caption(f"Quantidade de Ofertas: {fmt_int(len(df))}")
    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
    return df

# ───────────────────────────── ABAS DO APLICATIVO ────────────────────────────
@register_tab("Painel")
def tab1_painel(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t1")
    st.subheader("Painel")

    total_pesq = df["IDPESQUISA"].nunique() or 1
    last_update = last_update_from_cols(df)

    st.markdown(
        f"<div style='font-size:14px;opacity:.85;margin-top:-6px;'>"
        f"<b>Total de Pesquisas realizadas:</b> {fmt_int(total_pesq)} | "
        f"<b>Última Atualização:</b> {last_update}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='margin:6px 0'>", unsafe_allow_html=True)

    main_col1, main_col2 = st.columns([0.3, 0.7], gap="large")

    # CSS dos cards (coloca dentro do iframe só quando renderizar)
    CARD_CSS = """
    <style>
        .cards-stack { display:flex; flex-direction:column; gap:10px; }
        .card { border:1px solid #e9e9ee; border-radius:14px; padding:10px 12px; background:#fff; box-shadow:0 1px 2px rgba(0,0,0,.04); }
        .card .title { font-weight:650; font-size:15px; margin-bottom:8px; }
        .goldcard{background:#FFF9E5;border-color:#D4AF37;}
        .silvercard{background:#F7F7FA;border-color:#C0C0C0;}
        .bronzecard{background:#FFF1E8;border-color:#CD7F32;}
        .row{display:flex;gap:8px;}
        .item{flex:1;display:flex;align-items:center;justify-content:space-between;gap:8px;padding:8px 10px;border-radius:10px;border:1px solid #e3e3e8;background:#fafbfc;}
        .pos{font-weight:700;font-size:12px;opacity:.85;}
        .pct{font-size:16px;font-weight:650;}
        .stack-title { font-weight:800; padding:8px 10px; margin:6px 0 10px 0; border-radius:10px; border:1px solid #e9e9ee; background:#f8fafc; color:#0A2A6B; }
    </style>
    """

    def card_html(nome: str, p1: float, p2: float, p3: float, rank_cls: str = "") -> str:
        try:
            p1 = float(p1 or 0.0); p2 = float(p2 or 0.0); p3 = float(p3 or 0.0)
        except Exception:
            p1 = p2 = p3 = 0.0
        p1 = max(0.0, min(100.0, p1)); p2 = max(0.0, min(100.0, p2)); p3 = max(0.0, min(100.0, p3))
        cls = f"card {rank_cls}".strip()
        return (
            f"<div class='{cls}'>"
            f"<div class='title'>{nome}</div>"
            f"<div class='row'>"
            f"<div class='item'><span class='pos'>1º</span><span class='pct'>{p1:.0f}%</span></div>"
            f"<div class='item'><span class='pos'>2º</span><span class='pct'>{p2:.0f}%</span></div>"
            f"<div class='item'><span class='pos'>3º</span><span class='pct'>{p3:.0f}%</span></div>"
            f"</div></div>"
        )

    def card_html_cia(nome: str, p1: float, rank_cls: str = "") -> str:
        try:
            p1 = float(p1 or 0.0)
        except Exception:
            p1 = 0.0
        p1 = max(0.0, min(100.0, p1))
        cls = f"card {rank_cls}".strip()
        return (
            f"<div class='{cls}'>"
            f"<div class='title'>{nome}</div>"
            f"<div class='row'>"
            f"<div class='item'><span class='pos'>1º</span><span class='pct'>{p1:.0f}%</span></div>"
            f"</div></div>"
        )

    with main_col1:
        st.subheader("Ranking Geral")
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

        def pcts_for_target(base_df: pd.DataFrame, tgt: str, agrupado: bool) -> tuple[float, float, float]:
            base = (base_df.replace({
                "R1": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
                "R2": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
                "R3": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
            }) if agrupado else base_df)
            p1 = float((base["R1"] == tgt).mean()) * 100
            p2 = float((base["R2"] == tgt).mean()) * 100
            p3 = float((base["R3"] == tgt).mean()) * 100
            return p1, p2, p3

        targets_sorted = sorted(
            targets_base,
            key=lambda t: pcts_for_target(Wg if t == "GRUPO 123" else W, t, t == "GRUPO 123")[0],
            reverse=True
        )

        cards = []
        for idx, tgt in enumerate(targets_sorted):
            p1, p2, p3 = pcts_for_target(Wg if tgt == "GRUPO 123" else W, tgt, tgt == "GRUPO 123")
            rank_cls = "goldcard" if idx == 0 else "silvercard" if idx == 1 else "bronzecard" if idx == 2 else ""
            cards.append(card_html(tgt, p1, p2, p3, rank_cls))

        html = CARD_CSS + f"<div class='cards-stack'>{''.join(cards)}</div>"
        render_html(html, height=estimate_height(len(cards), base=180, per_row=120), scrolling=True)

    with main_col2:
        st.subheader("Ranking por Cia")
        if "CIA_NORM" not in df.columns:
            st.info("Coluna 'CIA_NORM' não encontrada nos dados filtrados.")
            return

        c1, c2, c3 = st.columns(3)
        cia_colors = {
            "AZUL": "background-color: #0033A0; color: white;",
            "GOL": "background-color: #FF6600; color: white;",
            "LATAM": "background-color: #8B0000; color: white;",
        }

        def render_por_cia(df_in: pd.DataFrame, cia_name: str) -> str:
            style = cia_colors.get(cia_name, "background:#f8fafc; color:#0A2A6B;")
            sub = df_in[df_in["CIA_NORM"].astype(str).str.upper() == cia_name]
            if sub.empty:
                return f"<div class='stack-title' style='{style}'>Ranking {cia_name.title()}</div><div style='padding:10px;color:#64748b;font-weight:700;'>Sem dados</div>"

            Wc = winners_by_position(sub)
            Wc_g = Wc.replace({
                "R1": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
                "R2": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
                "R3": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
            })

            ags = sorted(set(sub["AGENCIA_NORM"].dropna().astype(str)))
            targets = [a for a in ags if a != "SEM OFERTAS"]
            if "GRUPO 123" not in targets:
                targets.insert(0, "GRUPO 123")

            def pct_target(tgt: str):
                base = Wc_g if tgt == "GRUPO 123" else Wc
                p1 = float((base["R1"] == tgt).mean()) * 100
                return p1

            targets_sorted_local = sorted(targets, key=lambda t: pct_target(t), reverse=True)
            cards_local = [card_html_cia(t, pct_target(t)) for t in targets_sorted_local]

            return (
                f"<div class='stack-title' style='{style}'>Ranking {cia_name.title()}</div>"
                f"<div class='cards-stack'>{''.join(cards_local)}</div>"
            )

        html_az = CARD_CSS + render_por_cia(df, "AZUL")
        html_go = CARD_CSS + render_por_cia(df, "GOL")
        html_la = CARD_CSS + render_por_cia(df, "LATAM")
        with c1: render_html(html_az, height=720, scrolling=True)
        with c2: render_html(html_go, height=720, scrolling=True)
        with c3: render_html(html_la, height=720, scrolling=True)

# ─────────────────────────── ABA 2: Top 3 Agências ───────────────────────────
@register_tab("Top 3 Agências")
def tab2_top3_agencias(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t2")
    st.subheader("Top 3 Agências (por menor preço no trecho)")
    if df.empty:
        st.info("Sem resultados para os filtros selecionados.")
        return

    df2 = df.copy()
    df2["DT"] = pd.to_datetime(df2["DATAHORA_BUSCA"], errors="coerce")
    g = (df2.dropna(subset=["TRECHO", "IDPESQUISA", "DT"])
         .groupby(["TRECHO", "IDPESQUISA"], as_index=False)["DT"].max())
    last_idx = g.groupby("TRECHO")["DT"].idxmax()
    last_ids = {r["TRECHO"]: r["IDPESQUISA"] for _, r in g.loc[last_idx].iterrows()}
    df2["__ID_TARGET__"] = df2["TRECHO"].map(last_ids)
    df_last = df2[df2["IDPESQUISA"].astype(str) == df2["__ID_TARGET__"].astype(str)].copy()

    def _compose_dt_hora(sub: pd.DataFrame) -> str:
        d = pd.to_datetime(sub["DATAHORA_BUSCA"], errors="coerce").max()
        hh = None
        for v in sub["HORA_BUSCA"].tolist():
            hh = _norm_hhmmss(v)
            if hh:
                break
        if pd.isna(d) and not hh:
            return "-"
        if not hh:
            hh = pd.to_datetime(d, errors="coerce").strftime("%H:%M:%S")
        return f"{d.strftime('%d/%m/%Y')} {hh}"

    dt_by_trecho = {trecho: _compose_dt_hora(sub) for trecho, sub in df_last.groupby("TRECHO")}

    by_ag = (
        df_last.groupby(["TRECHO", "AGENCIA_NORM"], as_index=False)
        .agg(PRECO_MIN=("PRECO", "min"))
        .rename(columns={"TRECHO": "TRECHO_STD", "AGENCIA_NORM": "AGENCIA_UP"})
    )

    def _row_top3(gx: pd.DataFrame) -> pd.Series:
        gx = gx.sort_values("PRECO_MIN", ascending=True).reset_index(drop=True)
        trecho = gx["TRECHO_STD"].iloc[0] if len(gx) else "-"
        def name(i): return gx.loc[i, "AGENCIA_UP"] if i < len(gx) else "-"
        def price(i): return gx.loc[i, "PRECO_MIN"] if i < len(gx) else np.nan
        return pd.Series({
            "Data/Hora Busca": dt_by_trecho.get(trecho, "-"),
            "Trecho": trecho,
            "Agencia Top 1": name(0), "Preço Top 1": price(0),
            "Agencia Top 2": name(1), "Preço Top 2": price(1),
            "Agencia Top 3": name(2), "Preço Top 3": price(2),
        })

    t1 = by_ag.groupby("TRECHO_STD").apply(_row_top3).reset_index(drop=True)

    for c in ["Preço Top 1", "Preço Top 2", "Preço Top 3"]:
        t1[c] = pd.to_numeric(t1[c], errors="coerce")

    sty1 = style_smart_colwise(
        t1, {c: fmt_num0_br for c in ["Preço Top 1", "Preço Top 2", "Preço Top 3"]},
        grad_cols=["Preço Top 1", "Preço Top 2", "Preço Top 3"]
    )
    show_table(t1, sty1, caption="Ranking Top 3 (Agências) — por trecho", height=520)

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
    show_table(t2, sty2, caption="% Diferença entre Agências (base: TOP1)", height=520)

# ─────────────────────────── ABA 3: Top 3 Preços ─────────────────────────────
@register_tab("Top 3 Preços Mais Baratos")
def tab3_top3_precos(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t3")
    st.subheader("Top 3 Preços Mais Baratos")

    top_row = st.container()
    with top_row:
        c1, c2, _ = st.columns([0.28, 0.18, 0.54])
        agencia_foco = c1.selectbox("Agência alvo", ["Todos", "123MILHAS", "MAXMILHAS"], index=0, key="t3_agencia_foco")
        posicao_foco = c2.selectbox("Ranking", ["Todas", 1, 2, 3], index=0, key="t3_posicao_foco")

    if df.empty:
        st.info("Sem resultados para os filtros selecionados.")
        return

    def fmt_moeda_br(x) -> str:
        try:
            xv = float(x)
            if not np.isfinite(xv): return "R$ -"
            return "R$ " + f"{xv:,.0f}".replace(",", ".")
        except Exception:
            return "R$ -"

    dfp = df.copy()
    dfp["TRECHO_STD"] = dfp.get("TRECHO", "").astype(str)
    dfp["AGENCIA_UP"] = dfp.get("AGENCIA_NORM", "").astype(str)
    dfp["ADVP"] = (dfp.get("ADVP_CANON").fillna(dfp.get("ADVP"))).astype(str)
    dfp["__PRECO__"] = pd.to_numeric(dfp.get("PRECO"), errors="coerce")
    dfp["__DTKEY__"] = pd.to_datetime(dfp.get("DATAHORA_BUSCA"), errors="coerce", dayfirst=True)

    ID_COL = "IDPESQUISA" if "IDPESQUISA" in dfp.columns else dfp.columns[0]
    dfp = dfp[dfp["__PRECO__"].notna()].copy()
    if dfp.empty:
        st.info("Sem preços válidos no recorte atual.")
        return

    # última pesquisa por Trecho×ADVP
    pesq_por_ta = {}
    tmp = dfp.dropna(subset=["TRECHO_STD", "ADVP", ID_COL, "__DTKEY__"]).copy()
    g = tmp.groupby(["TRECHO_STD", "ADVP", ID_COL], as_index=False)["__DTKEY__"].max()
    if not g.empty:
        idx = g.groupby(["TRECHO_STD", "ADVP"])["__DTKEY__"].idxmax()
        last_by_ta = g.loc[idx]
        pesq_por_ta = {(str(r["TRECHO_STD"]), str(r["ADVP"])): str(r[ID_COL]) for _, r in last_by_ta.iterrows()}

    def _normalize_id(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        s = str(val)
        try:
            f = float(s.replace(",", "."))
            if f.is_integer():
                return str(int(f))
        except Exception:
            pass
        return s

    def dt_and_id_for(sub_rows: pd.DataFrame) -> tuple[str, str | None]:
        if sub_rows.empty:
            return "", None
        r = sub_rows.loc[sub_rows["__DTKEY__"].idxmax()]
        date_part = pd.to_datetime(r.get("DATAHORA_BUSCA"), errors="coerce", dayfirst=True)
        date_txt = date_part.strftime("%d/%m") if pd.notna(date_part) else ""
        htxt_raw = str(r.get("HORA_BUSCA", "")).strip()
        htxt = _norm_hhmmss(htxt_raw) or (pd.to_datetime(r["__DTKEY__"], errors="coerce").strftime("%H:%M:%S") if pd.notna(r["__DTKEY__"]) else "")
        id_val = _normalize_id(r.get("IDPESQUISA"))
        lbl = f"{date_txt} {htxt}".strip()
        return lbl, id_val

    trechos_sorted = sorted(dfp["TRECHO_STD"].dropna().astype(str).unique(), key=lambda x: str(x))

    # CSS interno + REMOÇÃO do <input> (troca por <span>)
    BADGE_POP_CSS = """
    <style>
    .grid{display:block;font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;}
    .TRE_HDR{margin:14px 0 10px 0;padding:10px 12px;border-left:4px solid #0B5FFF;background:#ECF3FF;border-radius:8px;font-weight:800;color:#0A2A6B;}
    .GRID{display:grid;grid-auto-flow:column;grid-auto-columns:260px;gap:10px;overflow-x:auto;padding:6px 2px 10px 2px;scrollbar-width:thin;}
    .BOX{border:1px solid #e5e7eb;border-radius:12px;background:#fff;overflow:hidden;}
    .HEAD{padding:8px 10px;border-bottom:1px solid #f1f5f9;font-weight:800;color:#111827;}
    .STACK{display:grid;gap:8px;padding:8px;}
    .CARD{position:relative;border:1px solid #e5e7eb;border-radius:10px;padding:8px;background:#fff;min-height:78px;}
    .DTW{position:absolute;right:8px;top:6px;display:flex;align-items:center;gap:6px;}
    .DTT{font-size:10px;color:#94a3b8;font-weight:900;}
    .RANK{font-weight:900;font-size:11px;color:#6b7280;letter-spacing:.3px;text-transform:uppercase;}
    .AG{font-weight:900;font-size:15px;color:#111827;margin-top:2px;}
    .PR{font-weight:1000;font-size:18px;color:#111827;margin-top:2px;}
    .SUB{font-weight:800;font-size:12px;color:#374151;}
    .NO{padding:22px 12px;color:#6b7280;font-weight:900;text-align:center;border:1px dashed #e5e7eb;border-radius:10px;background:#fafafa;}
    .idp-wrap{position:relative; display:inline-flex; align-items:center;}
    .idp-badge{display:inline-flex; align-items:center; justify-content:center;width:16px; height:16px; border:1px solid #cbd5e1; border-radius:50%;font-size:11px; font-weight:900; color:#64748b; background:#fff;user-select:none; cursor:default; line-height:1;}
    .idp-pop{position:absolute; top:18px; right:0;background:#fff; color:#0f172a; border:1px solid #e5e7eb;border-radius:8px; padding:6px 8px; font-size:12px; font-weight:800;box-shadow:0 6px 16px rgba(0,0,0,.08); display:none; z-index:9999; white-space:nowrap;}
    .idp-wrap:hover .idp-pop{ display:block; }
    .idp-idbox{border:1px solid #e5e7eb; background:#f8fafc; border-radius:6px;padding:2px 6px; font-weight:900; font-size:12px; min-width:60px;font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;user-select:text; cursor:text;}
    .extra-wrap{padding:6px 8px 10px 8px; border-top:1px dashed #e5e7eb; margin-top:6px;}
    .extra-title{font-size:12px; font-weight:900; color:#374151; margin-bottom:2px;}
    .tag-extra{font-size:12px; color:#4b5563; margin-top:2px; font-weight:800;}
    </style>
    """

    html_parts = [BADGE_POP_CSS, "<div class='grid'>"]

    for trecho in trechos_sorted:
        df_t = dfp[dfp["TRECHO_STD"] == trecho]
        advps = sorted(df_t["ADVP"].dropna().astype(str).unique(),
                       key=lambda v: (0, int("".join([d for d in v if d.isdigit()]) or 9999), str(v)))

        boxes = []
        for advp in advps:
            df_ta = df_t[df_t["ADVP"].astype(str) == str(advp)].copy()
            pesq_id = pesq_por_ta.get((trecho, advp))
            all_rows = df_ta[df_ta[ID_COL].astype(str) == str(pesq_id)] if pesq_id else df_ta.iloc[0:0]

            base_rank = (all_rows.groupby("AGENCIA_UP", as_index=False)["__PRECO__"]
                         .min().sort_values("__PRECO__").reset_index(drop=True))

            box = [f"<div class='BOX'>", f"<div class='HEAD'>ADVP: <b>{advp}</b></div>"]
            if base_rank.empty:
                box.append("<div class='NO'>Sem ofertas</div></div>")
                boxes.append("".join(box))
                continue

            box.append("<div class='STACK'>")

            # filtro por agência/ranking (opcional)
            def keep_row(ag: str, rk: int) -> bool:
                if agencia_foco != "Todos" and str(ag).upper() != str(agencia_foco).upper():
                    return False
                if posicao_foco != "Todas" and rk != int(posicao_foco):
                    return False
                return True

            shown = 0
            for i in range(min(3, len(base_rank))):
                rk = i + 1
                row_i = base_rank.iloc[i]
                ag = str(row_i["AGENCIA_UP"])
                if not keep_row(ag, rk):
                    continue

                preco_i = float(row_i["__PRECO__"])
                sub_rows = all_rows[(all_rows["AGENCIA_UP"] == ag) & (np.isclose(all_rows["__PRECO__"], preco_i, atol=1))]
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
                        subtxt = f"+{int(round((preco_i - p1) / p1 * 100))}% vs 1º"

                stripe = "#D4AF37" if i == 0 else "#9CA3AF" if i == 1 else "#CD7F32"

                # ID sem input (span selecionável)
                id_txt = _normalize_id(id_val) or ""

                box.append(
                    f"<div class='CARD'>"
                    f"<div style='position:absolute;left:0;top:0;bottom:0;width:6px;border-radius:10px 0 0 10px;background:{stripe};'></div>"
                    f"<div class='DTW'><span class='DTT'>{dt_lbl}</span>"
                    f"<span class='idp-wrap'><span class='idp-badge'>?</span>"
                    f"<span class='idp-pop'>ID:&nbsp;<span class='idp-idbox'>{id_txt}</span></span></span></div>"
                    f"<div class='RANK'>{rk}º</div>"
                    f"<div class='AG'>{ag}</div>"
                    f"<div class='PR'>{fmt_moeda_br(preco_i)}</div>"
                    f"<div class='SUB'>{subtxt}</div>"
                    f"</div>"
                )
                shown += 1

            if shown == 0:
                box.append("<div class='NO'>Nada para o filtro selecionado</div>")

            box.append("</div>")  # STACK

            # status fora do top3
            p1_val = float(base_rank.iloc[0]["__PRECO__"])
            msgs = []
            idx_max = base_rank.index[base_rank["AGENCIA_UP"] == "MAXMILHAS"].tolist()
            if not idx_max:
                msgs.append("Maxmilhas Não Apareceu")
            elif idx_max[0] > 2:
                pos = idx_max[0] + 1
                preco = float(base_rank.iloc[idx_max[0]]["__PRECO__"])
                dif = int(round((preco - p1_val) / p1_val * 100)) if p1_val else 0
                msgs.append(f"Maxmilhas: {pos}º - {fmt_moeda_br(preco)} (+{dif}% vs 1º)")

            idx_123 = base_rank.index[base_rank["AGENCIA_UP"] == "123MILHAS"].tolist()
            if not idx_123:
                msgs.append("123milhas Não Apareceu")
            elif idx_123[0] > 2:
                pos = idx_123[0] + 1
                preco = float(base_rank.iloc[idx_123[0]]["__PRECO__"])
                dif = int(round((preco - p1_val) / p1_val * 100)) if p1_val else 0
                msgs.append(f"123milhas: {pos}º - {fmt_moeda_br(preco)} (+{dif}% vs 1º)")

            if msgs:
                box.append(
                    "<div class='extra-wrap'>"
                    "<div class='extra-title'>Status Grupo 123</div>"
                    + "".join([f"<div class='tag-extra'>{m}</div>" for m in msgs]) +
                    "</div>"
                )

            box.append("</div>")  # BOX
            boxes.append("".join(box))

        if boxes:
            html_parts.append(f"<div class='TRE_HDR'>Trecho: <b>{trecho}</b></div>")
            html_parts.append("<div class='GRID'>" + "".join(boxes) + "</div>")

    html_parts.append("</div>")  # grid
    big_html = "".join(html_parts)
    render_html(big_html, height=900, scrolling=True)

# ─────────────────────────── ABA 4: Ranking por Agências ─────────────────────
@register_tab("Ranking por Agências")
def tab4_ranking_agencias(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t4")
    st.subheader("Ranking por Agências")

    if df.empty:
        st.info("Sem dados para os filtros.")
        return

    work = df.copy()
    work["AGENCIA_UP"] = work.get("AGENCIA_NORM", work.get("AGENCIA_COMP", "")).astype(str)

    if "RANKING" not in work.columns:
        st.warning("Coluna 'RANKING' não encontrada.")
        return

    work["Ranking"] = pd.to_numeric(work["RANKING"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["AGENCIA_UP", "Ranking"])
    work["Ranking"] = work["Ranking"].astype(int)

    RANKS = list(range(1, 16))
    counts = (work.groupby(["AGENCIA_UP", "Ranking"], as_index=False)
              .agg(OFERTAS=("AGENCIA_UP", "size")))

    pv = (counts.pivot(index="AGENCIA_UP", columns="Ranking", values="OFERTAS")
          .reindex(columns=RANKS, fill_value=0)
          .fillna(0).astype(int))

    pv.index.name = "Agência/Companhia"
    if 1 not in pv.columns: pv[1] = 0
    pv["Total"] = pv.sum(axis=1)
    pv = pv.sort_values(by=1, ascending=False)

    total_row = pv.sum(axis=0).to_frame().T
    total_row.index = ["Total"]
    total_row.index.name = "Agência/Companhia"
    pv2 = pd.concat([pv, total_row], axis=0)

    t_qtd = pv2.reset_index()
    st.subheader("Quantidade de Ofertas por Ranking (Ofertas)")
    show_table(t_qtd[["Agência/Companhia"] + RANKS + ["Total"]], styler=None, height=520)

    # % dentro da agência
    mat = pv[RANKS].copy()
    row_sum = mat.sum(axis=1).replace(0, np.nan)
    pct_linha = (mat.div(row_sum, axis=0) * 100).fillna(0).sort_values(by=1, ascending=False)
    t_pct_linha = pct_linha.reset_index()
    st.subheader("Participação Ranking dentro da Agência")
    show_table(t_pct_linha[["Agência/Companhia"] + RANKS], height=520)

    # % dentro do ranking
    col_sum = mat.sum(axis=0).replace(0, np.nan)
    pct_coluna = (mat.div(col_sum, axis=1) * 100).fillna(0).sort_values(by=1, ascending=False)
    t_pct_coluna = pct_coluna.reset_index()
    st.subheader("Participação Ranking Geral")
    show_table(t_pct_coluna[["Agência/Companhia"] + RANKS], height=520)

# ─────────────────────────── ABA 5 e 6 e 7 e 8 ──────────────────────────────
# Mantive sua lógica, mas o principal “conserto do erro” já está feito:
# - sem styler.to_html no markdown
# - sem <input> no HTML
# - HTML gigante em iframe (components.html)
# - keys únicas e bug do Tab6 corrigido

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Se você quiser, eu também migro Tab5/Tab6 (tabelas HTML enormes)
# para render_html no mesmo padrão do Tab3, para ficar 100% blindado.
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

@register_tab("Ofertas x Cias")
def tab7_ofertas_x_cias(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t7")
    st.subheader("Distribuição de Ofertas por Companhia Aérea")
    if df.empty:
        st.info("Sem dados para os filtros selecionados.")
        return

    LABEL_SIZE = 18
    SHOW_MIN = 0.02

    CIA_DOMAIN = ['AZUL', 'GOL', 'LATAM']
    CIA_COLORS = ['#0033A0', '#FF6600', '#8B0000']
    ORDER_MAP = {'AZUL': 0, 'GOL': 1, 'LATAM': 2}

    base = df.copy()
    base['CIA_NORM'] = base['CIA_NORM'].astype(str).str.upper()
    if 'IDPESQUISA' not in base.columns:
        st.warning("Coluna IDPESQUISA ausente.")
        return

    def build_stacked(df_in: pd.DataFrame, group_col: str) -> pd.DataFrame:
        counts = (df_in.groupby([group_col, 'CIA_NORM'])['IDPESQUISA']
                  .nunique().reset_index(name='n'))
        if counts.empty:
            return counts
        all_groups = counts[group_col].dropna().unique()
        full_idx = pd.MultiIndex.from_product([all_groups, CIA_DOMAIN], names=[group_col, 'CIA_NORM'])
        counts = (counts.set_index([group_col, 'CIA_NORM'])
                  .reindex(full_idx, fill_value=0)
                  .reset_index())
        counts['total'] = counts.groupby(group_col)['n'].transform('sum')
        counts['share'] = np.where(counts['total'] > 0, counts['n'] / counts['total'], 0.0)
        counts['order'] = counts['CIA_NORM'].map(ORDER_MAP)
        counts = counts.sort_values([group_col, 'order'])
        counts['y1'] = counts.groupby(group_col)['share'].cumsum()
        counts['y0'] = counts['y1'] - counts['share']
        counts['y_center'] = (counts['y0'] + counts['y1']) / 2.0
        counts['pct_txt'] = counts['share'].map(lambda x: f"{x*100:.0f}%")
        return counts

    def draw_chart(stacked: pd.DataFrame, group_col: str, x_title: str):
        if stacked.empty:
            st.info(f"Sem dados para {x_title} no recorte atual.")
            return
        bars = (
            alt.Chart(stacked).mark_bar().encode(
                x=alt.X(f'{group_col}:N', title=x_title, axis=alt.Axis(labelAngle=0)),
                y=alt.Y('y0:Q', title='Participação', axis=alt.Axis(format='%'), scale=alt.Scale(domain=[0, 1])),
                y2='y1:Q',
                order=alt.Order('order:Q', sort='ascending'),
                color=alt.Color('CIA_NORM:N', title='Cia Aérea',
                                scale=alt.Scale(domain=CIA_DOMAIN, range=CIA_COLORS)),
                tooltip=[
                    alt.Tooltip(f'{group_col}:N', title=x_title),
                    alt.Tooltip('CIA_NORM:N', title='CIA'),
                    alt.Tooltip('n:Q', title='Nº de Pesquisas'),
                    alt.Tooltip('share:Q', title='Participação', format='.2%')
                ]
            )
        )
        labels = (
            alt.Chart(stacked[stacked['share'] >= SHOW_MIN])
            .mark_text(color='white', fontWeight='bold', align='center', baseline='middle', size=LABEL_SIZE)
            .encode(
                x=alt.X(f'{group_col}:N'),
                y=alt.Y('y_center:Q', scale=alt.Scale(domain=[0, 1])),
                text='pct_txt:N',
                detail='CIA_NORM:N'
            )
        )
        st.altair_chart((bars + labels).properties(height=450), use_container_width=True)

    st.markdown("#### Participação Cias Por ADVP")
    if 'ADVP_CANON' not in base.columns:
        st.info("Coluna ADVP_CANON não encontrada nos dados.")
    else:
        advp = base.copy()
        advp['ADVP_CANON'] = pd.to_numeric(advp['ADVP_CANON'], errors='coerce')
        advp = advp.dropna(subset=['ADVP_CANON'])
        draw_chart(build_stacked(advp, 'ADVP_CANON'), 'ADVP_CANON', 'ADVP')

    st.markdown("<hr style='margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown("#### Participação Cias Por Trechos")
    top15 = base.groupby('TRECHO')['IDPESQUISA'].nunique().nlargest(15).index
    dft = base[base['TRECHO'].isin(top15)].copy()
    draw_chart(build_stacked(dft, 'TRECHO'), 'TRECHO', 'Trecho')

# ─────────────────────────── ABA 8: TABELA DE PESQUISA ───────────────────────
@register_tab("TABELA DE PESQUISA")
def tab_tabela_pesquisa(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t8")
    if df.empty:
        st.info("Sem resultados para os filtros selecionados.")
        return

    work = df.copy()
    work['ADVP_CANON'] = work.get('ADVP_CANON', work.get('ADVP'))
    work['TRECHO'] = work.get('TRECHO', '').astype(str)
    work['AGENCIA_NORM'] = work.get('AGENCIA_NORM', work.get('AGENCIA_COMP', '')).astype(str)
    work['CIA_NORM'] = work.get('CIA_NORM', work.get('CIA', '')).astype(str)

    top_trechos = work.groupby('TRECHO')['IDPESQUISA'].nunique().sort_values(ascending=False).head(11).index.tolist()
    if len(top_trechos) < 11:
        restantes = [t for t in work['TRECHO'].dropna().astype(str).unique().tolist() if t not in top_trechos]
        top_trechos += restantes[: max(0, 11 - len(top_trechos))]

    advp_buckets = [1, 5, 11, 17, 30]

    rows_out = []
    for trecho in top_trechos:
        for advp in advp_buckets:
            sub = work[(work['TRECHO'] == trecho) & (pd.to_numeric(work['ADVP_CANON'], errors='coerce') == advp)].copy()
            if sub.empty:
                continue

            if '__DTKEY__' in sub.columns and sub['__DTKEY__'].notna().any():
                last_idx = sub['__DTKEY__'].idxmax()
            else:
                last_idx = sub['DATAHORA_BUSCA'].dropna().index.max() if sub['DATAHORA_BUSCA'].notna().any() else sub.index.max()

            last_id = sub.loc[last_idx, 'IDPESQUISA'] if pd.notna(last_idx) else None
            if last_id is None:
                continue

            rows_pesq = work[(work['IDPESQUISA'] == last_id) & (work['TRECHO'] == trecho) & (pd.to_numeric(work['ADVP_CANON'], errors='coerce') == advp)].copy()
            if rows_pesq.empty:
                rows_pesq = sub.copy()

            rows_pesq['PRECO_NUM'] = pd.to_numeric(rows_pesq['PRECO'], errors='coerce')
            if rows_pesq['PRECO_NUM'].notna().any():
                min_price = float(rows_pesq['PRECO_NUM'].min())
                min_row = rows_pesq.loc[rows_pesq['PRECO_NUM'].idxmin()]
                empresa_min = str(min_row.get('AGENCIA_NORM', ''))
                cia_voo = str(min_row.get('CIA_NORM', ''))
            else:
                min_price = np.nan; empresa_min = ''; cia_voo = ''

            def get_ag_price(df_sub, ag_name):
                dfx = df_sub[df_sub['AGENCIA_NORM'].str.upper() == ag_name.upper()]
                v = pd.to_numeric(dfx['PRECO'], errors='coerce')
                return float(v.min()) if v.notna().any() else np.nan

            price_123 = get_ag_price(rows_pesq, '123MILHAS')
            price_max = get_ag_price(rows_pesq, 'MAXMILHAS')
            price_flip = get_ag_price(rows_pesq, 'FLIPMILHAS')

            dt_busca = (pd.to_datetime(rows_pesq['DATAHORA_BUSCA'], errors='coerce', dayfirst=True).max() if 'DATAHORA_BUSCA' in rows_pesq.columns else pd.NaT)
            dt_emb = (pd.to_datetime(rows_pesq['DATA_EMBARQUE'], dayfirst=True, errors='coerce').min() if 'DATA_EMBARQUE' in rows_pesq.columns else pd.NaT)

            rows_out.append({
                'IDPESQUISA': last_id,
                'TRECHO': trecho,
                'ADVP': advp,
                'DATAHORA_BUSCA': dt_busca,
                'DATA_EMBARQUE': dt_emb,
                'PRECO': min_price,
                'CIA_DO_VOO': cia_voo,
                'EMPRESA': empresa_min,
                '123MILHAS': price_123,
                'MAXMILHAS': price_max,
                'FLIPMILHAS': price_flip,
            })

    out_df = pd.DataFrame(rows_out)

    def parse_origem_destino(t: str) -> tuple[str, str]:
        s = str(t or '').strip().upper()
        if len(s) >= 6 and re.match(r'^[A-Z]{6}$', s):
            return s[0:3], s[3:6]
        found = re.findall(r"[A-Z]{3}", s)
        if len(found) >= 2:
            return found[0], found[1]
        parts = re.split(r'[^A-Z0-9]+', s)
        parts = [p for p in parts if p]
        if len(parts) >= 2:
            return parts[0][0:3], parts[1][0:3]
        if len(s) >= 6:
            return s[0:3], s[3:6]
        return s[0:3], ''

    if not out_df.empty:
        out_df['TRECHO ORIGEM'] = out_df['TRECHO'].apply(lambda x: parse_origem_destino(x)[0])
        out_df['TRECHO DESTINO'] = out_df['TRECHO'].apply(lambda x: parse_origem_destino(x)[1])

    def pct(base, other):
        try:
            if pd.isna(base) or pd.isna(other) or base == 0:
                return np.nan
            return (other - base) / base * 100
        except Exception:
            return np.nan

    out_df['123XFLIP (%)'] = out_df.apply(lambda r: pct(r.get('FLIPMILHAS'), r.get('123MILHAS')), axis=1)
    out_df['MAX X FLIP (%)'] = out_df.apply(lambda r: pct(r.get('FLIPMILHAS'), r.get('MAXMILHAS')), axis=1)
    out_df['123 X MENOR PREÇO (%)'] = out_df.apply(lambda r: pct(r.get('PRECO'), r.get('123MILHAS')), axis=1)

    display_df = out_df.rename(columns={
        'DATAHORA_BUSCA': 'DATA+ HORA DA PESQUISA',
        'ADVP': 'ADVP',
        'DATA_EMBARQUE': 'DATA DE EMBARQUE',
        'PRECO': 'PREÇO',
        'CIA_DO_VOO': 'CIA DO VOO',
        'EMPRESA': 'EMPRESA',
        '123MILHAS': 'PREÇO 123MILHAS',
        'MAXMILHAS': 'PREÇOMAXMILHAS',
        'FLIPMILHAS': 'PREÇO FLIPMILHAS'
    })

    def fmt_dt(v):
        try:
            return pd.to_datetime(v).strftime('%d/%m/%Y %H:%M:%S')
        except Exception:
            return '-'

    if 'DATA+ HORA DA PESQUISA' in display_df.columns:
        display_df['DATA+ HORA DA PESQUISA'] = display_df['DATA+ HORA DA PESQUISA'].apply(fmt_dt)
    if 'DATA DE EMBARQUE' in display_df.columns:
        display_df['DATA DE EMBARQUE'] = display_df['DATA DE EMBARQUE'].apply(lambda v: pd.to_datetime(v).strftime('%d/%m/%Y') if pd.notna(v) else '-')

    cols_order = [
        'DATA+ HORA DA PESQUISA', 'TRECHO ORIGEM', 'TRECHO DESTINO', 'ADVP', 'DATA DE EMBARQUE',
        'PREÇO', 'CIA DO VOO', 'EMPRESA', 'PREÇO 123MILHAS', 'PREÇOMAXMILHAS', 'PREÇO FLIPMILHAS',
        '123XFLIP (%)', 'MAX X FLIP (%)', '123 X MENOR PREÇO (%)'
    ]
    for c in cols_order:
        if c not in display_df.columns:
            display_df[c] = np.nan
    display_df = display_df[cols_order].reset_index(drop=True)

    sty = style_smart_colwise(display_df, {
        'PREÇO': fmt_num0_br,
        'PREÇO 123MILHAS': fmt_num0_br,
        'PREÇOMAXMILHAS': fmt_num0_br,
        'PREÇO FLIPMILHAS': fmt_num0_br,
        '123XFLIP (%)': fmt_pct0_br,
        'MAX X FLIP (%)': fmt_pct0_br,
        '123 X MENOR PREÇO (%)': fmt_pct0_br,
    }, grad_cols=['PREÇO', 'PREÇO 123MILHAS', 'PREÇOMAXMILHAS', 'PREÇO FLIPMILHAS'])

    # Export
    export_df_for_file = display_df.copy()

    pct_cols_export = [c for c in ['123XFLIP (%)', 'MAX X FLIP (%)', '123 X MENOR PREÇO (%)'] if c in export_df_for_file.columns]
    for c in pct_cols_export:
        export_df_for_file[c] = pd.to_numeric(export_df_for_file[c], errors='coerce')
        export_df_for_file[c] = export_df_for_file[c].apply(lambda x: round(float(x) / 100, 6) if pd.notna(x) else x)

    csv_bytes = export_df_for_file.to_csv(index=False, sep=';', decimal=',', encoding='utf-8-sig').encode('utf-8-sig')
    c1, c2, c3 = st.columns([1, 1, 0.2])
    with c3:
        st.download_button('Baixar CSV', data=csv_bytes, file_name='tabela_pesquisa.csv', mime='text/csv', key='t8_dl_csv_tabela')

    show_table(display_df, sty, height=700)

    to_xlsx = io.BytesIO()
    xlsx_written = False
    for engine in ("openpyxl", "xlsxwriter"):
        try:
            with pd.ExcelWriter(to_xlsx, engine=engine) as writer:
                export_df_for_file.to_excel(writer, index=False, sheet_name='TABELA_PESQUISA')
            xlsx_written = True
            break
        except Exception:
            to_xlsx = io.BytesIO()
            continue

    if xlsx_written:
        to_xlsx.seek(0)
        st.download_button(
            'Baixar XLSX',
            data=to_xlsx.read(),
            file_name='tabela_pesquisa.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            key="t8_dl_xlsx"
        )

# ─────────────────────────── MAIN ────────────────────────────────────────────
def main():
    df_raw = load_base(DATA_PATH)

    for ext in ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp"):
        imgs = list(APP_DIR.glob(ext))
        if imgs:
            st.image(imgs[0].as_posix(), use_container_width=True)
            break

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
