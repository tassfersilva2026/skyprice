from __future__ import annotations
from pathlib import Path
from datetime import date
from typing import Callable, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import re
import io

# ─────────────────────────── CONFIG DA PÁGINA ────────────────────────────────
st.set_page_config(page_title="Skyscanner — Painel", layout="wide", initial_sidebar_state="expanded")
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "OFERTASCONSOLIDADO_OFERTAS.parquet"

# ─────────────────────────── FUNÇÕES AUXILIARES BÁSICAS ───────────────────────────
def _norm_hhmmss(v: object) -> str | None:
    """Normaliza uma string de tempo para o formato HH:MM:SS."""
    s = str(v or "").strip()
    m = re.search(r"(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?", s)
    if not m:
        return None
    hh = max(0, min(23, int(m.group(1))))
    mm = max(0, min(59, int(m.group(2))))
    ss = max(0, min(59, int(m.group(3) or 0)))
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

def std_agencia(raw: str) -> str:
    """Padroniza nomes de agências para facilitar a análise."""
    ag = (raw or "").strip().upper()
    if ag == "BOOKINGCOM":
        return "BOOKING.COM"
    if ag == "KIWICOM":
        return "KIWI.COM"
    if ag.startswith("123MILHAS") or ag == "123":
        return "123MILHAS"
    if ag.startswith("MAXMILHAS") or ag == "MAX":
        return "MAXMILHAS"
    if ag.startswith("CAPOVIAGENS"):
        return "CAPOVIAGENS"
    if ag.startswith("FLIPMILHAS"):
        return "FLIPMILHAS"
    if ag.startswith("VAIDEPROMO"):
        return "VAIDEPROMO"
    if ag.startswith("KISSANDFLY"):
        return "KISSANDFLY"
    if ag.startswith("ZUPPER"):
        return "ZUPPER"
    if ag.startswith("MYTRIP"):
        return "MYTRIP"
    if ag.startswith("GOTOGATE"):
        return "GOTOGATE"
    if ag.startswith("DECOLAR"):
        return "DECOLAR"
    if ag.startswith("EXPEDIA"):
        return "EXPEDIA"
    if ag.startswith("GOL"):
        return "GOL"
    if ag.startswith("LATAM"):
        return "LATAM"
    if ag.startswith("TRIPCOM"):
        return "TRIP.COM"
    if ag.startswith("VIAJANET"):
        return "VIAJANET"
    if ag in ("", "NAN", "NONE", "NULL", "SKYSCANNER"):
        return "SEM OFERTAS"
    return ag

def std_cia(raw: str) -> str:
    """Padroniza nomes de companhias aéreas."""
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
    """Encontra o valor ADVP canônico mais próximo."""
    try:
        v = float(str(x).replace(",", "."))
    except Exception:
        v = np.nan
    if np.isnan(v):
        v = 1
    return min([1, 5, 11, 17, 30], key=lambda k: abs(v - k))

def fmt_int(n: int) -> str:
    """Formata um número inteiro com separador de milhar."""
    try:
        return f"{int(n):,}".replace(",", ".")
    except Exception:
        return "0"

def fmt_num0_br(x):
    """Formata um número float para o padrão brasileiro com 0 casas decimais."""
    try:
        v = float(x)
        if not np.isfinite(v):
            return "-"
        return f"{v:,.0f}".replace(",", ".")
    except Exception:
        return "-"

def fmt_pct2_br(v):
    """Formata um número para porcentagem no padrão brasileiro."""
    try:
        x = float(v)
        if not np.isfinite(x):
            return "-"
        return f"{x:.2f}%".replace(".", ",")
    except Exception:
        return "-"

def fmt_pct0_br(v):
    """Formata um número para porcentagem arredondada sem casas decimais (ex: 3%)."""
    try:
        x = float(v)
        if not np.isfinite(x):
            return "-"
        return f"{int(round(x))}%"
    except Exception:
        return "-"

# ─────────────────────────── CARREGAMENTO DA BASE ────────────────────────────
@st.cache_data(show_spinner=True)
def load_base(path: Path) -> pd.DataFrame:
    """Carrega o arquivo Parquet, faz a padronização e normalização dos dados."""
    if not path.exists():
        st.error(f"Arquivo obrigatório não encontrado: {path.as_posix()}"); st.stop()
    df = pd.read_parquet(path)
    # Padroniza as 13 primeiras colunas se necessário (mantém compatibilidade)
    colmap = {0:"IDPESQUISA",1:"CIA",2:"HORA_BUSCA",3:"HORA_PARTIDA",4:"HORA_CHEGADA",
              5:"TIPO_VOO",6:"DATA_EMBARQUE",7:"DATAHORA_BUSCA",8:"AGENCIA_COMP",9:"PRECO",
              10:"TRECHO",11:"ADVP",12:"RANKING"}
    if list(df.columns[:13]) != list(colmap.values()):
        rename = {df.columns[i]: colmap[i] for i in range(min(13, df.shape[1]))}
        df = df.rename(columns=rename)
    # Normalizações
    for c in ["HORA_BUSCA", "HORA_PARTIDA", "HORA_CHEGADA"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c].astype(str).str.strip(), errors="coerce").dt.strftime("%H:%M:%S")
    # HORA_HH para filtros por hora
    df["HORA_HH"] = pd.to_datetime(df.get("HORA_BUSCA"), errors="coerce").dt.hour
    # Datas de interesse (dayfirst, formato BR)
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
    # Ranking inteiro
    if "RANKING" in df.columns:
        df["RANKING"] = pd.to_numeric(df["RANKING"], errors="coerce").astype("Int64")
    # Normalizações auxiliares
    df["AGENCIA_NORM"] = df.get("AGENCIA_COMP", pd.Series([], dtype=str)).apply(std_agencia)
    df["ADVP_CANON"] = df.get("ADVP", pd.Series([], dtype=str)).apply(lambda x: advp_nearest(x) if pd.notna(x) else np.nan)
    df["CIA_NORM"] = df.get("CIA", pd.Series([None] * len(df))).apply(std_cia)
    # ── __DTKEY__ robusto (alinhado ao common.py)
    dt_base = pd.to_datetime(df.get("DATAHORA_BUSCA"), errors="coerce", dayfirst=True)
    def _hh_to_sec(hs: object) -> float:
        s = _norm_hhmmss(hs)
        if not s:
            return np.nan
        hh, mm, ss = [int(x) for x in s.split(":")]
        return hh * 3600 + mm * 60 + ss
    hora_sec = pd.to_numeric(df.get("HORA_BUSCA", pd.Series([np.nan] * len(df))).map(_hh_to_sec), errors="coerce")
    dt_norm = pd.to_datetime(dt_base.dt.date, errors="coerce")
    dtkey = dt_norm + pd.to_timedelta(hora_sec.fillna(0), unit="s")
    mask_dt_ok = pd.notna(dt_base)
    mask_h_ok = pd.notna(hora_sec)
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

# Ajuste global de cabeçalhos: diminuir fonte e evitar quebra de linha
GLOBAL_HEADER_CSS = """
<style>
table th { font-size:11px !important; white-space:nowrap !important; overflow:hidden !important; text-overflow:ellipsis !important; padding:4px 6px !important; }
table td { font-size:12px !important; padding:6px 8px !important; }
/* garantir que o conteúdo do cabeçalho seja truncado com reticências quando necessário */
table th div, table th span { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
</style>
"""
st.markdown(GLOBAL_HEADER_CSS, unsafe_allow_html=True)

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
    """Gera o HTML para um card de ranking, exibindo os 3 primeiros lugares."""
    try:
        p1 = float(p1 or 0.0)
        p2 = float(p2 or 0.0)
        p3 = float(p3 or 0.0)
    except Exception:
        p1, p2, p3 = 0.0, 0.0, 0.0
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
    """Gera o HTML para um card de ranking de Cia, exibindo apenas o 1º lugar."""
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

def make_bar(df: pd.DataFrame, x_col: str, y_col: str, sort_y_desc: bool = True):
    """Cria um gráfico de barras Altair."""
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
    """Cria um gráfico de linhas Altair."""
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
BLUE = "#cfe3ff"
ORANGE = "#fdd0a2"
GREEN = "#c7e9c0"
YELLOW = "#fee391"
PINK = "#f1b6da"

def _hex_to_rgb(h): return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
def _rgb_to_hex(t): return f"#{t[0]:02x}{t[1]:02x}{t[2]:02x}"
def _blend(c_from, c_to, t):
    f, to = _hex_to_rgb(c_from), _hex_to_rgb(c_to)
    return _rgb_to_hex(tuple(int(round(f[i] + (to[i] - f[i]) * t)) for i in range(3)))

def make_scale(base_hex, steps=5): return [_blend("#ffffff", base_hex, k / (steps - 1)) for k in range(steps)]

SCALE_BLUE = make_scale(BLUE)
SCALE_ORANGE = make_scale(ORANGE)
SCALE_GREEN = make_scale(GREEN)
SCALE_YELLOW = make_scale(YELLOW)
SCALE_PINK = make_scale(PINK)

def _pick_scale(colname: str):
    u = str(colname).upper()
    if "MAXMILHAS" in u:
        return SCALE_GREEN
    if "123" in u:
        return SCALE_ORANGE
    if "FLIP" in u:
        return SCALE_YELLOW
    if "CAPO" in u:
        return SCALE_PINK
    return SCALE_BLUE

def _is_null_like(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    if isinstance(v, str) and v.strip().lower() in {"none", "nan", ""}:
        return True
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
    sty = sty.set_table_styles([{"selector": "tbody td, th", "props": [("border", "1px solid #EEE")]}])
    return sty

def show_table(df: pd.DataFrame, styler: pd.io.formats.style.Styler | None = None, caption: str | None = None):
    """Exibe uma tabela no Streamlit, com ou sem estilo."""
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
    """Decorador para registrar funções como abas do aplicativo."""
    def _wrap(fn: Callable):
        TAB_REGISTRY.append((label, fn))
        return fn
    return _wrap

# ─────────────────────── FUNÇÕES DE LÓGICA DO APP ────────────────────────────
def winners_by_position(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a agência vencedora para cada pesquisa, por ranking."""
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
    """Retorna a data e hora da última atualização dos dados."""
    ts = pd.to_datetime(df.get("__DTKEY__"), errors="coerce").max()
    if pd.isna(ts):
        ts = pd.to_datetime(df.get("DATAHORA_BUSCA"), errors="coerce").max()
    return ts.strftime("%d/%m/%Y - %H:%M:%S") if pd.notna(ts) else "—"

def _init_filter_state(df_raw: pd.DataFrame):
    """Inicializa o estado dos filtros na sessão do Streamlit."""
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
    """Renderiza os widgets de filtro e aplica as seleções ao DataFrame."""
    _init_filter_state(df_raw)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # 7 colunas — a última (c7) é o botão ↻ discreto ao lado do filtro de CIA
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
    with c7:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)  # alinhamento vertical
        if st.button("↻", key=f"{key_prefix}_refresh", type="secondary", help="Recarregar base"):
            st.cache_data.clear()
            st.session_state.pop("flt", None)
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
    st.caption(f"Quantidade de Ofertas: {fmt_int(len(df))}")
    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
    return df

# ───────────────────────────── ABAS DO APLICATIVO ────────────────────────────
# =============================== ABAS (INÍCIO) ===============================
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
        unsafe_allow_html=True
    )
    st.markdown("<hr style='margin:6px 0'>", unsafe_allow_html=True)

    # Layout principal: 2 colunas (1 para geral, 2 para cias)
    main_col1, main_col2 = st.columns([0.3, 0.7], gap="large")

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
        if "GRUPO 123" not in targets_base:
            targets_base.insert(0, "GRUPO 123")
        if "SEM OFERTAS" not in targets_base:
            targets_base.append("SEM OFERTAS")
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
        # Usa 'cards-stack' para empilhar verticalmente
        st.markdown(f"<div class='cards-stack'>{''.join(cards)}</div>", unsafe_allow_html=True)

    with main_col2:
        st.subheader("Ranking por Cia")
        if "CIA_NORM" not in df.columns:
            st.info("Coluna 'CIA_NORM' não encontrada nos dados filtrados."); return
        c1, c2, c3 = st.columns(3)

        cia_colors = {
            "AZUL": "background-color: #0033A0; color: white;",
            "GOL": "background-color: #FF6600; color: white;",
            "LATAM": "background-color: #8B0000; color: white;"
        }
        def render_por_cia(container, df_in: pd.DataFrame, cia_name: str):
            with container:
                style = cia_colors.get(cia_name, "background:#f8fafc; color:#0A2A6B;")
                st.markdown(f"<div class='stack-title' style='{style}'>Ranking {cia_name.title()}</div>", unsafe_allow_html=True)
                sub = df_in[df_in["CIA_NORM"].astype(str).str.upper() == cia_name]
                if sub.empty:
                    st.info("Sem dados para os filtros atuais."); return
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
                    p2 = float((base["R2"] == tgt).mean()) * 100
                    p3 = float((base["R3"] == tgt).mean()) * 100
                    return p1, p2, p3
                targets_sorted_local = sorted(targets, key=lambda t: pct_target(t)[0], reverse=True)
                # Usa a nova função de card que mostra apenas o 1º lugar
                cards_local = [card_html_cia(t, pct_target(t)[0]) for t in targets_sorted_local]
                st.markdown(f"<div class='cards-stack'>{''.join(cards_local)}</div>", unsafe_allow_html=True)

        render_por_cia(c1, df, "AZUL"); render_por_cia(c2, df, "GOL"); render_por_cia(c3, df, "LATAM")

# ──────────────────────── ABA: Top 3 Agências (START) ────────────────────────
@register_tab("Top 3 Agências")
def tab2_top3_agencias(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t2")
    st.subheader("Top 3 Agências (por menor preço no trecho)")
    if df.empty:
        st.info("Sem resultados para os filtros selecionados.")
        return
    # 1) Garantir mesma pesquisa (pega a última por Trecho)
    df2 = df.copy()
    df2["DT"] = pd.to_datetime(df2["DATAHORA_BUSCA"], errors="coerce")
    g = (df2.dropna(subset=["TRECHO", "IDPESQUISA", "DT"])
         .groupby(["TRECHO", "IDPESQUISA"], as_index=False)["DT"].max())
    last_idx = g.groupby("TRECHO")["DT"].idxmax()
    last_ids = {r["TRECHO"]: r["IDPESQUISA"] for _, r in g.loc[last_idx].iterrows()}
    df2["__ID_TARGET__"] = df2["TRECHO"].map(last_ids)
    df_last = df2[df2["IDPESQUISA"].astype(str) == df2["__ID_TARGET__"].astype(str)].copy()
    # 2) Coluna "Data/Hora Busca" (DATAHORA + hora da coluna C/HORA_BUSCA)
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
    # 3) Ranking Top-3 por Trecho (mesma pesquisa)
    PRICE_COL, TRECHO_COL, AGENCIA_COL = "PRECO", "TRECHO", "AGENCIA_NORM"
    by_ag = (
        df_last.groupby([TRECHO_COL, AGENCIA_COL], as_index=False)
        .agg(PRECO_MIN=(PRICE_COL, "min"))
        .rename(columns={TRECHO_COL: "TRECHO_STD", AGENCIA_COL: "AGENCIA_UP"})
    )
    def _row_top3(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("PRECO_MIN", ascending=True).reset_index(drop=True)
        trecho = g["TRECHO_STD"].iloc[0] if len(g) else "-"
        def name(i): return g.loc[i, "AGENCIA_UP"] if i < len(g) else "-"
        def price(i): return g.loc[i, "PRECO_MIN"] if i < len(g) else np.nan
        return pd.Series({
            "Data/Hora Busca": dt_by_trecho.get(trecho, "-"),
            "Trecho": trecho,
            "Agencia Top 1": name(0), "Preço Top 1": price(0),
            "Agencia Top 2": name(1), "Preço Top 2": price(1),
            "Agencia Top 3": name(2), "Preço Top 3": price(2),
        })
    t1 = by_ag.groupby("TRECHO_STD").apply(_row_top3).reset_index(drop=True)
    t1 = t1.reset_index(drop=True)
    for c in ["Preço Top 1", "Preço Top 2", "Preço Top 3"]:
        t1[c] = pd.to_numeric(t1[c], errors="coerce")
    sty1 = style_smart_colwise(t1, {c: fmt_num0_br for c in ["Preço Top 1", "Preço Top 2", "Preço Top 3"]},
                               grad_cols=["Preço Top 1", "Preço Top 2", "Preço Top 3"])
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
    by_air = (df_last.groupby(["TRECHO", "CIA_NORM"], as_index=False)
              .agg(PRECO_AIR_MIN=("PRECO", "min"))
              .rename(columns={"TRECHO": "TRECHO_STD", "CIA_NORM": "Cia Menor Preço"}))
    idx = by_air.groupby("TRECHO_STD")["PRECO_AIR_MIN"].idxmin()
    base_min = by_air.loc[idx, ["TRECHO_STD", "Cia Menor Preço", "PRECO_AIR_MIN"]] \
        .rename(columns={"PRECO_AIR_MIN": "Preço Menor Valor"})
    def _best_price(sub: pd.DataFrame, ag: str) -> float:
        m = sub[sub["AGENCIA_NORM"] == ag]
        return float(m["PRECO"].min()) if not m.empty else np.nan
    rows3 = []
    for trecho, sub in df_last.groupby("TRECHO"):
        rows3.append({
            "Data/Hora Busca": dt_by_trecho.get(trecho, "-"),
            "Trecho": trecho,
            "Cia Menor Preço": base_min.loc[base_min["TRECHO_STD"].eq(trecho), "Cia Menor Preço"].squeeze() if (base_min["TRECHO_STD"] == trecho).any() else "-",
            "Preço Menor Valor": base_min.loc[base_min["TRECHO_STD"].eq(trecho), "Preço Menor Valor"].squeeze() if (base_min["TRECHO_STD"] == trecho).any() else np.nan,
            "123milhas": _best_price(sub, A_123),
            "Maxmilhas": _best_price(sub, A_MAX),
            "FlipMilhas": _best_price(sub, A_FLIP),
            "Capo Viagens": _best_price(sub, A_CAPO),
        })
    t3 = pd.DataFrame(rows3).reset_index(drop=True)
    fmt3 = {c: fmt_num0_br for c in ["Preço Menor Valor", "123milhas", "Maxmilhas", "FlipMilhas", "Capo Viagens"]}
    sty3 = style_smart_colwise(t3, fmt3, grad_cols=list(fmt3.keys()))
    show_table(t3, sty3, caption="Comparativo Menor Preço Cia × Agências de Milhas")
    # 6) % Dif. vs menor valor por Cia
    def pct_vs_base(b, x):
        if pd.isna(b) or b == 0 or pd.isna(x):
            return np.nan
        return (x - b) / b * 100
    t4 = t3[["Data/Hora Busca", "Trecho", "Cia Menor Preço", "Preço Menor Valor"]].copy()
    for label in ["123milhas", "Maxmilhas", "FlipMilhas", "Capo Viagens"]:
        t4[f"% Dif {label}"] = [pct_vs_base(b, x) for b, x in zip(t3["Preço Menor Valor"], t3[label])]
    fmt4 = {"Preço Menor Valor": fmt_num0_br} | {c: fmt_pct2_br for c in t4.columns if c.startswith("% Dif ")}
    sty4 = style_smart_colwise(t4.reset_index(drop=True), fmt4, grad_cols=list(fmt4.keys()))
    show_table(t4, sty4, caption="% Comparativo Menor Preço Cia × Agências de Milhas")

# ──────────────────── ABA: Top 3 Preços Mais Baratos (START) ─────────────────
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
            if not np.isfinite(xv):
                return "R$ -"
            return "R$ " + f"{xv:,.0f}".replace(",", ".")
        except Exception:
            return "R$ -"
    def _find_id_col(df_: pd.DataFrame) -> str | None:
        cands = ["IDPESQUISA", "ID_PESQUISA", "ID BUSCA", "IDBUSCA", "ID", "NOME_ARQUIVO_STD", "NOME_ARQUIVO", "NOME DO ARQUIVO", "ARQUIVO"]
        norm = {re.sub(r"[^A-Z0-9]+", "", c.upper()): c for c in df_.columns}
        for nm in cands:
            key = re.sub(r"[^A-Z0-9]+", "", nm.upper())
            if key in norm:
                return norm[key]
        return df_.columns[0] if len(df_.columns) else None
    GRID_STYLE = "display:grid;grid-auto-flow:column;grid-auto-columns:260px;gap:10px;overflow-x:auto;padding:6px 2px 10px 2px;scrollbar-width:thin;"
    BOX_STYLE = "border:1px solid #e5e7eb;border-radius:12px;background:#fff;"
    HEAD_STYLE = "padding:8px 10px;border-bottom:1px solid #f1f5f9;font-weight:700;color:#111827;"
    STACK_STYLE = "display:grid;gap:8px;padding:8px;"
    CARD_BASE = "position:relative;border:1px solid #e5e7eb;border-radius:10px;padding:8px;background:#fff;min-height:78px;"
    DT_WRAP_STYLE = "position:absolute;right:8px;top:6px;display:flex;align-items:center;gap:6px;"
    DT_TXT_STYLE = "font-size:10px;color:#94a3b8;font-weight:800;"
    RANK_STYLE = "font-weight:900;font-size:11px;color:#6b7280;letter-spacing:.3px;text-transform:uppercase;"
    AG_STYLE = "font-weight:800;font-size:15px;color:#111827;margin-top:2px;"
    PR_STYLE = "font-weight:900;font-size:18px;color:#111827;margin-top:2px;"
    SUB_STYLE = "font-weight:700;font-size:12px;color:#374151;"
    NO_STYLE = "padding:22px 12px;color:#6b7280;font-weight:800;text-align:center;border:1px dashed #e5e7eb;border-radius:10px;background:#fafafa;"
    TRE_HDR_STYLE = "margin:14px 0 10px 0;padding:10px 12px;border-left:4px solid #0B5FFF;background:#ECF3FF;border-radius:8px;font-weight:800;color:#0A2A6B;"
    BADGE_POP_CSS = """
    <style>
    .idp-wrap{position:relative; display:inline-flex; align-items:center;}
    .idp-badge{display:inline-flex; align-items:center; justify-content:center;width:16px; height:16px; border:1px solid #cbd5e1; border-radius:50%;font-size:11px; font-weight:900; color:#64748b; background:#fff;user-select:none; cursor:default; line-height:1;}
    .idp-pop{position:absolute; top:18px; right:0;background:#fff; color:#0f172a; border:1px solid #e5e7eb;border-radius:8px; padding:6px 8px; font-size:12px; font-weight:700;box-shadow:0 6px 16px rgba(0,0,0,.08); display:none; z-index:9999; white-space:nowrap;}
    .idp-wrap:hover .idp-pop{ display:block; }
    .idp-idbox{border:1px solid #e5e7eb; background:#f8fafc; border-radius:6px;padding:2px 6px; font-weight:800; font-size:12px; min-width:60px;font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;user-select:text; cursor:text;}
    .tag-extra{font-size:12px; color:#4b5563; margin-top:2px;}
    .extra-wrap{padding:6px 8px 10px 8px; border-top:1px dashed #e5e7eb; margin-top:6px;}
    .extra-title{font-size:12px; font-weight:700; color:#374151; margin-bottom:2px;}
    </style>
    """
    st.markdown(BADGE_POP_CSS, unsafe_allow_html=True)
    dfp = df.copy()
    dfp["TRECHO_STD"] = dfp.get("TRECHO", "").astype(str)
    dfp["AGENCIA_UP"] = dfp.get("AGENCIA_NORM", "").astype(str)
    dfp["ADVP"] = (dfp.get("ADVP_CANON").fillna(dfp.get("ADVP"))).astype(str)
    dfp["__PRECO__"] = pd.to_numeric(dfp.get("PRECO"), errors="coerce")
    dfp["__DTKEY__"] = pd.to_datetime(dfp.get("DATAHORA_BUSCA"), errors="coerce")
    ID_COL = "IDPESQUISA" if "IDPESQUISA" in dfp.columns else _find_id_col(dfp)
    dfp = dfp[dfp["__PRECO__"].notna()].copy()
    if dfp.empty:
        st.info("Sem preços válidos no recorte atual."); return
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
        """Data (DD/MM) + Hora HH:MM:SS, garantindo segundos."""
        if sub_rows.empty:
            return "", None
        r = sub_rows.loc[sub_rows["__DTKEY__"].idxmax()]
        date_part = pd.to_datetime(r["DATAHORA_BUSCA"], errors="coerce")
        date_txt = date_part.strftime("%d/%m") if pd.notna(date_part) else ""
        htxt_raw = str(r.get("HORA_BUSCA", "")).strip()
        htxt = _norm_hhmmss(htxt_raw) or (pd.to_datetime(r["__DTKEY__"], errors="coerce").strftime("%H:%M:%S") if pd.notna(r["__DTKEY__"]) else "")
        id_val = _normalize_id(r.get("IDPESQUISA"))
        lbl = f"{date_txt} {htxt}".strip()
        return lbl, id_val
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
            box_content = []
            box_content.append(f"<div style='{BOX_STYLE}'>")
            box_content.append(f"<div style='{HEAD_STYLE}'>ADVP: <b>{advp}</b></div>")
            if base_rank.empty:
                box_content.append(f"<div style='{NO_STYLE}'>Sem ofertas</div>")
                box_content.append("</div>"); boxes.append("".join(box_content)); continue
            box_content.append(f"<div style='{STACK_STYLE}'>")
            for i in range(min(3, len(base_rank))):
                row_i = base_rank.iloc[i]
                preco_i = float(row_i["__PRECO__"])
                sub_rows = all_rows[(all_rows["AGENCIA_UP"] == row_i["AGENCIA_UP"]) & (np.isclose(all_rows["__PRECO__"], preco_i, atol=1))]
                dt_lbl, id_val = dt_and_id_for(sub_rows)
                # >>> NOVO: para 1º lugar, mostrar quanto é mais barato que o 2º
                if i == 0:
                    subtxt = "—"
                    if len(base_rank) >= 2:
                        p1 = preco_i
                        p2 = float(base_rank.iloc[1]["__PRECO__"])
                        if np.isfinite(p2) and p2 != 0:
                            pct_below = int(round((p2 - p1) / p2 * 100))
                            subtxt = f"-{pct_below}% vs 2º"
                else:
                    # já existia: diferença do i-ésimo vs 1º
                    p1 = float(base_rank.iloc[0]["__PRECO__"])
                    subtxt = "—"
                    if np.isfinite(p1) and p1 != 0:
                        subtxt = f"+{int(round((preco_i - p1) / p1 * 100))}% vs 1º"
                stripe = "#D4AF37" if i == 0 else "#9CA3AF" if i == 1 else "#CD7F32"
                box_content.append(
                    f"<div style='{CARD_BASE}'>"
                    f"<div style='position:absolute;left:0;top:0;bottom:0;width:6px;border-radius:10px 0 0 10px;background:{stripe};'></div>"
                    f"<div style='{DT_WRAP_STYLE}'><span style='{DT_TXT_STYLE}'>{dt_lbl}</span>"
                    f"<span class='idp-wrap'><span class='idp-badge'>?</span>"
                    f"<span class='idp-pop'>ID:&nbsp;<input class='idp-idbox' type='text' value='{_normalize_id(id_val)}' readonly></span></span></div>"
                    f"<div style='{RANK_STYLE}'>{i + 1}º</div>"
                    f"<div style='{AG_STYLE}'>{row_i['AGENCIA_UP']}</div>"
                    f"<div style='{PR_STYLE}'>{fmt_moeda_br(preco_i)}</div>"
                    f"<div style='{SUB_STYLE}'>{subtxt}</div>"
                    f"</div>"
                )
            box_content.append("</div>")
            # NOVO BLOCO: status de MAXMILHAS / 123MILHAS (fora do Top3) + preço
            p1_val = float(base_rank.iloc[0]["__PRECO__"])
            msgs = []
            # MAXMILHAS
            idx_max = base_rank.index[base_rank["AGENCIA_UP"] == "MAXMILHAS"].tolist()
            if not idx_max:
                msgs.append("Maxmilhas Não Apareceu")
            elif idx_max[0] > 2:
                pos = idx_max[0] + 1
                preco = float(base_rank.iloc[idx_max[0]]["__PRECO__"])
                dif = int(round((preco - p1_val) / p1_val * 100)) if p1_val else 0
                msgs.append(f"Maxmilhas: {pos}º - {fmt_moeda_br(preco)} (+{dif}% vs 1º)")
            # 123MILHAS
            idx_123 = base_rank.index[base_rank["AGENCIA_UP"] == "123MILHAS"].tolist()
            if not idx_123:
                msgs.append("123milhas Não Apareceu")
            elif idx_123[0] > 2:
                pos = idx_123[0] + 1
                preco = float(base_rank.iloc[idx_123[0]]["__PRECO__"])
                dif = int(round((preco - p1_val) / p1_val * 100)) if p1_val else 0
                msgs.append(f"123milhas: {pos}º - {fmt_moeda_br(preco)} (+{dif}% vs 1º)")
            if msgs:
                box_content.append(
                    "<div class='extra-wrap'>"
                    "<div class='extra-title'>Status Grupo 123</div>"
                    + "".join([f"<div class='tag-extra'>{m}</div>" for m in msgs]) +
                    "</div>"
                )
            box_content.append("</div>")
            boxes.append("".join(box_content))
        if boxes:
            st.markdown(f"<div style='{TRE_HDR_STYLE}'>Trecho: <b>{trecho}</b></div>", unsafe_allow_html=True)
            st.markdown("<div style='" + GRID_STYLE + "'>" + "".join(boxes) + "</div>", unsafe_allow_html=True)

# ─────────────────────── ABA: Ranking por Agências (START) ────────────────────
@register_tab("Ranking por Agências")
def tab4_ranking_agencias(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t4")
    if df.empty:
        st.subheader("Ranking por Agências")
        st.info("Sem dados para os filtros.")
        return
    # 2) Normalização mínima para coincidir com o anexo
    work = df.copy()
    work["AGENCIA_UP"] = work.get("AGENCIA_UP", work.get("AGENCIA_NORM", work.get("AGENCIA_COMP", work.get("AGENCIA", "")))).astype(str)
    if "RANKING" not in work.columns:
        st.warning("Coluna 'RANKING' não encontrada."); return
    work["Ranking"] = pd.to_numeric(work["RANKING"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["AGENCIA_UP", "Ranking"])
    work["Ranking"] = work["Ranking"].astype(int)
    # 3) Funções de estilo (estilo semelhante ao arquivo anexo)
    def _fmt_pct(v):
        try:
            x = float(v)
            return "-" if not np.isfinite(x) else f"{x:.2f}%".replace(".", ",")
        except Exception:
            return "-"
    def style_heatmap(df_show: pd.DataFrame, percent_cols=None, int_cols=None,
                      highlight_total_row=False, highlight_total_col=None,
                      highlight_rows_map=None, height=440):
        percent_cols = set(percent_cols or [])
        int_cols = set(int_cols or [])
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
    def show_table(df_in: pd.DataFrame, percent_cols=None, int_cols=None,
                   highlight_total_row=False, highlight_total_col=None,
                   highlight_rows_map=None, height=440):
        df_disp = df_in.reset_index(drop=True)
        df_disp.index = np.arange(1, len(df_disp) + 1)
        st.dataframe(
            style_heatmap(df_disp, percent_cols=percent_cols, int_cols=int_cols,
                          highlight_total_row=highlight_total_row, highlight_total_col=highlight_total_col,
                          highlight_rows_map=highlight_rows_map, height=height),
            use_container_width=True, height=height
        )
    # 4) Pivot principal (Quantidade por Ranking)
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
    # ---------- Tabela 1 — Quantidades ----------
    t_qtd = pv2.reset_index()
    if t_qtd.columns[0] != "Agência/Companhia":
        t_qtd = t_qtd.rename(columns={t_qtd.columns[0]: "Agência/Companhia"})
    st.subheader("Quantidade de Ofertas por Ranking (Ofertas)")
    show_table(
        t_qtd[["Agência/Companhia"] + RANKS + ["Total"]],
        int_cols=set(RANKS + ["Total"]),
        highlight_total_row=True,
        highlight_total_col="Total",
        highlight_rows_map=HL_MAP,
        height=480
    )
    # ---------- Tabela 2 — % dentro da Agência (linha) ----------
    mat = pv[RANKS].copy()
    row_sum = mat.sum(axis=1).replace(0, np.nan)
    pct_linha = (mat.div(row_sum, axis=0) * 100).fillna(0)
    pct_linha = pct_linha.sort_values(by=1, ascending=False)
    t_pct_linha = pct_linha.reset_index()
    if t_pct_linha.columns[0] != "Agência/Companhia":
        t_pct_linha = t_pct_linha.rename(columns={t_pct_linha.columns[0]: "Agência/Companhia"})
    st.subheader("Participação Ranking dentro da Agência")
    show_table(
        t_pct_linha[["Agência/Companhia"] + RANKS],
        percent_cols=set(RANKS),
        highlight_rows_map=HL_MAP,
        height=440
    )
    # ---------- Tabela 3 — % dentro do Ranking (coluna) ----------
    col_sum = mat.sum(axis=0).replace(0, np.nan)
    pct_coluna = (mat.div(col_sum, axis=1) * 100).fillna(0)
    pct_coluna = pct_coluna.sort_values(by=1, ascending=False)
    t_pct_coluna = pct_coluna.reset_index()
    if t_pct_coluna.columns[0] != "Agência/Companhia":
        t_pct_coluna = t_pct_coluna.rename(columns={t_pct_coluna.columns[0]: "Agência/Companhia"})
    st.subheader("Participação Ranking Geral")
    show_table(
        t_pct_coluna[["Agência/Companhia"] + RANKS],
        percent_cols=set(RANKS),
        highlight_rows_map=HL_MAP,
        height=440
    )

# ─────────────────────── ABA 5: Competitividade Cia × Trecho (3 quadros + ADVP) ───────────────────────
@register_tab("Competitividade Cia x Trecho")
def tab5_competitividade(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t5")
    st.subheader("Competitividade Cia × Trecho")
    if df.empty:
        st.info("Sem resultados para os filtros atuais."); return
    need = {"RANKING", "CIA_NORM", "TRECHO", "AGENCIA_NORM", "IDPESQUISA"}
    if not need.issubset(df.columns):
        st.warning(f"Colunas ausentes: {sorted(list(need - set(df.columns)))}"); return
    # 1) Apenas 1º lugar
    df1 = df[df["RANKING"].astype("Int64") == 1].copy()
    if df1.empty:
        st.info("Nenhum 1º lugar no recorte."); return
    # 2) Normalizações
    df1["CIA_UP"] = df1["CIA_NORM"].astype(str).str.upper()
    df1["TRECHO_STD"] = df1["TRECHO"].astype(str)
    df1["AG_UP"] = df1["AGENCIA_NORM"].astype(str)
    # 3) Universo de 11 trechos: top 11 por nº de pesquisas no recorte
    top_trechos = (df1.groupby("TRECHO_STD")["IDPESQUISA"]
                   .nunique().sort_values(ascending=False).head(11).index.tolist())
    if len(top_trechos) < 11:
        restantes = [t for t in df1["TRECHO_STD"].unique().tolist() if t not in top_trechos]
        top_trechos += restantes[: max(0, 11 - len(top_trechos))]
    # 4) % de 1º por Cia×Trecho×Agência + total por Cia×Trecho
    tot = (df1.groupby(["CIA_UP", "TRECHO_STD"])["IDPESQUISA"]
           .nunique().reset_index(name="TotalPesq"))
    win = (df1.groupby(["CIA_UP", "TRECHO_STD", "AG_UP"])["IDPESQUISA"]
           .nunique().reset_index(name="QtdTop1"))
    base = win.merge(tot, on=["CIA_UP", "TRECHO_STD"], how="right")
    base["QtdTop1"] = base["QtdTop1"].fillna(0)
    base["Pct"] = (base["QtdTop1"] / base["TotalPesq"].replace(0, np.nan) * 100).fillna(0.0)
    # helper para pegar TotalPesq com fallback 0
    def _tot_for(cia: str, trecho: str) -> int:
        m = tot.loc[(tot["CIA_UP"] == cia) & (tot["TRECHO_STD"] == trecho), "TotalPesq"]
        if m.empty:
            return 0
        v = m.iloc[0]
        try:
            return int(v) if pd.notna(v) else 0
        except Exception:
            try:
                return int(float(v))
            except Exception:
                return 0
    # líderes por CIA (11 trechos fixos)
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
                ag = str(lideres.loc[t, "AG_UP"]) if pd.notna(lideres.loc[t, "AG_UP"]) else "SEM OFERTAS"
                pct = float(lideres.loc[t, "Pct"]) if pd.notna(lideres.loc[t, "Pct"]) else 0.0
                n = int(lideres.loc[t, "TotalPesq"]) if pd.notna(lideres.loc[t, "TotalPesq"]) else _tot_for(cia, t)
            else:
                ag, pct, n = "SEM OFERTAS", 0.0, _tot_for(cia, t)
            rows.append({"TRECHO": t, "AGENCIA": ag, "PCT": pct, "N": n})
        return pd.DataFrame(rows).sort_values("PCT", ascending=False, kind="mergesort").reset_index(drop=True)
    # cores por CIA
    cia_colors = {"AZUL": ("#2D6CDF", "#FFFFFF"), "GOL": ("#E67E22", "#FFFFFF"), "LATAM": ("#C0392B", "#FFFFFF")}
    # CSS geral + quadros por Trecho
    st.markdown("""
    <style>
        .comp-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;}
        @media (max-width:1100px){.comp-grid{grid-template-columns:repeat(2,minmax(0,1fr));}}
        @media (max-width:700px){.comp-grid{grid-template-columns:1fr;}}
        .comp-card{border:1px solid #e5e7eb;border-radius:12px;background:#fff;overflow:hidden;box-shadow:0 1px 2px rgba(0,0,0,.06);}
        .comp-head{padding:10px 12px;font-weight:900;letter-spacing:.2px;}
        .tbl,.tblc{width:100%;border-collapse:collapse;}
        .tbl th,.tbl td,.tblc th,.tblc td{border:1px solid #e5e7eb;padding:6px 8px;font-size:13px; text-align:center;} /* centraliza TUDO */
        .tbl td:nth-child(3){font-weight:800;} /* % em negrito */
        .muted{font-size:10px;color:#94a3b8;font-weight:800;display:block;line-height:1;margin-top:2px;}
        .row0{background:#ffffff;}
        .row1{background:#fcfcfc;}
        .g123{color:#0B6B2B;font-weight:900;} /* MAX/123 verde-escuro+negrito */
    </style>
    """, unsafe_allow_html=True)
    def fmt_pct(v: float) -> str:
        try:
            return f"{float(v):.2f}%".replace(".", ",")
        except:
            return "0,00%"
    # ---- Quadros por TRECHO (com contagem discreta) ----
    def quadro_html(cia: str, dfq: pd.DataFrame) -> str:
        bg, fg = cia_colors.get(cia, ("#0B5FFF", "#FFFFFF"))
        head = f"<div class='comp-head' style='background:{bg};color:{fg};'>{cia}</div>"
        if dfq.empty:
            body = "<div style='padding:14px;color:#64748b;font-weight:700;text-align:center;'>Sem dados</div>"
            return f"<div class='comp-card'>{head}{body}</div>"
        rows = []
        rows.append("<table class='tbl'><thead><tr><th>TRECHO</th><th>AGENCIA</th><th>% DE GANHO</th></tr></thead><tbody>")
        for i, r in dfq.iterrows():
            alt = "row1" if i % 2 else "row0"
            ag = str(r["AGENCIA"])
            ag_cls = "g123" if ag in {"MAXMILHAS", "123MILHAS"} else ""
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
    items = [quadro_html(cia, lideres_por_cia(cia)) for cia in ["AZUL", "GOL", "LATAM"]]
    st.markdown("<div class='comp-grid'>" + "".join(items) + "</div>", unsafe_allow_html=True)
    # ===================== RESUMO POR ADVP (abaixo dos quadros) =====================
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.subheader("Competitividade Cia x ADVP")
    # Buckets ADVP fixos na ordem solicitada
    advp_buckets = [1, 5, 11, 17, 30]
    # garantir coluna ADVP_BKT
    advp_series = pd.to_numeric(df1.get("ADVP_CANON"), errors="coerce")
    df1_advp = df1[advp_series.notna()].copy()
    df1_advp["ADVP_BKT"] = advp_series.loc[df1_advp.index].astype(int)
    # totais e vitórias por ADVP
    totA = (df1_advp.groupby(["CIA_UP", "ADVP_BKT"])["IDPESQUISA"]
            .nunique().reset_index(name="TotAdvp"))
    winA = (df1_advp.groupby(["CIA_UP", "ADVP_BKT", "AG_UP"])["IDPESQUISA"]
            .nunique().reset_index(name="QtdTop1"))
    baseA = winA.merge(totA, on=["CIA_UP", "ADVP_BKT"], how="right")
    baseA["QtdTop1"] = baseA["QtdTop1"].fillna(0)
    baseA["Pct"] = (baseA["QtdTop1"] / baseA["TotAdvp"].replace(0, np.nan) * 100).fillna(0.0)
    def lideres_por_cia_advp(cia: str) -> pd.DataFrame:
        sub = baseA[baseA["CIA_UP"] == cia].copy()
        rows = []
        if not sub.empty:
            sub = sub.sort_values(["ADVP_BKT", "Pct"], ascending=[True, False])
            idx = sub.groupby("ADVP_BKT")["Pct"].idxmax()
            lid = sub.loc[idx, ["ADVP_BKT", "AG_UP", "Pct"]].set_index("ADVP_BKT")
        else:
            lid = pd.DataFrame(columns=["AG_UP", "Pct"])
        for a in advp_buckets:  # ordem fixa 1,5,11,17,30
            if a in lid.index:
                ag = str(lid.loc[a, "AG_UP"]) if pd.notna(lid.loc[a, "AG_UP"]) else "SEM OFERTAS"
                pct = float(lid.loc[a, "Pct"]) if pd.notna(lid.loc[a, "Pct"]) else 0.0
            else:
                ag, pct = "SEM OFERTAS", 0.0
            rows.append({"ADVP": a, "AGENCIA": ag, "PCT": pct})
        # NÃO reordenar por % — manter a ordem 1,5,11,17,30
        return pd.DataFrame(rows)
    def quadro_advp_html(cia: str, dfq: pd.DataFrame) -> str:
        bg, fg = cia_colors.get(cia, ("#0B5FFF", "#FFFFFF"))
        head = f"<div class='comp-head' style='background:{bg};color:{fg};'>{cia}</div>"
        if dfq.empty:
            body = "<div style='padding:14px;color:#64748b;font-weight:700;text-align:center;'>Sem dados</div>"
            return f"<div class='comp-card'>{head}{body}</div>"
        rows = []
        rows.append("<table class='tblc'><thead><tr><th>ADVP</th><th>AGENCIA</th><th>% DE GANHO</th></tr></thead><tbody>")
        for i, r in dfq.iterrows():
            alt = "row1" if i % 2 else "row0"
            ag = str(r["AGENCIA"])
            ag_cls = "g123" if ag in {"MAXMILHAS", "123MILHAS"} else ""
            pct_txt = fmt_pct(r["PCT"])
            rows.append(
                f"<tr class='{alt}'>"
                f"<td>{r['ADVP']}</td>"
                f"<td><span class='{ag_cls}'>{ag}</span></td>"
                f"<td>{pct_txt}</td>"
                f"</tr>"
            )
        rows.append("</tbody></table>")
        return f"<div class='comp-card'>{head}{''.join(rows)}</div>"
    items_advp = [quadro_advp_html(cia, lideres_por_cia_advp(cia)) for cia in ["AZUL", "GOL", "LATAM"]]
    st.markdown("<div class='comp-grid'>" + "".join(items_advp) + "</div>", unsafe_allow_html=True)

# ─────────────────────── ABA 6: Competitividade (tabelas + 6 cards) ──────────
# ───────── ABA 6: Competitividade Cia x Trecho x ADVPs Agrupados (tabelas + 6 cards) ─────────
@register_tab("Competitividade Cia x Trecho x ADVPs Agrupados")
def tab6_compet_tabelas(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t6")
    if df.empty:
        st.subheader("Competividade  Cia x Trecho x ADVPs Agrupados")
        st.info("Sem resultados para os filtros atuais."); return
    need = {"RANKING", "CIA_NORM", "TRECHO", "AGENCIA_NORM", "IDPESQUISA"}
    if not need.issubset(df.columns):
        st.subheader("Competividade  Cia x Trecho x ADVPs Agrupados")
        st.warning(f"Colunas ausentes: {sorted(list(need - set(df.columns)))}"); return
    # winners
    d1 = df[df["RANKING"].astype("Int64") == 1].copy()
    d1["CIA_UP"] = d1["CIA_NORM"].astype(str).str.upper()
    d1["TRECHO_STD"] = d1["TRECHO"].astype(str)
    d1["AG_UP"] = d1["AGENCIA_NORM"].astype(str)
    trechos_all = sorted(df["TRECHO"].dropna().astype(str).unique().tolist())
    # Totais/Top1 por Cia×Trecho
    tot_t = (df.assign(CIA_UP=df["CIA_NORM"].astype(str).str.upper(),
                       TRECHO_STD=df["TRECHO"].astype(str))
             .groupby(["CIA_UP", "TRECHO_STD"])["IDPESQUISA"]
             .nunique().reset_index(name="TotPesq"))
    win_t = (d1.groupby(["CIA_UP", "TRECHO_STD", "AG_UP"])["IDPESQUISA"]
             .nunique().reset_index(name="QtdTop1"))
    base_t = win_t.merge(tot_t, on=["CIA_UP", "TRECHO_STD"], how="right").fillna({"QtdTop1": 0})
    base_t["Pct"] = (base_t["QtdTop1"] / base_t["TotPesq"].replace(0, np.nan) * 100).fillna(0.0)
    def pick_leader(cia: str, trecho: str):
        sub = base_t[(base_t["CIA_UP"] == cia) & (base_t["TRECHO_STD"] == trecho)]
        if sub.empty or (sub["TotPesq"].sum() == 0):
            return {"CIA": cia, "TRECHO": trecho, "AGENCIA": "SEM OFERTAS", "PCT": 0.0, "N": 0}
        top = sub.sort_values(["Pct", "QtdTop1", "TotPesq"], ascending=False).iloc[0]
        return {"CIA": cia, "TRECHO": trecho, "AGENCIA": str(top["AG_UP"]), "PCT": float(top["Pct"]), "N": int(top["TotPesq"])}
    # ===== CSS =====
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
    def cia_chip(cia: str) -> str:
        cls = "az" if cia == "AZUL" else "go" if cia == "GOL" else "la"
        return f"<span class='chip'><span class='dot {cls}'></span>{cia}</span>"
    def pct_cell(pct, n) -> str:
        try:
            p = int(round(float(pct)))
        except:
            p = 0
        try:
            k = int(n)
        except:
            k = 0
        return f"<span class='pct-val'>{p}%</span><span class='pesq'>( {k} pesq )</span>"
    def ag_fmt(ag: str) -> str:
        return f"<span class='g123'>{ag}</span>" if ag in {"123MILHAS", "MAXMILHAS"} else ag
    # ===== helpers de resumo =====
    def compute_summary(win_df: pd.DataFrame, total_base: int):
        if win_df.empty or total_base == 0:
            return ("SEM OFERTAS", 0, 0, "SEM OFERTAS", 0, 0, 0)
        cia_s = win_df.groupby("CIA_UP")["IDPESQUISA"].nunique().sort_values(ascending=False)
        ag_s = win_df.groupby("AG_UP")["IDPESQUISA"].nunique().sort_values(ascending=False)
        cia_nome, cia_qtd = (str(cia_s.index[0]), int(cia_s.iloc[0])) if not cia_s.empty else ("SEM OFERTAS", 0)
        ag_nome, ag_qtd = (str(ag_s.index[0]), int(ag_s.iloc[0])) if not ag_s.empty else ("SEM OFERTAS", 0)
        cia_pct = int(round(cia_qtd / total_base * 100)) if total_base else 0
        ag_pct = int(round(ag_qtd / total_base * 100)) if total_base else 0
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
    # ===== bloco especial: Competitividade - Grupo123 (apenas ADVP) =====
    def cards_block_grupo123(d1_advp: pd.DataFrame, base_advp_n: int):
        gset = {"123MILHAS", "MAXMILHAS"}
        sub = d1_advp[d1_advp["AG_UP"].isin(gset)].copy()
        total_gwins = sub["IDPESQUISA"].nunique() or 0
        if total_gwins:
            cia_cnt = sub.groupby("CIA_UP")["IDPESQUISA"].nunique().sort_values(ascending=False)
            cia_gnome = str(cia_cnt.index[0]); cia_gqtd = int(cia_cnt.iloc[0])
            cia_gpct = int(round(cia_gqtd / total_gwins * 100))
        else:
            cia_gnome, cia_gqtd, cia_gpct = "SEM OFERTAS", 0, 0
        c123 = int(sub[sub["AG_UP"] == "123MILHAS"]["IDPESQUISA"].nunique())
        cmax = int(sub[sub["AG_UP"] == "MAXMILHAS"]["IDPESQUISA"].nunique())
        cgrp = c123 + cmax
        p123 = int(round((c123 / base_advp_n * 100))) if base_advp_n else 0
        pmax = int(round((cmax / base_advp_n * 100))) if base_advp_n else 0
        pgrp = int(round((cgrp / base_advp_n * 100))) if base_advp_n else 0
        return f"""
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
        """
    # ======= Tabela 1: Cia × Trecho (SEM cards) =======
    def render_tbl_trecho():
        html = ["<table class='t6'>",
                "<thead><tr><th class='cia'>CIA</th><th class='trc l'>TRECHO</th><th class='ag l'>AGENCIA</th><th class='pct'>% DE GANHO</th></tr></thead><tbody>"]
        for t in trechos_all:
            rows = [pick_leader(cia, t) for cia in ["AZUL", "GOL", "LATAM"]]
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
    # ======= Tabela 2: Cia × ADVP (com 6 cards abaixo) =======
    buckets = [1, 5, 11, 17, 30]
    advp_series = pd.to_numeric(df.get("ADVP_CANON"), errors="coerce")
    df_advp = df[advp_series.notna()].copy()
    df_advp["ADVP_BKT"] = advp_series.loc[df_advp.index].astype(int)
    d1a = d1.copy()
    d1a["ADVP_BKT"] = pd.to_numeric(df.get("ADVP_CANON"), errors="coerce")
    d1a = d1a.dropna(subset=["ADVP_BKT"])
    tot_a = (df_advp.assign(CIA_UP=df_advp["CIA_NORM"].astype(str).str.upper())
             .groupby(["CIA_UP", "ADVP_BKT"])["IDPESQUISA"]
             .nunique().reset_index(name="TotAdvp"))
    win_a = (d1a.groupby(["CIA_UP", "ADVP_BKT", "AG_UP"])["IDPESQUISA"]
             .nunique().reset_index(name="QtdTop1"))
    base_a = win_a.merge(tot_a, on=["CIA_UP", "ADVP_BKT"], how="right").fillna({"QtdTop1": 0})
    base_a["Pct"] = (base_a["QtdTop1"] / base_a["TotAdvp"].replace(0, np.nan) * 100).fillna(0.0)
    def pick_leader_advp(cia: str, advp: int):
        sub = base_a[(base_a["CIA_UP"] == cia) & (base_a["ADVP_BKT"] == advp)]
        if sub.empty or (sub["TotAdvp"].sum() == 0):
            return {"CIA": cia, "ADVP": advp, "AGENCIA": "SEM OFERTAS", "PCT": 0.0, "N": 0}
        top = sub.sort_values(["Pct", "QtdTop1", "TotAdvp"], ascending=False).iloc[0]
        return {"CIA": cia, "ADVP": advp, "AGENCIA": str(top["AG_UP"]), "PCT": float(top["Pct"]), "N": int(top["TotAdvp"])}
    def render_tbl_advp_and_cards():
        # tabela
        html = ["<table class='t6'>",
                "<thead><tr><th class='cia'>CIA</th><th style='width:56px'>ADVP</th><th class='ag l'>AGENCIA</th><th class='pct'>% DE GANHO</th></tr></thead><tbody>"]
        for a in buckets:
            rows = [pick_leader_advp(cia, a) for cia in ["AZUL", "GOL", "LATAM"]]
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
        # 3 cards padrão (Resumo do Vencedor) + 3 cards Grupo123
        total_base_trechos = int(df["IDPESQUISA"].nunique() or 0)
        c_nome, c_qtd, c_pct, a_nome, a_qtd, a_pct, base = compute_summary(d1, total_base_trechos)
        st.markdown(cards_block("Resumo do Vencedor", c_nome, c_qtd, c_pct, a_nome, a_qtd, a_pct, base),
                    unsafe_allow_html=True)
        base_advp_n = int(df_advp["IDPESQUISA"].nunique() or 0)
        st.markdown(cards_block_grupo123(d1a, base_advp_n), unsafe_allow_html=True)
    # ===== Render =====
    st.subheader("Competividade  Cia x Trecho x ADVPs Agrupados")
    c1, c2 = st.columns(2, gap="small")
    with c1:
        st.caption("Cia × Trecho")
        render_tbl_trecho()
    with c2:
        st.caption("Cia × ADVP")
        render_tbl_advp_and_cards()

# ─────────────────────── ABA 7: Ofertas x Cias (rótulo com tamanho fixo) ─────
@register_tab("Ofertas x Cias")
def tab7_ofertas_x_cias(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t7")
    st.subheader("Distribuição de Ofertas por Companhia Aérea")
    if df.empty:
        st.info("Sem dados para os filtros selecionados.")
        return
    # ======== ajuste rápido do tamanho do rótulo ========
    LABEL_SIZE = 18
    SHOW_MIN = 0.02
    # ====================================================
    # Cores/ordem fixas
    CIA_DOMAIN = ['AZUL', 'GOL', 'LATAM']
    CIA_COLORS = ['#0033A0', '#FF6600', '#8B0000']
    ORDER_MAP = {'AZUL': 0, 'GOL': 1, 'LATAM': 2}
    base = df.copy()
    base['CIA_NORM'] = base['CIA_NORM'].astype(str).str.upper()
    if 'IDPESQUISA' not in base.columns:
        st.warning("Coluna IDPESQUISA ausente.")
        return
    import numpy as np
    # -------- prepara dados empilhados 0..1 com y0/y1 e centro --------
    def build_stacked(df_in: pd.DataFrame, group_col: str) -> pd.DataFrame:
        counts = (df_in.groupby([group_col, 'CIA_NORM'])['IDPESQUISA']
                  .nunique().reset_index(name='n'))
        if counts.empty:
            return counts
        # garantir 3 cias em todos os grupos
        all_groups = counts[group_col].dropna().unique()
        full_idx = pd.MultiIndex.from_product([all_groups, CIA_DOMAIN],
                                              names=[group_col, 'CIA_NORM'])
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
        # rótulo sem casas decimais
        counts['pct_txt'] = counts['share'].map(lambda x: f"{x*100:.0f}%")
        return counts
    # -------- desenha (barras 100% + rótulo no centro) --------
    def draw_chart(stacked: pd.DataFrame, group_col: str, x_title: str):
        if stacked.empty:
            st.info(f"Sem dados para {x_title} no recorte atual.")
            return
        bars = (
            alt.Chart(stacked)
            .mark_bar()
            .encode(
                x=alt.X(f'{group_col}:N', title=x_title, axis=alt.Axis(labelAngle=0)),
                y=alt.Y('y0:Q', title='Participação', axis=alt.Axis(format='%'),
                        scale=alt.Scale(domain=[0, 1])),
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
            .mark_text(
                color='white',
                fontWeight='bold',
                align='center',
                baseline='middle',
                size=LABEL_SIZE
            )
            .encode(
                x=alt.X(f'{group_col}:N'),
                y=alt.Y('y_center:Q', scale=alt.Scale(domain=[0, 1])),
                text='pct_txt:N',
                detail='CIA_NORM:N'
            )
        )
        st.altair_chart((bars + labels).properties(height=450), use_container_width=True)
    # ===================== 1) ADVP =====================
    st.markdown("#### Participação Cias Por ADVP")
    if 'ADVP_CANON' not in base.columns:
        st.info("Coluna ADVP_CANON não encontrada nos dados.")
    else:
        advp = base.copy()
        advp['ADVP_CANON'] = pd.to_numeric(advp['ADVP_CANON'], errors='coerce')
        advp = advp.dropna(subset=['ADVP_CANON'])
        draw_chart(build_stacked(advp, 'ADVP_CANON'), 'ADVP_CANON', 'ADVP')
    # ===================== 2) TRECHO (Top 15) =====================
    st.markdown("<hr style='margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown("#### Participação Cias Por Trechos")
    top15 = base.groupby('TRECHO')['IDPESQUISA'].nunique().nlargest(15).index
    dft = base[base['TRECHO'].isin(top15)].copy()
    draw_chart(build_stacked(dft, 'TRECHO'), 'TRECHO', 'Trecho')


@register_tab("TABELA DE PESQUISA")
def tab_tabela_pesquisa(df_raw: pd.DataFrame):
    """Tabela detalhada por pesquisa com colunas e downloads (XLSX/CSV).

    Assunções razoáveis:
    - Agrupamos por IDPESQUISA + TRECHO + ADVP_CANON para obter o menor preço por pesquisa/trecho/advp.
    - TRECHO é uma string que contém códigos IATA (ex: 'GRU REC' ou 'GRU-REC'); tentamos extrair os dois códigos.
    """
    df = render_filters(df_raw, key_prefix="t8")
    st.subheader("TABELA DE PESQUISA")
    if df.empty:
        st.info("Sem resultados para os filtros selecionados.")
        return

    work = df.copy()
    work['ADVP_CANON'] = work.get('ADVP_CANON', work.get('ADVP'))
    work['TRECHO'] = work.get('TRECHO', '').astype(str)
    work['AGENCIA_NORM'] = work.get('AGENCIA_NORM', work.get('AGENCIA_COMP', '')).astype(str)
    work['CIA_NORM'] = work.get('CIA_NORM', work.get('CIA', '')).astype(str)

    # Selecionar top 11 trechos por número de pesquisas (IDPESQUISA distintas)
    top_trechos = work.groupby('TRECHO')['IDPESQUISA'].nunique().sort_values(ascending=False).head(11).index.tolist()
    if len(top_trechos) < 11:
        restantes = [t for t in work['TRECHO'].dropna().astype(str).unique().tolist() if t not in top_trechos]
        top_trechos += restantes[: max(0, 11 - len(top_trechos))]

    # ADVP buckets fixos (5 por trecho)
    advp_buckets = [1, 5, 11, 17, 30]

    rows_out = []
    # para cada trecho+advp, escolher a última pesquisa (DATAHORA_BUSCA) e extrair preços mínimos naquela pesquisa
    for trecho in top_trechos:
        for advp in advp_buckets:
            sub = work[(work['TRECHO'] == trecho) & (pd.to_numeric(work['ADVP_CANON'], errors='coerce') == advp)].copy()
            if sub.empty:
                # se não houver essa combinação, pula
                continue
            # escolher a última pesquisa (por DATAHORA_BUSCA ou __DTKEY__)
            if '__DTKEY__' in sub.columns and sub['__DTKEY__'].notna().any():
                last_idx = sub['__DTKEY__'].idxmax()
            else:
                # fallback para DATAHORA_BUSCA
                last_idx = sub['DATAHORA_BUSCA'].dropna().index.max() if sub['DATAHORA_BUSCA'].notna().any() else sub.index.max()
            last_id = sub.loc[last_idx, 'IDPESQUISA'] if pd.notna(last_idx) else None
            if last_id is None:
                continue
            # linhas pertencentes a essa pesquisa (filtrar por IDPESQUISA)
            rows_pesq = work[(work['IDPESQUISA'] == last_id) & (work['TRECHO'] == trecho) & (pd.to_numeric(work['ADVP_CANON'], errors='coerce') == advp)].copy()
            if rows_pesq.empty:
                # fallback: pegar as linhas do trecho+advp independ. de id
                rows_pesq = sub.copy()

            # preço mais barato e empresa responsável
            rows_pesq['PRECO_NUM'] = pd.to_numeric(rows_pesq['PRECO'], errors='coerce')
            if rows_pesq['PRECO_NUM'].notna().any():
                min_price = float(rows_pesq['PRECO_NUM'].min())
                min_row = rows_pesq.loc[rows_pesq['PRECO_NUM'].idxmin()]
                empresa_min = str(min_row.get('AGENCIA_NORM', ''))
                cia_voo = str(min_row.get('CIA_NORM', ''))
            else:
                min_price = np.nan; empresa_min = ''; cia_voo = ''

            # preços por agência específicas (se apareceram nessa pesquisa)
            def get_ag_price(df_sub, ag_name):
                dfx = df_sub[df_sub['AGENCIA_NORM'].str.upper() == ag_name.upper()]
                v = pd.to_numeric(dfx['PRECO'], errors='coerce')
                return float(v.min()) if v.notna().any() else np.nan

            price_123 = get_ag_price(rows_pesq, '123MILHAS')
            price_max = get_ag_price(rows_pesq, 'MAXMILHAS')
            price_flip = get_ag_price(rows_pesq, 'FLIPMILHAS')

            # data/hora e data embarque
            dt_busca = (pd.to_datetime(rows_pesq['DATAHORA_BUSCA'], errors='coerce').max() if 'DATAHORA_BUSCA' in rows_pesq.columns else pd.NaT)
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

    # Extrair ORIGEM e DESTINO seguindo sua regra: se TRECHO estiver concatenado (ex: BELGRU), usar 1-3 e 4-6
    def parse_origem_destino(t: str) -> tuple[str, str]:
        s = str(t or '').strip().upper()
        if len(s) >= 6 and re.match(r'^[A-Z]{6}$', s):
            return s[0:3], s[3:6]
        found = re.findall(r"[A-Z]{3}", s)
        if len(found) >= 2:
            return found[0], found[1]
        # try split by non-alnum
        parts = re.split(r'[^A-Z0-9]+', s)
        parts = [p for p in parts if p]
        if len(parts) >= 2:
            return parts[0][0:3], parts[1][0:3]
        # fallback: first 3 and next 3 if possible
        if len(s) >= 6:
            return s[0:3], s[3:6]
        return s[0:3], ''

    if not out_df.empty:
        out_df['TRECHO ORIGEM'] = out_df['TRECHO'].apply(lambda x: parse_origem_destino(x)[0])
        out_df['TRECHO DESTINO'] = out_df['TRECHO'].apply(lambda x: parse_origem_destino(x)[1])

    # Cálculo %: seguiremos (other - base)/base * 100, usando FLIP como base para 123XFLIP e PRECO (menor) como base para 123 X MENOR PREÇO
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

    # Formatação para exibição
    display_df = out_df.rename(columns={
        'DATAHORA_BUSCA': 'DATA+ HORA DA PESQUISA',
        'TRECHO ORIGEM': 'TRECHO ORIGEM', 'TRECHO DESTINO': 'TRECHO DESTINO',
        'ADVP': 'ADVP', 'DATA_EMBARQUE': 'DATA DE EMBARQUE',
        'PRECO': 'PREÇO', 'CIA_DO_VOO': 'CIA DO VOO', 'EMPRESA': 'EMPRESA',
        '123MILHAS': 'PREÇO 123MILHAS', 'MAXMILHAS': 'PREÇOMAXMILHAS', 'FLIPMILHAS': 'PREÇO FLIPMILHAS'
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

    # Ordenar alfabeticamente por TRECHO ORIGEM conforme solicitado
    if 'TRECHO ORIGEM' in display_df.columns:
        # ordenar por TRECHO ORIGEM (A->Z) e dentro de cada origem por ADVP (menor->maior)
        if 'ADVP' in display_df.columns:
            display_df = display_df.sort_values(by=['TRECHO ORIGEM', 'ADVP'], key=lambda s: s.fillna('').str.upper() if s.dtype == object else s).reset_index(drop=True)
        else:
            display_df = display_df.sort_values(by='TRECHO ORIGEM', key=lambda s: s.fillna('').str.upper()).reset_index(drop=True)

    sty = style_smart_colwise(display_df, {
        'PREÇO': fmt_num0_br,
        'PREÇO 123MILHAS': fmt_num0_br, 'PREÇOMAXMILHAS': fmt_num0_br, 'PREÇO FLIPMILHAS': fmt_num0_br,
        '123XFLIP (%)': fmt_pct0_br, 'MAX X FLIP (%)': fmt_pct0_br, '123 X MENOR PREÇO (%)': fmt_pct0_br,
    }, grad_cols=['PREÇO', 'PREÇO 123MILHAS', 'PREÇOMAXMILHAS', 'PREÇO FLIPMILHAS'])
    show_table(display_df, sty, caption='Tabela (11 trechos × 5 ADVPs = até 55 pesquisas)')
    # Downloads: botão CSV discreto no topo (sempre com separador ",")
    csv_bytes = display_df.to_csv(index=False, sep=',', decimal=',').encode('utf-8')
    # coloca o botão mais discreto no topo direito usando columns
    c1, c2, c3 = st.columns([1, 1, 0.2])
    with c3:
        st.download_button('Baixar CSV', data=csv_bytes, file_name='tabela_pesquisa.csv', mime='text/csv', key='dl_csv_tabela')

    to_xlsx = io.BytesIO()
    try:
        with pd.ExcelWriter(to_xlsx, engine='openpyxl') as writer:
            display_df.to_excel(writer, index=False, sheet_name='TABELA_PESQUISA')
        to_xlsx.seek(0)
        st.download_button('Baixar XLSX', data=to_xlsx.read(), file_name='tabela_pesquisa.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    except Exception:
        # Se não for possível gerar XLSX (biblioteca ausente), não exibir mensagem — CSV já disponível
        pass

# ================================ MAIN ========================================
def main():
    """Função principal que executa o aplicativo Streamlit."""
    df_raw = load_base(DATA_PATH)
    # Tenta exibir uma imagem de logo
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp"):
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
