# streamlit_app.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta, date, time as dtime
from typing import Callable, List, Tuple
import re
import unicodedata as ud

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Flight Deal Scanner — Painel", layout="wide", initial_sidebar_state="expanded")
APP_DIR   = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "OFERTAS.parquet"

# -----------------------------------------------------------------------------
# CSS (tema + cards + tabelas largura total)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
:root{
  --bg:#0b1220; --card:#0f172a; --muted:#0b1324; --text:#e5e7eb; --sub:#94a3b8;
  --primary:#38bdf8; --primary-glow:#60a5fa; --border:#1f2a44; --ok:#10b981; --bad:#ef4444;
}
html, body, .main { background: var(--bg) !important; color: var(--text); }
.block-container{ padding-top:0 !important; }
.stAlert, .stAlert p, .stAlert div { color: #111 !important; }

/* KPI cards */
.kpi { background: var(--card); border:1px solid var(--border); border-radius:14px; padding:16px 18px; }
.kpi .label { font-size:12px; color:var(--sub); }
.kpi .value { font-size:28px; font-weight:800; }
.pill { display:inline-flex; align-items:center; gap:6px; padding:4px 8px; border-radius:999px; font-weight:700; font-size:12px; }
.pill.ok { color:#052e1c; background: #bbf7d0; }
.pill.bad{ color:#3d0a0a; background: #fecaca; }
.card { background: var(--card); border:1px solid var(--border); border-radius:14px; }
.card-hover { transition: transform .2s ease, box-shadow .2s ease; }
.card-hover:hover { transform: translateY(-2px); box-shadow: 0 10px 30px rgba(0,0,0,.25); }
.hstack { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }

/* Tabelas largura total e sem índice visível */
table { width:100% !important; }
.dataframe { width:100% !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
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
    ag = (str(raw or "")).strip().upper()
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
    except Exception: return "-"

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

# -----------------------------------------------------------------------------
# Estilização de tabelas (sem matplotlib)
# -----------------------------------------------------------------------------
BLUE  = "#cfe3ff"; ORANGE= "#fdd0a2"; GREEN = "#c7e9c0"; YELLOW= "#fee391"; PINK = "#f1b6da"
def _hex_to_rgb(h): return tuple(int(h[i:i+2], 16) for i in (1,3,5))
def _rgb_to_hex(t): return f"#{t[0]:02x}{t[1]:02x}{t[2]:02x}"
def _blend(c_from, c_to, t):
    f, to = _hex_to_rgb(c_from), _hex_to_rgb(c_to)
    return _rgb_to_hex(tuple(int(round(f[i] + (to[i]-f[i])*t)) for i in range(3)))
def make_scale(base_hex, steps=5): return [_blend("#ffffff", base_hex, k/(steps-1)) for k in range(steps)]
SCALE_BLUE   = make_scale(BLUE); SCALE_ORANGE = make_scale(ORANGE); SCALE_GREEN = make_scale(GREEN)
SCALE_YELLOW = make_scale(YELLOW); SCALE_PINK = make_scale(PINK)

def _pick_scale(colname: str):
    u = str(colname).upper()
    if "MAX" in u:      return SCALE_GREEN
    if "123" in u:      return SCALE_ORANGE
    if "FLIP" in u:     return SCALE_YELLOW
    if "CAPO" in u:     return SCALE_PINK
    return SCALE_BLUE

def _is_null_like(v) -> bool:
    if v is None: return True
    if isinstance(v, float) and np.isnan(v): return True
    if isinstance(v, str) and v.strip().lower() in {"none", "nan", ""}: return True
    return False

def style_heatmap_discrete(styler: pd.io.formats.style.Styler, col: str, scale_colors: list[str]):
    s = pd.to_numeric(styler.data[col], errors="coerce")
    if s.notna().sum() == 0: return styler
    try:    bins = pd.qcut(s.rank(method="average"), q=5, labels=False, duplicates="drop")
    except: bins = pd.cut(s.rank(method="average"), bins=5, labels=False)
    bins = bins.fillna(-1).astype(int)
    def _fmt(val, idx):
        if pd.isna(val) or bins.iloc[idx] == -1: return "background-color:#ffffff;color:#111111"
        color = scale_colors[int(bins.iloc[idx])]
        return f"background-color:{color};color:#111111"
    return styler.apply(lambda col_vals: [_fmt(v, i) for i, v in enumerate(col_vals)], subset=[col])

def style_smart_colwise(df_show: pd.DataFrame, fmt_map: dict, grad_cols: list[str]):
    sty = (df_show.style
           .set_properties(**{"background-color": "#FFFFFF", "color": "#111111"})
           .set_table_attributes('style="width:100%; table-layout:fixed"'))
    if fmt_map: sty = sty.format(fmt_map, na_rep="-")
    for c in grad_cols:
        if c in df_show.columns:
            sty = style_heatmap_discrete(sty, c, _pick_scale(c))
    try: sty = sty.hide(axis="index")
    except Exception:
        try: sty = sty.hide_index()
        except Exception: pass
    sty = sty.applymap(lambda v: "background-color: #FFFFFF; color: #111111" if _is_null_like(v) else "")
    sty = sty.set_table_styles([{"selector":"tbody td, th","props":[("border","1px solid #EEE")]}])
    return sty

def show_table(df: pd.DataFrame, styler: pd.io.formats.style.Styler | None = None, caption: str | None = None):
    if caption: st.markdown(f"**{caption}**")
    try:
        if styler is not None: st.markdown(styler.to_html(), unsafe_allow_html=True)
        else:                  st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.warning(f"Falha ao aplicar estilo ({e}). Exibindo tabela simples.")
        st.dataframe(df, use_container_width=True)

# -----------------------------------------------------------------------------
# Carregamento robusto da base (aliases + Data(H)+Hora(C))
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_base(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Arquivo obrigatório não encontrado: {path.as_posix()}"); st.stop()
    df = pd.read_parquet(path)

    # Fallback de cabeçalho 0..12
    colmap = {0:"IDPESQUISA",1:"CIA",2:"HORA_BUSCA",3:"HORA_PARTIDA",4:"HORA_CHEGADA",
              5:"TIPO_VOO",6:"DATA_EMBARQUE",7:"DATAHORA_BUSCA",8:"AGENCIA_COMP",9:"PRECO",
              10:"TRECHO",11:"ADVP",12:"RANKING"}
    try:
        first13 = list(df.columns[:13])
        if all(isinstance(c, (int, np.integer, float)) for c in first13):
            rename = {df.columns[i]: colmap[i] for i in range(min(13, df.shape[1]))}
            df = df.rename(columns=rename)
    except Exception:
        pass

    # normalizador de chaves
    def normkey(s: str) -> str:
        s = str(s or "")
        s = ud.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        s = s.lower()
        s = re.sub(r"[^a-z0-9]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    cols_norm_map = {normkey(c): c for c in df.columns}

    alias_sets = {
        "IDPESQUISA": ["idpesquisa","id pesquisa","id busca","idbusca","search id","id"],
        "CIA": ["cia","cia aerea","companhia","airline","ciaerea"],
        "HORA_BUSCA": [
            "hora busca","hora da busca","coluna c",
            "hora da busca posicao coluna c","hora da busca posucao coluna c",  # cobre 'posuÇão'
            "hora c","hora da coleta"
        ],
        "HORA_PARTIDA": ["hora partida","hora de partida","partida hora"],
        "HORA_CHEGADA": ["hora chegada","hora de chegada","chegada hora"],
        "TIPO_VOO": ["tipo voo","tipo","cabine","classe"],
        "DATA_EMBARQUE": ["data embarque","data de embarque","embarque data","dat emb"],
        "DATAHORA_BUSCA": [
            "datahora busca","data hora busca","timestamp","dt busca","data busca","coleta dh",
            "data da busca posicao coluna h","data da busca coluna h","coluna h","data da coleta"
        ],
        "AGENCIA_COMP": ["agencia comp","agencia norm","agencia","agencia compra","agência","agencia_comp"],
        "PRECO": ["preco","preço","valor","amount","price","preco brl","preco rs"],
        "TRECHO": ["trecho","rota","origem destino","od","route","origem destino od"],
        "ADVP": ["advp","antecedencia","antecedencia dias","d0 d30","advp canon"],
        "RANKING": ["ranking","posicao","posição","rank","place"],
    }

    def pick(col_aliases: list[str]) -> str | None:
        for alias in col_aliases:
            a_norm = normkey(alias)
            if a_norm in cols_norm_map:
                return cols_norm_map[a_norm]
            toks = a_norm.split()
            for col_n, original in cols_norm_map.items():
                if all(t in col_n for t in toks):
                    return original
        return None

    selected: dict[str, str] = {}
    missing: list[str] = []
    for canon, alist in alias_sets.items():
        real = pick(alist)
        if real is not None: selected[canon] = real
        else:                missing.append(canon)

    required = ["DATAHORA_BUSCA", "PRECO", "TRECHO"]
    still_missing = [c for c in required if c not in selected]
    if still_missing:
        st.error("Colunas obrigatórias ausentes: " + ", ".join(still_missing))
        with st.expander("Colunas detectadas no arquivo"):
            st.write(list(df.columns))
        st.stop()

    # recorta/renomeia
    df2 = df[list(selected.values())].copy()
    df2.columns = list(selected.keys())

    # opcionais ausentes
    for opt in missing:
        if opt in required: continue
        df2[opt] = np.nan

    # DATA + HORA -> DATAHORA_BUSCA
    df2["DATAHORA_BUSCA"] = pd.to_datetime(df2["DATAHORA_BUSCA"], errors="coerce", dayfirst=True)
    if "HORA_BUSCA" in df2.columns:
        df2["HORA_BUSCA"] = df2["HORA_BUSCA"].apply(_norm_hhmmss)
        mask_comb = df2["DATAHORA_BUSCA"].notna() & df2["HORA_BUSCA"].notna()
        if mask_comb.any():
            dt_part = df2.loc[mask_comb, "DATAHORA_BUSCA"].dt.strftime("%Y-%m-%d")
            df2.loc[mask_comb, "DATAHORA_BUSCA"] = pd.to_datetime(
                dt_part + " " + df2.loc[mask_comb, "HORA_BUSCA"], errors="coerce"
            )
    df2["HORA_HH"] = pd.to_datetime(df2.get("HORA_BUSCA"), errors="coerce").dt.hour.fillna(0).astype(int)

    # preço
    df2["PRECO"] = (
        df2["PRECO"].astype(str)
        .str.replace(r"[^\d,.-]", "", regex=True)
        .str.replace(",", ".", regex=False)
    )
    df2["PRECO"] = pd.to_numeric(df2["PRECO"], errors="coerce")

    # normalizações
    if "RANKING" in df2.columns:
        df2["RANKING"] = pd.to_numeric(df2["RANKING"], errors="coerce").astype("Int64")
    df2["AGENCIA_NORM"] = df2.get("AGENCIA_COMP", pd.Series([None]*len(df2))).apply(std_agencia)
    df2["CIA_NORM"]     = df2.get("CIA",           pd.Series([None]*len(df2))).apply(std_cia)
    df2["ADVP_CANON"]   = (df2.get("ADVP") if "ADVP" in df2.columns else pd.Series([1]*len(df2))).apply(advp_nearest)

    # IDPESQUISA se faltar
    if "IDPESQUISA" not in df2.columns or df2["IDPESQUISA"].isna().all():
        ts = pd.to_datetime(df2["DATAHORA_BUSCA"], errors="coerce").astype("int64")
        df2["IDPESQUISA"] = pd.factorize(ts)[0] + 1

    # limpa linhas inválidas
    df2 = df2.dropna(subset=["DATAHORA_BUSCA","PRECO"]).reset_index(drop=True)

    with st.expander("Detalhes de mapeamento de colunas", expanded=False):
        ok_map = ", ".join([f"{k} ← {v}" for k, v in selected.items()])
        st.caption(f"Mapeadas: {ok_map}")
        if missing:
            st.caption("Ausentes (criadas vazias): " + ", ".join([m for m in missing if m not in required]))
    return df2

# -----------------------------------------------------------------------------
# Registro de abas
# -----------------------------------------------------------------------------
TAB_REGISTRY: List[Tuple[str, Callable]] = []
def register_tab(label: str):
    def _wrap(fn: Callable):
        TAB_REGISTRY.append((label, fn))
        return fn
    return _wrap

# -----------------------------------------------------------------------------
# Filtros (Período, ADVP, Trecho, Hora, CIA)
# -----------------------------------------------------------------------------
def _init_filter_state(df_raw: pd.DataFrame):
    if "flt" in st.session_state: return
    dmin = pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce").min()
    dmax = pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce").max()
    st.session_state["flt"] = {
        "dt_ini": (dmin.date() if pd.notna(dmin) else date(2000, 1, 1)),
        "dt_fim": (dmax.date() if pd.notna(dmax) else date.today()),
        "advp": [], "trechos": [], "hh": [], "cia": [],
    }

def render_filters(df_raw: pd.DataFrame, key_prefix: str = "flt") -> pd.DataFrame:
    _init_filter_state(df_raw)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns([1.4, 1, 1, 1, 1])

    dmin_abs = pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce").min()
    dmax_abs = pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce").max()
    dmin_abs = dmin_abs.date() if pd.notna(dmin_abs) else date(2000, 1, 1)
    dmax_abs = dmax_abs.date() if pd.notna(dmax_abs) else date.today()

    with c1:
        st.caption("Período")
        dt_ini = st.date_input("Data inicial", key=f"{key_prefix}_dtini",
                               value=st.session_state["flt"]["dt_ini"],
                               min_value=dmin_abs, max_value=dmax_abs, format="DD/MM/YYYY")
        dt_fim = st.date_input("Data final", key=f"{key_prefix}_dtfim",
                               value=st.session_state["flt"]["dt_fim"],
                               min_value=dmin_abs, max_value=dmax_abs, format="DD/MM/YYYY")
    with c2:
        st.caption("ADVP")
        advp_all = sorted(set(pd.to_numeric(df_raw["ADVP_CANON"], errors="coerce").dropna().astype(int).tolist()))
        advp_sel = st.multiselect(" ", options=advp_all,
                                  default=st.session_state["flt"]["advp"], key=f"{key_prefix}_advp",
                                  label_visibility="collapsed")
    with c3:
        st.caption("Trecho")
        trechos_all = sorted([t for t in df_raw["TRECHO"].dropna().unique().tolist() if str(t).strip() != ""])
        tr_sel = st.multiselect("  ", options=trechos_all,
                                default=st.session_state["flt"]["trechos"], key=f"{key_prefix}_trechos",
                                label_visibility="collapsed")
    with c4:
        st.caption("Hora")
        hh_sel = st.multiselect("   ", options=list(range(24)),
                                default=st.session_state["flt"]["hh"], key=f"{key_prefix}_hh",
                                label_visibility="collapsed",
                                format_func=lambda x: f"{x:02d}:00")
    with c5:
        st.caption("CIA")
        cia_presentes = set(str(x).upper() for x in df_raw.get("CIA_NORM", pd.Series([], dtype=str)).dropna().unique())
        ordem = ["AZUL", "GOL", "LATAM"]
        cia_opts = [c for c in ordem if (not cia_presentes or c in cia_presentes)] or ordem
        cia_default = [c for c in st.session_state["flt"]["cia"] if c in cia_opts]
        cia_sel = st.multiselect("    ", options=cia_opts, default=cia_default,
                                 key=f"{key_prefix}_cia", label_visibility="collapsed")

    st.session_state["flt"] = {
        "dt_ini": dt_ini, "dt_fim": dt_fim, "advp": advp_sel or [],
        "trechos": tr_sel or [], "hh": hh_sel or [], "cia": cia_sel or []
    }

    mask = pd.Series(True, index=df_raw.index)
    mask &= (pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce") >= pd.Timestamp(dt_ini))
    mask &= (pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce") <= pd.Timestamp(dt_fim) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    if advp_sel: mask &= df_raw["ADVP_CANON"].isin(advp_sel)
    if tr_sel:  mask &= df_raw["TRECHO"].isin(tr_sel)
    if hh_sel:  mask &= df_raw["HORA_HH"].isin(hh_sel)
    if cia_sel: mask &= df_raw["CIA_NORM"].astype(str).str.upper().isin(cia_sel)

    df = df_raw[mask].copy()
    st.caption(f"Linhas após filtros: {fmt_int(len(df))} • Última atualização: {last_update_from_cols(df)}")
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    return df

# -----------------------------------------------------------------------------
# Funções auxiliares de análise
# -----------------------------------------------------------------------------
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

def make_bar(df: pd.DataFrame, x_col: str, y_col: str, sort_y_desc: bool = True):
    d = df[[y_col, x_col]].copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = d[y_col].astype(str)
    d = d.dropna(subset=[x_col])
    if sort_y_desc: d = d.sort_values(x_col, ascending=False)
    if d.empty: return alt.Chart(pd.DataFrame({x_col: [], y_col: []})).mark_bar()
    return (alt.Chart(d)
            .mark_bar()
            .encode(x=alt.X(f"{x_col}:Q", title=x_col), y=alt.Y(f"{y_col}:N", sort="-x", title=y_col),
                    tooltip=[f"{y_col}:N", alt.Tooltip(f"{x_col}:Q", format=",.0f")])
            .properties(height=300))

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
    if color: d[color] = d[color].astype(str)
    d = d.dropna(subset=[x_col, y_col])
    if d.empty: return alt.Chart(pd.DataFrame({x_col: [], y_col: []})).mark_line()
    enc = dict(x=x_enc, y=alt.Y(f"{y_col}:Q", title=y_col), tooltip=[f"{x_col}", alt.Tooltip(f"{y_col}:Q", format=",.0f")])
    if color: enc["color"] = alt.Color(f"{color}:N", title=color)
    return alt.Chart(d).mark_line(point=True).encode(**enc).properties(height=300)

# -----------------------------------------------------------------------------
# ABA: Painel (KPIs + Ranking por CIA + Tendência/Competitividade)
# -----------------------------------------------------------------------------
@register_tab("Painel")
def tab1_painel(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t1")
    st.subheader("Painel")

    # KPIs
    def dd_pct(curr: float, prev: float) -> tuple[float,bool] | None:
        if prev is None or prev==0 or not np.isfinite(prev): return None
        pct = (curr - prev)/prev*100
        return pct, (pct>=0)

    total_pesquisas = df["IDPESQUISA"].nunique()
    total_ofertas   = len(df)
    menor_preco     = df["PRECO"].min()

    last_dt = pd.to_datetime(df["DATAHORA_BUSCA"], errors="coerce").max()
    prev_day = last_dt - timedelta(days=1) if pd.notna(last_dt) else None
    cur = df[pd.to_datetime(df["DATAHORA_BUSCA"]).dt.date == last_dt.date()] if pd.notna(last_dt) else df.iloc[0:0]
    prv = df[pd.to_datetime(df["DATAHORA_BUSCA"]).dt.date == prev_day.date()] if prev_day else df.iloc[0:0]

    pesq_dd = dd_pct(cur["IDPESQUISA"].nunique() or 0, prv["IDPESQUISA"].nunique() or 0) if prev_day else None
    of_dd   = dd_pct(len(cur) or 0, len(prv) or 0) if prev_day else None
    pre_dd  = dd_pct(cur["PRECO"].min() if not cur.empty else np.nan,
                     prv["PRECO"].min() if not prv.empty else np.nan) if prev_day else None

    # Última atualização (Data + Hora col C)
    if not df.empty:
        last_row = df.loc[pd.to_datetime(df["DATAHORA_BUSCA"]).idxmax()]
        last_hh = _norm_hhmmss(last_row.get("HORA_BUSCA")) or pd.to_datetime(last_row["DATAHORA_BUSCA"]).strftime("%H:%M:%S")
        last_label = f"{pd.to_datetime(last_row['DATAHORA_BUSCA']).strftime('%d/%m/%Y')} {last_hh}"
    else:
        last_label = "—"

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
            <div class="value">{fmt_int(total_pesquisas)}</div>
            {pill(pesq_dd)}
          </div>
          <div class="label">Variação vs dia anterior</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi card-hover">
          <div class="label">Total de Ofertas</div>
          <div class="hstack" style="justify-content:space-between;">
            <div class="value">{fmt_int(total_ofertas)}</div>
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

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Ranking por CIA (1º/2º/3º %)
    st.subheader("Ranking de Agências — por CIA")
    base = (df.groupby(["IDPESQUISA","CIA_NORM","AGENCIA_NORM"], as_index=False)
              .agg(PRECO_MIN=("PRECO","min")))
    def ranking_por_cia(df_in: pd.DataFrame, cia: str) -> pd.DataFrame:
        sub = df_in[df_in["CIA_NORM"]==cia].copy()
        if sub.empty: return pd.DataFrame(columns=["Agência","1º%","2º%","3º%"])
        pos_rows = []
        for _, g in sub.groupby(["IDPESQUISA"]):
            g = g.sort_values("PRECO_MIN").reset_index(drop=True)
            for i in range(min(3, len(g))):
                pos_rows.append({"Agência": g.loc[i,"AGENCIA_NORM"], "pos": i+1})
        pos_df = pd.DataFrame(pos_rows)
        total_ids = sub["IDPESQUISA"].nunique() or 1
        agg = (pos_df.pivot_table(index="Agência", columns="pos", values="pos", aggfunc="count", fill_value=0)
                .reindex(columns=[1,2,3], fill_value=0)
                .rename(columns={1:"1º%",2:"2º%",3:"3º%"}))
        agg = (agg/total_ids*100).reset_index()
        for c in ["1º%","2º%","3º%"]:
            agg[c] = agg[c].map(lambda v: f"{v:.2f}".replace(".", ",") + "%")
        return agg.sort_values("1º%", ascending=False)

    c1,c2,c3 = st.columns(3)
    for cia, col in zip(["AZUL","GOL","LATAM"], [c1,c2,c3]):
        with col:
            st.markdown(f"**{cia}**")
            tbl = ranking_por_cia(base, cia)
            if tbl.empty: st.caption("Sem dados neste recorte.")
            else:         st.dataframe(tbl, use_container_width=True, hide_index=True)

    # Tendência por Hora — 4 principais agências
    st.subheader("Tendência de Preços por Hora — 4 principais agências")
    ag_principais = ["123MILHAS","MAXMILHAS","FLIPMILHAS","CAPOVIAGENS"]
    buckets = []
    for h in range(24):
        row = {"Hora": f"{h:02d}"}
        subset = df[df["HORA_HH"]==h]
        for ag in ag_principais:
            m = subset.loc[subset["AGENCIA_NORM"]==ag, "PRECO"]
            row[ag] = float(m.min()) if not m.empty else None
        buckets.append(row)
    df_hour = pd.DataFrame(buckets).melt(id_vars="Hora", var_name="Agência", value_name="Preço")
    st.altair_chart(make_line(df_hour, "Hora", "Preço", color="Agência"), use_container_width=True)

    # Competitividade — participação das CIAs no menor preço
    st.subheader("Análise de Competitividade — participação das CIAs no menor preço")
    ids = df["IDPESQUISA"].unique().tolist(); wins = {"AZUL":0,"GOL":0,"LATAM":0}
    for idp in ids:
        sub = df[df["IDPESQUISA"]==idp]
        if sub.empty: continue
        best_row = sub.loc[sub["PRECO"].idxmin()]
        cia = str(best_row["CIA_NORM"]); 
        if cia in wins: wins[cia]+=1
    total = sum(wins.values()) or 1
    comp_df = pd.DataFrame([{"CIA":"AZUL","Share":wins["AZUL"]/total*100},
                            {"CIA":"GOL","Share":wins["GOL"]/total*100},
                            {"CIA":"LATAM","Share":wins["LATAM"]/total*100}])
    st.altair_chart(make_bar(comp_df.rename(columns={"Share":"% de vezes mais barata"}), "% de vezes mais barata", "CIA"),
                    use_container_width=True)

# -----------------------------------------------------------------------------
# ABA: Top 3 Preços Mais Baratos (mesma pesquisa + Data/Hora + 123/MAX status)
# -----------------------------------------------------------------------------
@register_tab("Top 3 Preços Mais Baratos")
def tab2_top3_precos(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t2")
    if df.empty:
        st.info("Sem resultados para os filtros selecionados."); return

    st.subheader("Top 3 Preços por Trecho (última pesquisa por Trecho/ADVP)")

    dfp = df.copy()
    dfp["PAR"] = dfp["TRECHO"].astype(str) + "||" + dfp["ADVP_CANON"].astype(str)
    # pega a pesquisa mais recente por Trecho/ADVP
    last_per_pair = dfp.groupby("PAR")["DATAHORA_BUSCA"].transform("max")==dfp["DATAHORA_BUSCA"]
    df_last = dfp[last_per_pair].copy()

    for par, sub in df_last.groupby("PAR"):
        trecho, advp = par.split("||")
        # garante mesma pesquisa (ID) do registro mais recente dentro do par
        id_last = sub.loc[pd.to_datetime(sub["DATAHORA_BUSCA"]).idxmax(), "IDPESQUISA"]
        sub_id = sub[sub["IDPESQUISA"]==id_last].copy()

        # carimbo Data + Hora (C)
        data_hora = pd.to_datetime(sub_id["DATAHORA_BUSCA"]).max()
        hh_colC = None
        for v in sub_id["HORA_BUSCA"].tolist():
            hh_colC = _norm_hhmmss(v)
            if hh_colC: break
        label = f"{data_hora.strftime('%d/%m/%Y')} {hh_colC or data_hora.strftime('%H:%M:%S')}"

        # top3 por agência (preço mínimo da pesquisa)
        best_by_ag = (sub_id.groupby("AGENCIA_NORM", as_index=False)["PRECO"].min()
                            .sort_values("PRECO", ascending=True).head(3).reset_index(drop=True))

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
              <div style="font-size:12px;color:var(--sub)">Data/Hora da busca: {label}</div>
            </div>
            <button class="pill" style="background:#0b1324;border:1px solid var(--border);color:#cbd5e1;"
                    onclick="navigator.clipboard.writeText('{id_last}')" title="Copiar ID da pesquisa">
              ID: {id_last}
            </button>
          </div>
        """, unsafe_allow_html=True)

        for i in range(len(best_by_ag)):
            ag = best_by_ag.iloc[i]["AGENCIA_NORM"]
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

        # 123Milhas / MaxMilhas: se não aparecem no Top3, mostrar % vs 1º ou "não apareceu"
        for alvo in ["123MILHAS", "MAXMILHAS"]:
            preco_alvo = sub_id.loc[sub_id["AGENCIA_NORM"]==alvo, "PRECO"].min()
            if pd.isna(preco_alvo):
                st.markdown(f"<div style='font-size:12px;color:var(--sub);margin-top:6px;'>• {alvo}: não apareceu</div>", unsafe_allow_html=True)
            else:
                pct_vs_1 = ((preco_alvo - top1)/top1*100) if np.isfinite(top1) else np.nan
                st.markdown(
                    f"<div style='font-size:12px;color:var(--sub);margin-top:4px;'>• {alvo}: {fmt_pct2_br(pct_vs_1)} vs 1º</div>",
                    unsafe_allow_html=True
                )

        st.markdown("</div>", unsafe_allow_html=True)

    # Tabela consolidada (opcional)
    st.markdown("—")

# -----------------------------------------------------------------------------
# ABA: Top 3 Agências (resumo por trecho)
# -----------------------------------------------------------------------------
@register_tab("Top 3 Agências")
def tab3_top3_agencias(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t3")
    if df.empty:
        st.info("Sem resultados para os filtros selecionados."); return

    # Para cada Trecho pega a última pesquisa
    df2 = df.copy()
    df2["DT"] = pd.to_datetime(df2["DATAHORA_BUSCA"], errors="coerce")
    g = (df2.dropna(subset=["TRECHO","IDPESQUISA","DT"])
              .groupby(["TRECHO","IDPESQUISA"], as_index=False)["DT"].max())
    last_idx = g.groupby("TRECHO")["DT"].idxmax()
    last_ids = {r["TRECHO"]: r["IDPESQUISA"] for _, r in g.loc[last_idx].iterrows()}
    df2["__ID_TARGET__"] = df2["TRECHO"].map(last_ids)
    df_last = df2[df2["IDPESQUISA"].astype(str) == df2["__ID_TARGET__"].astype(str)].copy()

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
    show_table(t1, sty1, caption="Ranking Top 3 (Agências) — por trecho (última pesquisa)")

# -----------------------------------------------------------------------------
# ABA: Ranking por Agências (Top 15 por vitórias em 1º)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# ABA: Qtde de Buscas x Ofertas
# -----------------------------------------------------------------------------
@register_tab("Qtde de Buscas x Ofertas")
def tab5_buscas_vs_ofertas(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t5")
    st.subheader("Quantidade de Buscas x Ofertas")
    searches = df["IDPESQUISA"].nunique(); offers = len(df)
    c1, c2 = st.columns(2)
    c1.metric("Pesquisas únicas", fmt_int(searches))
    c2.metric("Ofertas (linhas)", fmt_int(offers))
    t = pd.DataFrame({"Métrica": ["Pesquisas", "Ofertas"], "Valor": [searches, offers]})
    st.altair_chart(make_bar(t, "Valor", "Métrica"), use_container_width=True)

# -----------------------------------------------------------------------------
# ABA: Comportamento Cias (share por Trecho)
# -----------------------------------------------------------------------------
@register_tab("Comportamento Cias")
def tab6_comportamento_cias(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t6")
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

# -----------------------------------------------------------------------------
# ABA: Competitividade (Δ mediano vs melhor preço por pesquisa)
# -----------------------------------------------------------------------------
@register_tab("Competitividade")
def tab7_competitividade(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t7")
    st.subheader("Competitividade (Δ mediano vs melhor preço por pesquisa)")
    best = df.groupby("IDPESQUISA")["PRECO"].min().rename("BEST").reset_index()
    t = df.merge(best, on="IDPESQUISA", how="left")
    t["DELTA"] = t["PRECO"] - t["BEST"]
    agg = t.groupby("AGENCIA_NORM", as_index=False)["DELTA"].median().rename(columns={"DELTA":"Δ Mediano"})
    st.altair_chart(make_bar(agg, "Δ Mediano", "AGENCIA_NORM"), use_container_width=True)

# -----------------------------------------------------------------------------
# ABA: Melhor Preço Diário
# -----------------------------------------------------------------------------
@register_tab("Melhor Preço Diário")
def tab8_melhor_preco_diario(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t8")
    st.subheader("Melhor Preço Diário (pela Data/Hora da busca)")
    t = df.groupby(df["DATAHORA_BUSCA"].dt.date, as_index=False)["PRECO"].min().rename(
        columns={"DATAHORA_BUSCA":"Data","PRECO":"Melhor Preço"}
    )
    if t.empty: st.info("Sem dados."); return
    t["Data"] = pd.to_datetime(t["Data"], dayfirst=True)
    st.altair_chart(make_line(t, "Data", "Melhor Preço"), use_container_width=True)

# -----------------------------------------------------------------------------
# ABA: Exportar
# -----------------------------------------------------------------------------
@register_tab("Exportar")
def tab9_exportar(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t9")
    st.subheader("Exportar dados filtrados")
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Baixar CSV (filtro aplicado)", data=csv_bytes, file_name="OFERTAS_filtrado.csv", mime="text/csv")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    df_raw = load_base(DATA_PATH)

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
