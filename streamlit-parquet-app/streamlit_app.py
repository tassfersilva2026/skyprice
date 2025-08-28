# streamlit_app.py
from __future__ import annotations

from pathlib import Path
from datetime import date, datetime, time as dtime, timedelta
from typing import Callable, List, Tuple
import re

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ─────────────────────────── CONFIG E CONSTANTES ─────────────────────────────
st.set_page_config(page_title="Skyscanner — Painel", layout="wide", initial_sidebar_state="expanded")
APP_DIR   = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "OFERTAS.parquet"

# Plotly opcional (cai para Altair se não estiver instalado)
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:  # sem plotly -> usamos apenas Altair
    HAS_PLOTLY = False


# ───────────────────────────── UTIL / NORMALIZAÇÃO ───────────────────────────
def _norm_hhmmss(v: object) -> str | None:
    """Aceita '8:5', '08:05', '08:05:00' etc e devolve HH:MM:SS."""
    s = str(v or "").strip()
    m = re.search(r"(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?", s)
    if not m:
        return None
    hh = max(0, min(23, int(m.group(1))))
    mm = max(0, min(59, int(m.group(2))))
    ss = max(0, min(59, int(m.group(3) or 0)))
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def std_agencia(raw: str) -> str:
    ag = (str(raw) or "").strip().upper().replace(" ", "")
    if ag in {"", "NAN", "NONE", "NULL", "SKYSCANNER"}: return "SEM OFERTAS"
    if ag == "BOOKINGCOM":  return "BOOKING.COM"
    if ag == "KIWICOM":     return "KIWI.COM"
    if ag.startswith("123MILHAS") or ag == "123":  return "123MILHAS"
    if ag.startswith("MAXMILHAS") or ag == "MAX":  return "MAXMILHAS"
    if ag.startswith("CAPOVIAGENS"): return "CAPOVIAGENS"
    if ag.startswith("FLIPMILHAS"):  return "FLIPMILHAS"
    if ag.startswith("VAIDEPROMO"):  return "VAIDEPROMO"
    if ag.startswith("KISSANDFLY"):  return "KISSANDFLY"
    if ag.startswith("ZUPPER"):      return "ZUPPER"
    if ag.startswith("MYTRIP"):      return "MYTRIP"
    if ag.startswith("GOTOGATE"):    return "GOTOGATE"
    if ag.startswith("DECOLAR"):     return "DECOLAR"
    if ag.startswith("EXPEDIA"):     return "EXPEDIA"
    if ag.startswith("TRIPCOM"):     return "TRIP.COM"
    if ag.startswith("VIAJANET"):    return "VIAJANET"
    if ag.startswith("GOL"):         return "GOL"
    if ag.startswith("LATAM"):       return "LATAM"
    return (raw or "").strip().upper()


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
    return f"{int(n):,}".replace(",", ".")


def fmt_num0_br(x):
    try:
        v = float(x)
        if not np.isfinite(v):
            return "-"
        return f"{v:,.0f}".replace(",", ".")
    except Exception:
        return "-"


def fmt_pct2_br(v):
    try:
        x = float(v)
        if not np.isfinite(x):
            return "-"
        return f"{x:.2f}%".replace(".", ",")
    except Exception:
        return "-"


def last_update_from_cols(df: pd.DataFrame) -> str:
    if df.empty:
        return "—"
    max_d = pd.to_datetime(df["DATAHORA_BUSCA"], errors="coerce").max()
    if pd.isna(max_d):
        return "—"
    same_day = df[pd.to_datetime(df["DATAHORA_BUSCA"], errors="coerce").dt.date == max_d.date()]
    hh = pd.to_datetime(same_day["HORA_BUSCA"], errors="coerce").dt.time
    max_h = max([h for h in hh if pd.notna(h)], default=None)
    if isinstance(max_h, dtime):
        return f"{max_d.strftime('%d/%m/%Y')} - {max_h.strftime('%H:%M:%S')}"
    return f"{max_d.strftime('%d/%m/%Y')}"


# ─────────────────────────── CARREGAMENTO ROBUSTO ────────────────────────────
@st.cache_data(show_spinner=True)
def load_base(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Arquivo obrigatório não encontrado: {path.as_posix()}")
        st.stop()

    df = pd.read_parquet(path)

    # Fallback: se as 13 primeiras colunas vierem como 0..12, renomeia
    colmap = {
        0: "IDPESQUISA", 1: "CIA", 2: "HORA_BUSCA", 3: "HORA_PARTIDA", 4: "HORA_CHEGADA",
        5: "TIPO_VOO", 6: "DATA_EMBARQUE", 7: "DATAHORA_BUSCA", 8: "AGENCIA_COMP", 9: "PRECO",
        10: "TRECHO", 11: "ADVP", 12: "RANKING"
    }
    try:
        first13 = list(df.columns[:13])
        if all(isinstance(c, (int, np.integer, float)) for c in first13):
            rename = {df.columns[i]: colmap[i] for i in range(min(13, df.shape[1]))}
            df = df.rename(columns=rename)
    except Exception:
        pass

    # Aliases (case-insensitive)
    aliases: dict[str, list[str]] = {
        "IDPESQUISA": ["IDPESQUISA", "ID_PESQUISA", "ID-BUSCA", "IDBUSCA", "ID", "SEARCH_ID"],
        "CIA": ["CIA", "CIA_NORM", "CIAEREA", "COMPANHIA", "CIA_AEREA", "AIRLINE"],
        "HORA_BUSCA": ["HORA_BUSCA", "HORA DA BUSCA", "HORA_COLETA", "COLUNA_C", "C", "HORA"],
        "HORA_PARTIDA": ["HORA_PARTIDA", "HORA DE PARTIDA", "PARTIDA_HORA"],
        "HORA_CHEGADA": ["HORA_CHEGADA", "HORA DE CHEGADA", "CHEGADA_HORA"],
        "TIPO_VOO": ["TIPO_VOO", "TIPO", "CABINE", "CLASSE"],
        "DATA_EMBARQUE": ["DATA_EMBARQUE", "DATA DE EMBARQUE", "EMBARQUE_DATA", "DAT_EMB"],
        "DATAHORA_BUSCA": ["DATAHORA_BUSCA", "DATA_HORA_BUSCA", "TIMESTAMP", "DT_BUSCA", "DATA_BUSCA", "COLETA_DH"],
        "AGENCIA_COMP": ["AGENCIA_COMP", "AGENCIA_NORM", "AGENCIA", "AGENCIA_COMPRA", "AGÊNCIA"],
        "PRECO": ["PRECO", "PREÇO", "PRICE", "VALOR", "AMOUNT"],
        "TRECHO": ["TRECHO", "ROTA", "ORIGEM-DESTINO", "OD", "ORIGEM_DESTINO", "ROUTE"],
        "ADVP": ["ADVP", "ADVP_CANON", "ANTECEDENCIA", "ANTECEDENCIA_DIAS", "D0_D30"],
        "RANKING": ["RANKING", "POSICAO", "POSIÇÃO", "RANK", "PLACE"],
    }

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
        st.error("Colunas obrigatórias ausentes: " + ", ".join(still_missing))
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
    for c in ["HORA_BUSCA", "HORA_PARTIDA", "HORA_CHEGADA"]:
        if c in df2.columns:
            # preserva HH:MM:SS mesmo se originalmente texto
            df2[c] = df2[c].apply(_norm_hhmmss)

    # hora (HH) para filtro por hora
    hh = pd.to_datetime(df2["HORA_BUSCA"], errors="coerce").dt.hour
    df2["HORA_HH"] = hh.fillna(0).astype(int)

    # preço
    df2["PRECO"] = (
        df2["PRECO"].astype(str)
        .str.replace(r"[^\d,.-]", "", regex=True)
        .str.replace(",", ".", regex=False)
    )
    df2["PRECO"] = pd.to_numeric(df2["PRECO"], errors="coerce")

    # tipos/normais
    if "RANKING" in df2.columns:
        df2["RANKING"] = pd.to_numeric(df2["RANKING"], errors="coerce").astype("Int64")

    df2["AGENCIA_NORM"] = df2.get("AGENCIA_COMP", pd.Series([None]*len(df2))).apply(std_agencia)
    df2["CIA_NORM"]     = df2.get("CIA",           pd.Series([None]*len(df2))).apply(std_cia)
    df2["ADVP_CANON"]   = (df2.get("ADVP") if "ADVP" in df2.columns else pd.Series([1]*len(df2))).apply(advp_nearest)

    # Se IDPESQUISA não veio, gera um estável pelo timestamp
    if "IDPESQUISA" not in df2.columns or df2["IDPESQUISA"].isna().all():
        ts = pd.to_datetime(df2["DATAHORA_BUSCA"], errors="coerce").astype("int64")
        df2["IDPESQUISA"] = pd.factorize(ts)[0] + 1

    # limpa
    df2 = df2.dropna(subset=["DATAHORA_BUSCA", "PRECO"]).reset_index(drop=True)

    # debug amigável
    with st.expander("Detalhes de mapeamento de colunas", expanded=False):
        ok_map = ", ".join([f"{k} ← {v}" for k, v in selected.items()])
        st.caption(f"Mapeadas: {ok_map}")
        if missing:
            st.caption("Ausentes (criadas vazias): " + ", ".join([m for m in missing if m not in required]))
    return df2


# ─────────────────────────── ESTILO DE TABELAS ───────────────────────────────
GLOBAL_TABLE_CSS = """
<style>
table { width:100% !important; }
.dataframe { width:100% !important; }
</style>
"""
st.markdown(GLOBAL_TABLE_CSS, unsafe_allow_html=True)

def style_smart_colwise(df_show: pd.DataFrame, fmt_map: dict, grad_cols: list[str]):
    """Styler simples: formatações e leve 'heatmap' por quantis nas colunas de grad_cols."""
    BLUE  = "#cfe3ff"; ORANGE = "#fdd0a2"; GREEN = "#c7e9c0"; YELLOW = "#fee391"; PINK  = "#f1b6da"
    def _hex_to_rgb(h): return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))
    def _rgb_to_hex(t): return f"#{t[0]:02x}{t[1]:02x}{t[2]:02x}"
    def _blend(c_from, c_to, t):
        f, to = _hex_to_rgb(c_from), _hex_to_rgb(c_to)
        return _rgb_to_hex(tuple(int(round(f[i] + (to[i]-f[i])*t)) for i in range(3)))
    def make_scale(base_hex, steps=5): return [_blend("#ffffff", base_hex, k/(steps-1)) for k in range(steps)]
    def _pick_scale(colname: str):
        u = str(colname).upper()
        if "MAXMILHAS" in u:   return make_scale(GREEN)
        if "123"       in u:   return make_scale(ORANGE)
        if "FLIP"      in u:   return make_scale(YELLOW)
        if "CAPO"      in u:   return make_scale(PINK)
        return make_scale(BLUE)

    sty = (df_show.style
           .set_properties(**{"background-color": "#FFFFFF", "color": "#111111"})
           .set_table_attributes('style="width:100%; table-layout:fixed"'))
    if fmt_map:
        sty = sty.format(fmt_map, na_rep="-")
    # heatmap por quantis
    for c in grad_cols:
        if c in df_show.columns:
            s = pd.to_numeric(df_show[c], errors="coerce")
            if s.notna().sum():
                try:
                    bins = pd.qcut(s.rank(method="average"), q=5, labels=False, duplicates="drop")
                except Exception:
                    bins = pd.cut(s.rank(method="average"), bins=5, labels=False)
                bins = bins.fillna(-1).astype(int)
                scale = _pick_scale(c)
                def _fmt(val, idx):
                    if pd.isna(val) or bins.iloc[idx] == -1: return "background-color:#ffffff;color:#111111"
                    color = scale[int(bins.iloc[idx])]
                    return f"background-color:{color};color:#111111"
                sty = sty.apply(lambda col_vals: [_fmt(v, i) for i, v in enumerate(col_vals)], subset=[c])
    try:
        sty = sty.hide(axis="index")
    except Exception:
        try: sty = sty.hide_index()
        except Exception: pass
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


# ───────────────────────── REGISTRO DE ABAS (DECORATOR) ──────────────────────
TAB_REGISTRY: List[Tuple[str, Callable]] = []
def register_tab(label: str):
    def _wrap(fn: Callable):
        TAB_REGISTRY.append((label, fn))
        return fn
    return _wrap


# ────────────────────────────── FILTROS (UI + MASK) ──────────────────────────
def _init_filter_state(df_raw: pd.DataFrame):
    if "flt" in st.session_state:
        return
    dmin = pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce").min()
    dmax = pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce").max()
    st.session_state["flt"] = {
        "dt_ini": (dmin.date() if pd.notna(dmin) else date(2000, 1, 1)),
        "dt_fim": (dmax.date() if pd.notna(dmax) else date.today()),
        "advp": [], "trechos": [], "hh": [], "cia": [],
    }

def render_filters(df_raw: pd.DataFrame, key_prefix: str = "flt"):
    _init_filter_state(df_raw)
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
                                default=st.session_state["flt"]["hh"], key=f"{key_prefix}_hh",
                                format_func=lambda x: f"{x:02d}:00")
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


# ─────────────────────────── HELPERS PARA AS ABAS ────────────────────────────
def winners_by_position(df: pd.DataFrame) -> pd.DataFrame:
    """Para cada IDPESQUISA, pega o vencedor do ranking 1/2/3 (quando existir)."""
    base = pd.DataFrame({"IDPESQUISA": df["IDPESQUISA"].unique()})
    for r in (1, 2, 3):
        s = (df[df["RANKING"] == r]
             .sort_values(["IDPESQUISA"])
             .drop_duplicates(subset=["IDPESQUISA"]))
        base = base.merge(
            s[["IDPESQUISA", "AGENCIA_NORM"]].rename(columns={"AGENCIA_NORM": f"R{r}"}),
            on="IDPESQUISA", how="left"
        )
    for r in (1, 2, 3): base[f"R{r}"] = base[f"R{r}"].fillna("SEM OFERTAS")
    return base


# ─────────────────────────────── ABAS ────────────────────────────────────────
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

    # cards de participação por agência (com grupo 123)
    W = winners_by_position(df)
    Wg = W.replace({"R1": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
                    "R2": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
                    "R3": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"}})

    agencias_all = sorted(set(df["AGENCIA_NORM"].dropna().astype(str)))
    targets_base = list(agencias_all)
    if "GRUPO 123" not in targets_base: targets_base.insert(0, "GRUPO 123")
    if "SEM OFERTAS" not in targets_base: targets_base.append("SEM OFERTAS")

    def pcts_for_target(base_df: pd.DataFrame, tgt: str, agrupado: bool) -> tuple[float, float, float]:
        base = (base_df.replace({"R1": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
                                 "R2": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
                                 "R3": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"}})
                if agrupado else base_df)
        p1 = float((base["R1"] == tgt).mean())*100
        p2 = float((base["R2"] == tgt).mean())*100
        p3 = float((base["R3"] == tgt).mean())*100
        return p1, p2, p3

    def card_html(nome: str, p1: float, p2: float, p3: float, rank_cls: str = "") -> str:
        cls = f"card {rank_cls}".strip()
        return (
            f"<div class='{cls}' style='border:1px solid #e9e9ee;border-radius:14px;padding:10px 12px;margin-bottom:8px;'>"
            f"<div class='title' style='font-weight:650;margin-bottom:6px'>{nome}</div>"
            f"<div class='row' style='display:flex;gap:8px'>"
            f"<div class='item' style='flex:1;display:flex;justify-content:space-between;border:1px solid #eee;padding:6px 8px;border-radius:10px'><span>1º</span><b>{p1:.2f}%</b></div>"
            f"<div class='item' style='flex:1;display:flex;justify-content:space-between;border:1px solid #eee;padding:6px 8px;border-radius:10px'><span>2º</span><b>{p2:.2f}%</b></div>"
            f"<div class='item' style='flex:1;display:flex;justify-content:space-between;border:1px solid #eee;padding:6px 8px;border-radius:10px'><span>3º</span><b>{p3:.2f}%</b></div>"
            f"</div></div>"
        )

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
    st.markdown(f"<div class='cards-grid' style='display:grid;grid-template-columns:repeat(3, minmax(0,1fr));gap:10px'>{''.join(cards)}</div>", unsafe_allow_html=True)

    # Ranking por CIA
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.subheader("Painel por Cia")

    def render_por_cia(df_in: pd.DataFrame, cia_name: str):
        st.markdown(f"**Ranking {cia_name}**")
        sub = df_in[df_in["CIA_NORM"].astype(str).str.upper() == cia_name]
        if sub.empty:
            st.info("Sem dados para os filtros atuais.")
            return
        Wc = winners_by_position(sub)
        Wc_g = Wc.replace({"R1": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
                           "R2": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
                           "R3": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"}})
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
        st.markdown("".join(cards_local), unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: render_por_cia(df, "AZUL")
    with c2: render_por_cia(df, "GOL")
    with c3: render_por_cia(df, "LATAM")


@register_tab("Top 3 Agências")
def tab2_top3_agencias(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t2")
    if df.empty:
        st.info("Sem resultados para os filtros selecionados.")
        return

    # Garante mesma pesquisa por Trecho: pega o último IDPESQUISA de cada Trecho
    df2 = df.copy()
    df2["DT"] = pd.to_datetime(df2["DATAHORA_BUSCA"], errors="coerce")
    g = (df2.dropna(subset=["TRECHO", "IDPESQUISA", "DT"])
              .groupby(["TRECHO", "IDPESQUISA"], as_index=False)["DT"].max())
    last_idx = g.groupby("TRECHO")["DT"].idxmax()
    last_ids = {r["TRECHO"]: r["IDPESQUISA"] for _, r in g.loc[last_idx].iterrows()}
    df2["__ID_TARGET__"] = df2["TRECHO"].map(last_ids)
    df_last = df2[df2["IDPESQUISA"].astype(str) == df2["__ID_TARGET__"].astype(str)].copy()

    # col "Data/Hora Busca": DATAHORA + hora da coluna C (texto HH:MM:SS)
    def _compose_dt_hora(sub: pd.DataFrame) -> str:
        d = pd.to_datetime(sub["DATAHORA_BUSCA"], errors="coerce").max()
        hh = None
        for v in sub["HORA_BUSCA"].tolist():
            hh = _norm_hhmmss(v); 
            if hh: break
        if pd.isna(d) and not hh:
            return "-"
        if not hh:
            hh = pd.to_datetime(d, errors="coerce").strftime("%H:%M:%S")
        return f"{d.strftime('%d/%m/%Y')} {hh}"

    dt_by_trecho = {trecho: _compose_dt_hora(sub) for trecho, sub in df_last.groupby("TRECHO")}

    # Top3 por trecho
    PRICE_COL, TRECHO_COL, AGENCIA_COL = "PRECO", "TRECHO", "AGENCIA_NORM"
    by_ag = (df_last.groupby([TRECHO_COL, AGENCIA_COL], as_index=False)
                   .agg(PRECO_MIN=(PRICE_COL, "min"))
                   .rename(columns={TRECHO_COL: "TRECHO_STD", AGENCIA_COL: "AGENCIA_UP"}))

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
    for c in ["Preço Top 1", "Preço Top 2", "Preço Top 3"]:
        t1[c] = pd.to_numeric(t1[c], errors="coerce")
    sty1 = style_smart_colwise(t1, {c: fmt_num0_br for c in ["Preço Top 1", "Preço Top 2", "Preço Top 3"]},
                               grad_cols=["Preço Top 1", "Preço Top 2", "Preço Top 3"])
    show_table(t1, sty1, caption="Ranking Top 3 (Agências) — por trecho")

    # % Diferença vs Top1
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

    # Comparativo Cia × Agências de milhas (mesma pesquisa)
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

    # % Dif. vs menor valor por Cia
    def pct_vs_base(b, x):
        if pd.isna(b) or b == 0 or pd.isna(x): return np.nan
        return (x - b) / b * 100

    t4 = t3[["Data/Hora Busca","Trecho","Cia Menor Preço","Preço Menor Valor"]].copy()
    for label in ["123milhas","Maxmilhas","FlipMilhas","Capo Viagens"]:
        t4[f"% Dif {label}"] = [pct_vs_base(b, x) for b, x in zip(t3["Preço Menor Valor"], t3[label])]
    fmt4 = {"Preço Menor Valor": fmt_num0_br} | {c: fmt_pct2_br for c in t4.columns if c.startswith("% Dif ")}
    sty4 = style_smart_colwise(t4.reset_index(drop=True), fmt4, grad_cols=list(fmt4.keys()))
    show_table(t4, sty4, caption="%Comparativo Menor Preço Cia × Agências de Milhas")


@register_tab("Ranking por Agências")
def tab4_ranking_agencias(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t4")
    st.subheader("Ranking por Agências (1º ao 15º)")
    if df.empty:
        st.info("Sem dados para os filtros.")
        return
    wins = (df[df["RANKING"].eq(1)].groupby("AGENCIA_NORM", as_index=False)
            .size().rename(columns={"size":"Top1 Wins"}))
    wins = wins.sort_values("Top1 Wins", ascending=False)
    top15 = wins.head(15).reset_index(drop=True)
    sty = style_smart_colwise(top15, {"Top1 Wins": fmt_num0_br}, grad_cols=["Top1 Wins"])
    show_table(top15, sty, caption="Top 15 — Contagem de 1º lugar por Agência")
    chart = alt.Chart(top15).mark_bar().encode(
        x=alt.X("Top1 Wins:Q", title="Top1 Wins"),
        y=alt.Y("AGENCIA_NORM:N", sort="-x", title="Agência"),
        tooltip=["AGENCIA_NORM", "Top1 Wins"]
    ).properties(height=320)
    st.altair_chart(chart, use_container_width=True)


@register_tab("Qtde de Buscas x Ofertas")
def tab6_buscas_vs_ofertas(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t6")
    st.subheader("Quantidade de Buscas x Ofertas")
    searches = df["IDPESQUISA"].nunique(); offers = len(df)
    c1, c2 = st.columns(2)
    c1.metric("Pesquisas únicas", fmt_int(searches))
    c2.metric("Ofertas (linhas)", fmt_int(offers))
    t = pd.DataFrame({"Métrica": ["Pesquisas", "Ofertas"], "Valor": [searches, offers]})
    chart = alt.Chart(t).mark_bar().encode(
        x=alt.X("Valor:Q"), y=alt.Y("Métrica:N", sort="-x"), tooltip=["Métrica", "Valor"]
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)


@register_tab("Comportamento Cias")
def tab7_comportamento_cias(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t7")
    st.subheader("Comportamento Cias (share por Trecho)")
    base = df.groupby(["TRECHO","AGENCIA_NORM"]).size().rename("Qtde").reset_index()
    if base.empty:
        st.info("Sem dados.")
        return
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
    chart = alt.Chart(agg).mark_bar().encode(
        x=alt.X("Δ Mediano:Q", title="Δ Mediano"),
        y=alt.Y("AGENCIA_NORM:N", sort="-x", title="Agência"),
        tooltip=["AGENCIA_NORM", "Δ Mediano"]
    ).properties(height=320)
    st.altair_chart(chart, use_container_width=True)


@register_tab("Melhor Preço Diário")
def tab9_melhor_preco_diario(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t9")
    st.subheader("Melhor Preço Diário (DATAHORA_BUSCA)")
    t = df.groupby(df["DATAHORA_BUSCA"].dt.date, as_index=False)["PRECO"].min().rename(
        columns={"DATAHORA_BUSCA":"Data","PRECO":"Melhor Preço"}
    )
    if t.empty:
        st.info("Sem dados.")
        return
    t["Data"] = pd.to_datetime(t["Data"], dayfirst=True)
    chart = alt.Chart(t).mark_line(point=True).encode(
        x=alt.X("Data:T"), y=alt.Y("Melhor Preço:Q"),
        tooltip=["Data:T", "Melhor Preço:Q"]
    ).properties(height=320)
    st.altair_chart(chart, use_container_width=True)


@register_tab("Exportar")
def tab10_exportar(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t10")
    st.subheader("Exportar dados filtrados")
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Baixar CSV (filtro aplicado)", data=csv_bytes, file_name="OFERTAS_filtrado.csv", mime="text/csv")


# ─────────────────────────────────── MAIN ────────────────────────────────────
def main():
    df_raw = load_base(DATA_PATH)

    # Banner opcional (primeira imagem da pasta do app, se houver)
    for ext in ("*.png","*.jpg","*.jpeg","*.gif","*.webp"):
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
