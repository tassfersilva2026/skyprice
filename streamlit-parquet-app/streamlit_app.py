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

# ============================= UTIL & FORMATAÇÃO ==============================

def fmt_num0_br(x):
    try:
        if pd.isna(x): return "-"
        return f"R$ {float(x):,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "-"

def fmt_num2_br(x):
    try:
        if pd.isna(x): return "-"
        return f"R$ {float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "-"

def fmt_pct2(x):
    try:
        if pd.isna(x): return "-"
        return f"{float(x):.2%}".replace(".", ",")
    except Exception:
        return "-"

def fmt_diff_pct(x):
    try:
        if pd.isna(x): return "-"
        s = f"{float(x):+.0%}".replace(".", ",")
        return s
    except Exception:
        return "-"

def fmt_int(x):
    try:
        if pd.isna(x): return "-"
        return f"{int(x)}"
    except Exception:
        return "-"

def center(text: str) -> str:
    return f"<div style='text-align:center'>{text}</div>"

def _pick_scale(col_name: str) -> list[str]:
    return ["#f1f5f9", "#e2e8f0", "#cbd5e1", "#94a3b8", "#64748b"]

def style_heatmap_discrete(sty: pd.io.formats.style.Styler, col: str, palette: list[str]):
    try:
        sty = sty.background_gradient(cmap=alt.themes.get()(), subset=[col])  # fallback
    except Exception:
        # degrade para uma “pseudo” escala discreta
        def _bg(v):
            if pd.isna(v): return ""
            idx = int(min(len(palette)-1, max(0, round(v))))
            return f"background-color:{palette[idx]}"
        sty = sty.applymap(_bg, subset=[col])
    return sty

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
        pass
    return sty

# ============================== PADRONIZAÇÕES ================================

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
    if ag in ("", "NAN", "NONE", "NULL", "SKYSCANNER"): return ""
    return ag

def std_cia(raw: str | None) -> str | None:
    s = (raw or "").strip().upper()
    if s.startswith("AZUL"):  return "AZUL"
    if s.startswith("GOL"):   return "GOL"
    if s.startswith("LATAM"): return "LATAM"
    return s if s else None

def advp_nearest(raw) -> int | None:
    try:
        x = int(float(str(raw).replace(",", ".")))
        choices = [1,3,7,14,21,30,60,90,120,150,180,365]
        return min(choices, key=lambda c: abs(c - x))
    except Exception:
        return None

# =========================== CARREGAMENTO & STATE ============================

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

    # Normaliza horas como texto HH:MM:SS via _norm_hhmmss (não cola data)
    for c in ["HORA_BUSCA","HORA_PARTIDA","HORA_CHEGADA"]:
        if c in df.columns:
            df[c] = df[c].astype(str).map(_norm_hhmmss)

    # Datas conhecidas
    for c in ["DATA_EMBARQUE","DATAHORA_BUSCA"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)

    # --- Combina DATA (coluna D em DATAHORA_BUSCA) + HORA (coluna C em HORA_BUSCA) ---
    dt = pd.to_datetime(df.get("DATAHORA_BUSCA"), errors="coerce", dayfirst=True)
    dt = dt.dt.normalize()
    h = df.get("HORA_BUSCA", pd.Series([""]*len(df))).fillna("").astype(str)
    h = h.where(h.str.len() > 0, "00:00:00")
    df["DATAHORA_CHAVE"] = pd.to_datetime(dt.dt.strftime("%Y-%m-%d") + " " + h, errors="coerce")
    df["HORA_HH"] = pd.to_numeric(h.str.slice(0, 2), errors="coerce")

    # Preço
    if "PRECO" in df.columns:
        df["PRECO"] = (
            df["PRECO"].astype(str)
            .str.replace(r"[^\d,.-]", "", regex=True)
            .str.replace(",", ".", regex=False)
        )
        df["PRECO"] = pd.to_numeric(df["PRECO"], errors="coerce")

    # Ranking
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

# ------------------------------ Session state -------------------------------

def _init_filter_state(df: pd.DataFrame):
    if "flt" in st.session_state: return
    dmin = pd.to_datetime(df["DATAHORA_CHAVE"], errors="coerce").min()
    dmax = pd.to_datetime(df["DATAHORA_CHAVE"], errors="coerce").max()
    st.session_state["flt"] = {
        "dt_ini": (dmax.date() if pd.notna(dmax) else date.today()),
        "dt_fim": (dmax.date() if pd.notna(dmax) else date.today()),
        "advp":   [],
        "trechos": [],
        "hh":     list(range(24)),
        "cia":    []
    }

def last_update_from_cols(df: pd.DataFrame) -> str:
    if df.empty: return "—"
    mx = pd.to_datetime(df.get("DATAHORA_CHAVE"), errors="coerce").max()
    if pd.isna(mx): return "—"
    return mx.strftime("%d/%m/%Y - %H:%M:%S")

# ---- CSS global: largura container
st.markdown("""
<style>
.block-container { padding-top: 0.6rem; }
.dataframe { font-size: 0.92rem; }
</style>
""", unsafe_allow_html=True)

# ================================ FILTROS =====================================

def render_filters(df_raw: pd.DataFrame, key_prefix: str = "flt"):
    _init_filter_state(df_raw)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns([1.1, 1.1, 1, 2, 1, 1.4])

    dmin_abs = pd.to_datetime(df_raw["DATAHORA_CHAVE"], errors="coerce").min()
    dmax_abs = pd.to_datetime(df_raw["DATAHORA_CHAVE"], errors="coerce").max()
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

    # aplica filtros
    mask = pd.Series(True, index=df_raw.index)
    dt_col = pd.to_datetime(df_raw["DATAHORA_CHAVE"], errors="coerce")
    mask &= (dt_col >= pd.Timestamp(dt_ini)) & (dt_col <= pd.Timestamp(dt_fim))
    if advp_sel:
        mask &= df_raw["ADVP_CANON"].astype("Int64").isin(advp_sel)
    if tr_sel:
        mask &= df_raw["TRECHO"].isin(tr_sel)
    if hh_sel and "HORA_HH" in df_raw.columns:
        mask &= df_raw["HORA_HH"].isin(hh_sel)
    if cia_sel:
        mask &= df_raw["CIA_NORM"].isin(cia_sel)

    return df_raw[mask].copy()

# ============================== ABA 1 — PAINEL ================================

def tab1_painel(df_raw: pd.DataFrame):
    st.subheader("Painel Geral")
    df = render_filters(df_raw, key_prefix="t1")

    col1, col2, col3, col4 = st.columns(4)
    total_reg = len(df)
    total_pesq = df["IDPESQUISA"].nunique() if "IDPESQUISA" in df.columns else 0
    dt_last = last_update_from_cols(df)
    ags = df["AGENCIA_NORM"].dropna().unique().tolist()

    col1.metric("Registros", f"{total_reg:,}".replace(",", "."))
    col2.metric("Pesquisas", f"{total_pesq:,}".replace(",", "."))
    col3.metric("Última atualização", dt_last)
    col4.metric("Agências (distintas)", f"{len(ags)}")

    # Pequena tabela por cia × ranking (quantidades)
    if df.empty:
        st.info("Sem dados no recorte atual."); return

    RANKS = [1,2,3]
    base = df.copy()
    base["CIA_NORM"] = base["CIA_NORM"].fillna("N/A")
    pv = (base[base["RANKING"].isin(RANKS)]
          .groupby(["CIA_NORM","RANKING"], as_index=True)
          .size().unstack(fill_value=0))
    for r in RANKS:
        if r not in pv.columns:
            pv[r] = 0
    pv["Total"] = pv.sum(axis=1)
    pv = pv.sort_values(by=1, ascending=False)

    total_row = pv.sum(axis=0).to_frame().T
    total_row.index = ["Total"]
    total_row.index.name = "Cia"
    pv2 = pd.concat([pv, total_row], axis=0)

    st.markdown("**Distribuição por Cia × Ranking (Top 3)**")
    st.dataframe(
        style_smart_colwise(
            pv2.reset_index().rename(columns={"CIA_NORM":"Cia"}),
            fmt_map={1:fmt_int,2:fmt_int,3:fmt_int,"Total":fmt_int},
            grad_cols=[1,2,3,"Total"]
        ),
        use_container_width=True, height=360
    )

# ================== ABA 2 — TOP 3 AGÊNCIAS (MENOR PREÇO) =====================

def tab2_top3_agencias(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t2")
    st.subheader("Top 3 Agências (por menor preço no trecho)")
    if df.empty:
        st.info("Sem resultados para os filtros selecionados.")
        return

    # 1) Garantir mesma pesquisa (pega a última por Trecho)
    df2 = df.copy()
    df2["DT"] = pd.to_datetime(df2["DATAHORA_CHAVE"], errors="coerce")
    g = (df2.dropna(subset=["TRECHO","IDPESQUISA","DT"])
              .groupby(["TRECHO","IDPESQUISA"], as_index=False)["DT"].max())
    last_idx = g.groupby("TRECHO")["DT"].idxmax()
    last_ids = {r["TRECHO"]: r["IDPESQUISA"] for _, r in g.loc[last_idx].iterrows()}
    df2["__ID_TARGET__"] = df2["TRECHO"].map(last_ids)
    df_last = df2[df2["IDPESQUISA"] == df2["__ID_TARGET__"]].copy()

    # 2) Data/hora de referência por trecho
    def _compose_dt_hora(sub: pd.DataFrame) -> str:
        d = pd.to_datetime(sub["DATAHORA_CHAVE"], errors="coerce").max()
        return "-" if pd.isna(d) else d.strftime("%d/%m/%Y %H:%M:%S")

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
        def price(i): return g.loc[i, "PRECO_MIN"] if i < len(g) else np.nan
        r1, r2, r3 = name(0), name(1), name(2)
        p1, p2, p3 = price(0), price(1), price(2)
        pct_vs2 = (p1/p2 - 1.0) if (pd.notna(p1) and pd.notna(p2) and p2>0) else np.nan
        return pd.Series({
            "Trecho": trecho,
            "Data/Hora Busca": dt_by_trecho.get(trecho, "-"),
            "1º Agência": r1, "1º Preço": p1,
            "2º Agência": r2, "2º Preço": p2,
            "3º Agência": r3, "3º Preço": p3,
            "% 1º vs 2º": pct_vs2
        })

    t = (by_ag.groupby("TRECHO_STD", as_index=False)
               .apply(_row_top3).reset_index(drop=True))

    st.dataframe(
        style_smart_colwise(
            t, fmt_map={"1º Preço":fmt_num0_br,"2º Preço":fmt_num0_br,"3º Preço":fmt_num0_br,"% 1º vs 2º":fmt_pct2},
            grad_cols=[]
        ),
        use_container_width=True, height=420
    )

# ================== ABA 3 — TOP 3 PREÇOS MAIS BARATOS ========================

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

    # --- Normalizações de apoio ---
    dfp = df.copy()
    dfp["AGENCIA_NORM"] = dfp["AGENCIA_NORM"].fillna("")
    if agencia_foco != "Todos":
        dfp = dfp[dfp["AGENCIA_NORM"] == agencia_foco]
    if posicao_foco != "Todas":
        dfp = dfp[dfp["RANKING"] == int(posicao_foco)]
    dfp["__PRECO__"] = pd.to_numeric(dfp.get("PRECO"), errors="coerce")
    dfp["__DTKEY__"]  = pd.to_datetime(dfp.get("DATAHORA_CHAVE"), errors="coerce")

    # Heurística p/ achar possível coluna de ID se não houver "IDPESQUISA"
    def _find_id_col(df_: pd.DataFrame) -> str | None:
        cands = ["IDPESQUISA","ID_PESQUISA","ID BUSCA","IDBUSCA","ID","ID ARQUIVO","IDARQUIVO",
                 "ID_PDF","IDPDF","ARQUIVO_ID","NOME_ARQUIVO_STD","NOME_ARQUIVO","NOME DO ARQUIVO","ARQUIVO"]
        norm = { re.sub(r"[^A-Z0-9]+","", c.upper()): c for c in df_.columns }
        for nm in cands:
            key = re.sub(r"[^A-Z0-9]+","", nm.upper())
            if key in norm: return norm[key]
        return df_.columns[0] if len(df_.columns) else None

    GRID_STYLE    = "display:grid;grid-auto-flow:column;grid-auto-columns:minmax(300px,1fr);gap:10px;overflow-x:auto;padding:6px 2px 10px 2px;scrollbar-width:thin;"
    BOX_STYLE     = "border:1px solid #e5e7eb;border-radius:12px;background:#fff;"
    HEAD_STYLE    = "padding:8px 10px;border-bottom:1px solid #f1f5f9;font-weight:700;color:#111827;"
    STACK_STYLE   = "display:grid;gap:8px;padding:8px;"
    CARD_BASE     = "position:relative;border:1px solid #e5e7eb;border-radius:10px;padding:8px;background:#fff;min-height:78px;"
    DT_WRAP_STYLE = "position:absolute;right:8px;top:6px;display:flex;align-items:center;gap:6px;"
    DT_TXT_STYLE  = "font-size:10px;color:#94a3b8;font-weight:800;"
    RANK_STYLE    = "font-weight:900;font-size:11px;color:#6b7280;letter-spacing:.3px;text-transform:uppercase;"
    BADGE_123     = "font-size:10px;font-weight:900;color:#92400e;background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:2px 6px;"
    BADGE_MAX     = "font-size:10px;font-weight:900;color:#14532d;background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:2px 6px;"

    ID_COL = "IDPESQUISA" if "IDPESQUISA" in dfp.columns else _find_id_col(dfp)
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
        """Data (DD/MM) + Hora HH:MM:SS, vindo de __DTKEY__."""
        if sub_rows.empty: return "", None
        r = sub_rows.loc[sub_rows["__DTKEY__"].idxmax()]
        dt_txt = pd.to_datetime(r["__DTKEY__"], errors="coerce")
        lbl = dt_txt.strftime("%d/%m %H:%M:%S") if pd.notna(dt_txt) else ""
        return lbl, _normalize_id(r.get(ID_COL))

    # cards por Trecho
    TRECHO_COL = "TRECHO"
    ADVP_COL   = "ADVP"
    AGENCIA_COL= "AGENCIA_NORM"
    PRICE_COL  = "__PRECO__"

    grid = st.container()
    with grid:
        st.markdown(f"<div style='{GRID_STYLE}'>", unsafe_allow_html=True)

        for (trecho, advp), df_sub in dfp.groupby([TRECHO_COL, ADVP_COL]):
            # última pesquisa desse Trecho×ADVP
            dtx, id_last = dt_and_id_for(df_sub)

            # top 3 por preço
            sub = df_sub.sort_values(PRICE_COL, ascending=True).head(3).copy()
            # Computa presença de 123 e MAX + diferença vs 1º
            def _pos_e_delta(ag_nome: str) -> str:
                row = df_sub[df_sub[AGENCIA_COL] == ag_nome].sort_values(PRICE_COL, ascending=True).head(1)
                if row.empty: return f"{ag_nome.title() if ag_nome!='123MILHAS' else '123milhas'} Não Apareceu"
                pos = int(df_sub.sort_values(PRICE_COL).reset_index(drop=True).index[row.index[0]]) + 1
                preco_ag = float(row.iloc[0][PRICE_COL])
                preco_1  = float(sub.iloc[0][PRICE_COL]) if not sub.empty else np.nan
                if pd.notna(preco_1) and preco_1>0:
                    diff = (preco_ag/preco_1 - 1.0)
                    sinal = "+" if diff>=0 else "-"
                    pct   = f"{sinal}{abs(diff):.0%}".replace(".", ",")
                else:
                    pct = "-"
                ag_label = "123milhas" if ag_nome=="123MILHAS" else "Maxmilhas"
                return f"{ag_label}: {pos}º - {fmt_num0_br(preco_ag)} ({pct} vs 1º)"

            # Monta HTML dos cards
            cards = []
            for i in range(3):
                if i < len(sub):
                    ag  = sub.iloc[i][AGENCIA_COL]
                    prc = sub.iloc[i][PRICE_COL]
                    cards.append(
                        f"<div style='{CARD_BASE}'>"
                        f"  <div style='{DT_WRAP_STYLE}'><span style='{DT_TXT_STYLE}'>{dtx}</span></div>"
                        f"  <div style='{RANK_STYLE}'>TOP {i+1}</div>"
                        f"  <div style='font-size:14px;font-weight:800'>{ag}</div>"
                        f"  <div style='font-size:22px;font-weight:900;margin-top:2px'>{fmt_num0_br(prc)}</div>"
                        f"</div>"
                    )
                else:
                    cards.append(f"<div style='{CARD_BASE}'><div style='{RANK_STYLE}'>TOP {i+1}</div><div>-</div></div>")

            # badges 123/MAX
            badge_123 = _pos_e_delta("123MILHAS")
            badge_MAX = _pos_e_delta("MAXMILHAS")
            badge_html = (
                f"<div style='display:flex;gap:8px;align-items:center'>"
                f"  <span style='{BADGE_123}'>{badge_123}</span>"
                f"  <span style='{BADGE_MAX}'>{badge_MAX}</span>"
                f"</div>"
            )

            # Cabeçalho da caixa
            head = f"<div style='{HEAD_STYLE}'>{trecho} — ADVP {advp}d</div>"
            box  = (
                f"<div style='{BOX_STYLE}'>"
                f"{head}"
                f"<div style='{STACK_STYLE}'>"
                f"  <div style='{GRID_STYLE}'>{''.join(cards)}</div>"
                f"  {badge_html}"
                f"</div>"
                f"</div>"
            )
            st.markdown(box, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ================== ABA 4 — RANKING POR AGÊNCIAS =============================

def tab4_ranking_agencias(df_raw: pd.DataFrame):
    st.subheader("Ranking por Agências (quantidade de aparições no Top 3)")
    df = render_filters(df_raw, key_prefix="t4")
    if df.empty:
        st.info("Sem resultados para os filtros selecionados."); return

    base = df[df["RANKING"].isin([1,2,3])].copy()
    base["AGENCIA_NORM"] = base["AGENCIA_NORM"].fillna("")
    pv = (base.groupby(["AGENCIA_NORM","RANKING"], as_index=True)
               .size()
               .unstack(fill_value=0))
    for r in (1,2,3):
        if r not in pv.columns:
            pv[r] = 0
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

    st.markdown("**Aparições por Agência (Top 3)**")
    st.dataframe(
        style_smart_colwise(
            t_qtd,
            fmt_map={1:fmt_int,2:fmt_int,3:fmt_int,"Total":fmt_int},
            grad_cols=[1,2,3,"Total"]
        ),
        use_container_width=True, height=420
    )

# =========================== REGISTRO DE ABAS ================================

TAB_REGISTRY: List[Tuple[str, Callable[[pd.DataFrame], None]]] = [
    ("Painel", tab1_painel),
    ("Top 3 Agências", tab2_top3_agencias),
    ("Top 3 Preços", tab3_top3_precos),
    ("Ranking Agências", tab4_ranking_agencias),
]

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
