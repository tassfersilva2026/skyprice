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

# ============================== UTILIDADES GERAIS ==============================

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
    if np.isnan(v):
        v = 1
    return min([1,5,11,17,30], key=lambda k: abs(v-k))

@st.cache_data(show_spinner=True)
def load_base(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Arquivo obrigatório não encontrado: {path.as_posix()}"); st.stop()
    df = pd.read_parquet(path)

    colmap = {
        0:"IDPESQUISA", 1:"CIA", 2:"HORA_BUSCA", 3:"HORA_PARTIDA", 4:"HORA_CHEGADA",
        5:"TIPO_VOO", 6:"DATA_EMBARQUE", 7:"DATAHORA_BUSCA", 8:"AGENCIA_COMP",
        9:"PRECO", 10:"TRECHO", 11:"ADVP", 12:"RANKING"
    }
    if list(df.columns[:13]) != list(colmap.values()):
        rename = {df.columns[i]: colmap[i] for i in range(min(13, df.shape[1]))}
        df = df.rename(columns=rename)

    for c in ["HORA_BUSCA", "HORA_PARTIDA", "HORA_CHEGADA"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c].astype(str).str.strip(), errors="coerce").dt.strftime("%H:%M:%S")
    df["HORA_HH"] = pd.to_datetime(df["HORA_BUSCA"], errors="coerce").dt.hour

    for c in ["DATA_EMBARQUE", "DATAHORA_BUSCA"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)

    if "PRECO" in df.columns:
        df["PRECO"] = (df["PRECO"].astype(str)
                        .str.replace(r"[^\d,.-]", "", regex=True)
                        .str.replace(",", ".", regex=False))
        df["PRECO"] = pd.to_numeric(df["PRECO"], errors="coerce")

    if "RANKING" in df.columns:
        df["RANKING"] = pd.to_numeric(df["RANKING"], errors="coerce").astype("Int64")

    df["AGENCIA_NORM"] = df["AGENCIA_COMP"].apply(std_agencia)
    df["ADVP_CANON"]   = df["ADVP"].apply(advp_nearest)
    return df

def winners_by_position(df: pd.DataFrame) -> pd.DataFrame:
    base = pd.DataFrame({"IDPESQUISA": df["IDPESQUISA"].unique()})
    for r in (1,2,3):
        s = (df[df["RANKING"]==r]
             .sort_values(["IDPESQUISA"])
             .drop_duplicates(subset=["IDPESQUISA"]))
        base = base.merge(
            s[["IDPESQUISA","AGENCIA_NORM"]].rename(columns={"AGENCIA_NORM":f"R{r}"}),
            on="IDPESQUISA", how="left"
        )
    for r in (1,2,3):
        base[f"R{r}"] = base[f"R{r}"].fillna("SEM OFERTAS")
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

# ---- Estilos dos cards (Painel) ----
CARD_CSS = """
<style>
.cards-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
@media (max-width: 1100px) { .cards-grid { grid-template-columns: repeat(2, minmax(0,1fr)); } }
@media (max-width: 700px)  { .cards-grid { grid-template-columns: 1fr; } }
.card { border:1px solid #e9e9ee; border-radius:14px; padding:10px 12px; background:#fff; box-shadow:0 1px 2px rgba(0,0,0,.04); }
.card .title { font-weight:650; font-size:15px; margin-bottom:8px; }
.goldcard{background:#FFF9E5;border-color:#D4AF37;} .silvercard{background:#F7F7FA;border-color:#C0C0C0;} .bronzecard{background:#FFF1E8;border-color:#CD7F32;}
.row{display:flex;gap:8px;} .item{flex:1;display:flex;align-items:center;justify-content:space-between;gap:8px;padding:8px 10px;border-radius:10px;border:1px solid #e3e3e8;background:#fafbfc;}
.pos{font-weight:700;font-size:12px;opacity:.85;} .pct{font-size:16px;font-weight:650;}
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

# ---- Gráficos utilitários ----
def make_bar(df: pd.DataFrame, x_col: str, y_col: str, sort_y_desc: bool = True):
    d = df[[y_col, x_col]].copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = d[y_col].astype(str)
    d = d.dropna(subset=[x_col])
    if sort_y_desc: d = d.sort_values(x_col, ascending=False)
    if d.empty: return alt.Chart(pd.DataFrame({x_col: [], y_col: []})).mark_bar()
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
    if color: d[color] = d[color].astype(str)
    d = d.dropna(subset=[x_col, y_col])
    if d.empty: return alt.Chart(pd.DataFrame({x_col: [], y_col: []})).mark_line()
    enc = dict(x=x_enc, y=alt.Y(f"{y_col}:Q", title=y_col), tooltip=[f"{x_col}", f"{y_col}:Q"])
    if color: enc["color"] = alt.Color(f"{color}:N", title=color)
    return alt.Chart(d).mark_line(point=True).encode(**enc).properties(height=300)

# ---- Estado dos filtros ----
def _init_filter_state(df_raw: pd.DataFrame):
    if "flt" in st.session_state: return
    dmin = pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce").min()
    dmax = pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce").max()
    st.session_state["flt"] = {
        "dt_ini": (dmin.date() if pd.notna(dmin) else date(2000,1,1)),
        "dt_fim": (dmax.date() if pd.notna(dmax) else date.today()),
        "advp": [], "trechos": [], "hh": [],
    }

def render_filters(df_raw: pd.DataFrame, key_prefix: str = "flt"):
    _init_filter_state(df_raw)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns([1.1,1.1,1,2,1])

    dmin_abs = pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce").min()
    dmax_abs = pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce").max()
    dmin_abs = dmin_abs.date() if pd.notna(dmin_abs) else date(2000,1,1)
    dmax_abs = dmax_abs.date() if pd.notna(dmax_abs) else date.today()

    with c1:
        dt_ini = st.date_input "Data inicial", key=f"{key_prefix}_dtini",
                               value=st.session_state["flt"]["dt_ini"],
                               min_value=dmin_abs, max_value=dmax_abs, format="DD/MM/YYYY")
    with c2:
        dt_fim = st.date_input "Data final", key=f"{key_prefix}_dtfim",
                               value=st.session_state["flt"]["dt_fim"],
                               min_value=dmin_abs, max_value=dmax_abs, format="DD/MM/YYYY")
    with c3:
        advp_all = sorted(set(pd.to_numeric(df_raw["ADVP_CANON"], errors="coerce").dropna().astype(int).tolist()))
        advp_sel = st.multiselect ADVP", options=advp_all,
                                  default=st.session_state["flt"]["advp"], key=f"{key_prefix}_advp")
    with c4:
        trechos_all = sorted([t for t in df_raw["TRECHO"].dropna().unique().tolist() if str(t).strip()!=""])
        tr_sel = st.multiselect "Trechos", trechos_all,
                                default=st.session_state["flt"]["trechos"], key=f"{key_prefix}_trechos")
    with c5:
        hh_sel = st.multiselect "Hora da busca", list(range(24)),
                                default=st.session_state["flt"]["hh"], key=f"{key_prefix}_hh")

    st.session_state["flt"] = {"dt_ini": dt_ini, "dt_fim": dt_fim,
                               "advp": advp_sel or [], "trechos": tr_sel or [], "hh": hh_sel or []}

    mask = pd.Series(True, index=df_raw.index)
    mask &= (pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce") >= pd.Timestamp(dt_ini))
    mask &= (pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce") <= pd.Timestamp(dt_fim))
    if advp_sel: mask &= df_raw["ADVP_CANON"].isin(advp_sel)
    if tr_sel:   mask &= df_raw["TRECHO"].isin(tr_sel)
    if hh_sel:   mask &= df_raw["HORA_HH"].isin(hh_sel)

    df = df_raw[mask].copy()
    st.caption(f"Linhas após filtros: {fmt_int(len(df))} • Última atualização: {last_update_from_cols(df)}")
    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
    return df

# ============================ REGISTRO DE ABAS ================================
TAB_REGISTRY: list[tuple[str, callable]] = []

def register_tab(label: str):
    """Decorator para registrar uma aba por nome. Troque só a função da aba e pronto."""
    def _wrap(fn):
        TAB_REGISTRY.append((label, fn))
        return fn
    return _wrap

# =============================== ABAS (INÍCIO) ===============================

# ──────────────────────────── ABA: Painel (START) ────────────────────────────
@register_tab("Painel")
def tab1_painel(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t1")
    st.subheader("Painel")

    total_pesq = df["IDPESQUISA"].nunique() or 1
    cov = {r: df.loc[df["RANKING"].eq(r), "IDPESQUISA"].nunique() for r in (1,2,3)}
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
        "R1":{"MAXMILHAS":"GRUPO 123","123MILHAS":"GRUPO 123"},
        "R2":{"MAXMILHAS":"GRUPO 123","123MILHAS":"GRUPO 123"},
        "R3":{"MAXMILHAS":"GRUPO 123","123MILHAS":"GRUPO 123"},
    })

    agencias_all = sorted(set(df_raw["AGENCIA_NORM"].dropna().astype(str)))
    targets_base = list(agencias_all)
    if "GRUPO 123" not in targets_base: targets_base.insert(0, "GRUPO 123")
    if "SEM OFERTAS" not in targets_base: targets_base.append("SEM OFERTAS")

    def pcts_for_target(tgt: str):
        base = Wg if tgt == "GRUPO 123" else W
        p1 = float((base["R1"] == tgt).mean())*100
        p2 = float((base["R2"] == tgt).mean())*100
        p3 = float((base["R3"] == tgt).mean())*100
        return p1, p2, p3

    targets_sorted = sorted(targets_base, key=lambda t: pcts_for_target(t)[0], reverse=True)

    cards = []
    for idx, tgt in enumerate(targets_sorted):
        p1, p2, p3 = pcts_for_target(tgt)
        rank_cls = "goldcard" if idx == 0 else "silvercard" if idx == 1 else "bronzecard" if idx == 2 else ""
        cards.append(card_html(tgt, p1, p2, p3, rank_cls))
    st.markdown(f"<div class='cards-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)
# ───────────────────────────── ABA: Painel (END) ─────────────────────────────

# ──────────────────────── ABA: Top 3 Agências (START) ────────────────────────
@register_tab("Top 3 Agências")
def tab2_top3_agencias(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t2")
    st.subheader("Top 3 Agências (por menor preço no trecho)")
    if df.empty:
        st.info("Sem dados para os filtros."); 
        return

    import numpy as _np
    import pandas as _pd

    # ======= Paleta fixa por coluna =======
    BLUE   = "#cfe3ff"  # Preços Top 1/2/3
    ORANGE = "#fdd0a2"  # 123milhas
    GREEN  = "#c7e9c0"  # Maxmilhas
    YELLOW = "#fee391"  # FlipMilhas
    PINK   = "#f1b6da"  # Capo Viagens/Capoviagens

    # ======= util: tons discretos (5 níveis) a partir da cor base =======
    def _hex_to_rgb(h): return tuple(int(h[i:i+2], 16) for i in (1,3,5))
    def _rgb_to_hex(t): return f"#{t[0]:02x}{t[1]:02x}{t[2]:02x}"
    def _blend(c_from, c_to, t):
        f, to = _hex_to_rgb(c_from), _hex_to_rgb(c_to)
        return _rgb_to_hex(tuple(int(round(f[i] + (to[i]-f[i])*t)) for i in range(3)))
    def make_scale(base_hex, steps=5):
        # do branco -> base, 5 níveis (mais escuro = valor maior)
        return [_blend("#ffffff", base_hex, k/(steps-1)) for k in range(steps)]

    SCALE_BLUE   = make_scale(BLUE)
    SCALE_ORANGE = make_scale(ORANGE)
    SCALE_GREEN  = make_scale(GREEN)
    SCALE_YELLOW = make_scale(YELLOW)
    SCALE_PINK   = make_scale(PINK)

    def style_heatmap_discrete(styler: _pd.io.formats.style.Styler, col: str, scale_colors: list[str]):
        """Aplica 5 faixas por quantis (ou rank) à coluna informada."""
        s = _pd.to_numeric(styler.data[col], errors="coerce")
        if s.notna().sum() == 0:
            return styler
        # tenta quantis; se tiver poucos valores distintos, usa rank
        try:
            bins = _pd.qcut(s.rank(method="average"), q=5, labels=False, duplicates="drop")
        except Exception:
            bins = _pd.cut(s.rank(method="average"), bins=5, labels=False)
        bins = bins.fillna(-1).astype(int)

        def _fmt(val, idx):
            if _pd.isna(val) or bins.iloc[idx] == -1:
                return "background-color: #ffffff; color:#111111"
            color = scale_colors[int(bins.iloc[idx])]
            return f"background-color: {color}; color:#111111"

        styler = styler.apply(lambda col_vals: [_fmt(v, i) for i, v in enumerate(col_vals)], subset=[col])
        return styler

    # ======= Constantes de nomes =======
    A_MAX, A_123, A_FLIP, A_CAPO = "MAXMILHAS", "123MILHAS", "FLIPMILHAS", "CAPOVIAGENS"

    # ======= Tabela 1: Top3 por trecho (menor preço) =======
    by_ag = (
        df.groupby(["TRECHO", "AGENCIA_NORM"], as_index=False)
          .agg(PRECO_MIN=("PRECO", "min"))
    )

    def _row_top3(g: _pd.DataFrame) -> _pd.Series:
        g = g.sort_values("PRECO_MIN", ascending=True).reset_index(drop=True)
        def name(i):  return g.loc[i, "AGENCIA_NORM"] if i < len(g) else "-"
        def price(i): return g.loc[i, "PRECO_MIN"]    if i < len(g) else _np.nan
        def price_of(ag):
            m = g[g["AGENCIA_NORM"] == ag]
            return (m["PRECO_MIN"].min() if not m.empty else _np.nan)

        return _pd.Series({
            "Trecho": g["TRECHO"].iloc[0] if len(g) else "-",
            "Agencia Top 1": name(0), "Preço Top 1": price(0),
            "Agencia Top 2": name(1), "Preço Top 2": price(1),
            "Agencia Top 3": name(2), "Preço Top 3": price(2),
            "123milhas":     price_of(A_123),
            "Maxmilhas":     price_of(A_MAX),
            "FlipMilhas":    price_of(A_FLIP),
            "Capo Viagens":  price_of(A_CAPO),
        })

    t1 = by_ag.groupby("TRECHO").apply(_row_top3).reset_index(drop=True)

    # preços inteiros (0 casas) preservando NA
    preco_cols = ["Preço Top 1","Preço Top 2","Preço Top 3","123milhas","Maxmilhas","FlipMilhas","Capo Viagens"]
    for c in preco_cols:
        t1[c] = _pd.to_numeric(t1[c], errors="coerce").round(0).astype("Int64")

    t1.index = _np.arange(1, len(t1) + 1); t1.index.name = "#"

    st.markdown("**Ranking Top 3 (Agências)**")
    fmt_map_t1 = {c: "{:,.0f}" for c in preco_cols}
    sty1 = t1.style.format(fmt_map_t1, na_rep="-", decimal=",", thousands=".") \
                   .set_table_styles([{"selector":"tbody td, th","props":[("border","1px solid #EEE")]}]) \
                   .set_properties(**{"background-color":"#ffffff","color":"#111111"})

    # heatmap discreto por coluna com sua cor
    for c in ["Preço Top 1","Preço Top 2","Preço Top 3"]:
        sty1 = style_heatmap_discrete(sty1, c, SCALE_BLUE)
    sty1 = style_heatmap_discrete(sty1, "123milhas",    SCALE_ORANGE)
    sty1 = style_heatmap_discrete(sty1, "Maxmilhas",    SCALE_GREEN)
    sty1 = style_heatmap_discrete(sty1, "FlipMilhas",   SCALE_YELLOW)
    # aceita as duas grafias:
    if "Capo Viagens" in t1.columns:
        sty1 = style_heatmap_discrete(sty1, "Capo Viagens", SCALE_PINK)
    if "Capoviagens" in t1.columns:
        sty1 = style_heatmap_discrete(sty1, "Capoviagens", SCALE_PINK)

    st.dataframe(sty1, use_container_width=True)

    # ======= Tabela 2: % Diferença (base: Top1) =======
    def pct_diff(base, other):
        if _pd.isna(base) or base == 0 or _pd.isna(other): 
            return _np.nan
        return (other - base) / base * 100

    rows2 = []
    for _, r in t1.reset_index().iterrows():
        base = r["Preço Top 1"]
        rows2.append({
            "#": r["#"], "Trecho": r["Trecho"],
            "Agencia Top 1": r["Agencia Top 1"], "Preço Top 1": r["Preço Top 1"],
            "Agencia Top 2": r["Agencia Top 2"], "% Dif Top2 vs Top1": pct_diff(base, r["Preço Top 2"]),
            "Agencia Top 3": r["Agencia Top 3"], "% Dif Top3 vs Top1": pct_diff(base, r["Preço Top 3"]),
            "123milhas": pct_diff(base, r["123milhas"]),
            "Maxmilhas": pct_diff(base, r["Maxmilhas"]),
            "FlipMilhas": pct_diff(base, r["FlipMilhas"]),
            "Capo Viagens": pct_diff(base, r["Capo Viagens"]),
        })
    t2 = _pd.DataFrame(rows2).set_index("#"); t2.index.name = "#"

    st.markdown("**% Diferença entre Agências (base: Top 1)**")
    pct_cols   = ["% Dif Top2 vs Top1","% Dif Top3 vs Top1","123milhas","Maxmilhas","FlipMilhas","Capo Viagens"]
    fmt_map_t2 = {"Preço Top 1": "{:,.0f}"} | {c: "{:.2f}%" for c in pct_cols}
    sty2 = t2.style.format(fmt_map_t2, na_rep="-", decimal=",", thousands=".") \
                   .set_table_styles([{"selector":"tbody td, th","props":[("border","1px solid #EEE")]}]) \
                   .set_properties(**{"background-color":"#ffffff","color":"#111111"})

    # cores dos % seguem a mesma lógica das colunas
    sty2 = style_heatmap_discrete(sty2, "Preço Top 1",      SCALE_BLUE)
    sty2 = style_heatmap_discrete(sty2, "% Dif Top2 vs Top1", SCALE_BLUE)
    sty2 = style_heatmap_discrete(sty2, "% Dif Top3 vs Top1", SCALE_BLUE)
    sty2 = style_heatmap_discrete(sty2, "123milhas",        SCALE_ORANGE)
    sty2 = style_heatmap_discrete(sty2, "Maxmilhas",        SCALE_GREEN)
    sty2 = style_heatmap_discrete(sty2, "FlipMilhas",       SCALE_YELLOW)
    sty2 = style_heatmap_discrete(sty2, "Capo Viagens",     SCALE_PINK)

    st.dataframe(sty2, use_container_width=True)

    # ======= Tabelas 3/4: Comparativo com Cia (se existir 'CIA') =======
    if "CIA" not in df.columns:
        st.info("Coluna de Cia Aérea ('CIA') não encontrada. As tabelas 3/4 dependem dela.")
        return

    by_air = (df.groupby(["TRECHO","CIA"], as_index=False)
                .agg(PRECO_AIR_MIN=("PRECO", "min")))
    idx = by_air.groupby("TRECHO")["PRECO_AIR_MIN"].idxmin()
    min_air = by_air.loc[idx, ["TRECHO", "CIA", "PRECO_AIR_MIN"]] \
                    .rename(columns={"CIA": "Cia Menor Preço", "PRECO_AIR_MIN": "Preço Menor Valor"})

    def best_ag(sub: _pd.DataFrame, ag_norm: str):
        dfa = sub[sub["AGENCIA_NORM"] == ag_norm]
        if dfa.empty: 
            return (_np.nan, "-")
        i = dfa["PRECO"].idxmin()
        return (float(dfa.loc[i, "PRECO"]), str(dfa.loc[i, "CIA"]))

    rows3 = []
    for trecho, sub in df.groupby("TRECHO"):
        p123, c123   = best_ag(sub, A_123)
        pmax, cmax   = best_ag(sub, A_MAX)
        pflip, cflip = best_ag(sub, A_FLIP)
        pcapo, ccapo = best_ag(sub, A_CAPO)

        base = min_air[min_air["TRECHO"] == trecho]
        if base.empty:
            cia_min, pmin = "-", _np.nan
        else:
            cia_min = base["Cia Menor Preço"].iloc[0]
            pmin    = float(base["Preço Menor Valor"].iloc[0])

        rows3.append({
            "Trecho": trecho,
            "Cia Menor Preço": cia_min,
            "Preço Menor Valor": pmin,
            "Preço Maxmilhas": pmax, "Cia Maxmilhas": cmax,
            "Preço 123milhas": p123, "Cia 123milhas": c123,
            "Preço FlipMilhas": pflip, "Cia FlipMilhas": cflip,
            "Preço Capo Viagens": pcapo, "Cia Capo Viagens": ccapo,
        })

    t3 = _pd.DataFrame(rows3)
    for c in [c for c in t3.columns if c.startswith("Preço ")]:
        t3[c] = _pd.to_numeric(t3[c], errors="coerce").round(0).astype("Int64")
    t3.index = _np.arange(1, len(t3) + 1); t3.index.name = "#"

    st.markdown("**Comparativo Menor Preço Cia × Agências de Milhas**")
    preco_cols_t3 = [c for c in t3.columns if c.startswith("Preço ")]
    fmt_map_t3 = {c: "{:,.0f}" for c in preco_cols_t3}
    sty3 = t3.style.format(fmt_map_t3, na_rep="-", decimal=",", thousands=".") \
                   .set_table_styles([{"selector":"tbody td, th","props":[("border","1px solid #EEE")]}]) \
                   .set_properties(**{"background-color":"#ffffff","color":"#111111"})

    # heatmap cores
    for c in [c for c in t3.columns if c.startswith("Preço ")]:
        if "Maxmilhas"   in c: sty3 = style_heatmap_discrete(sty3, c, SCALE_GREEN)
        elif "123milhas" in c: sty3 = style_heatmap_discrete(sty3, c, SCALE_ORANGE)
        elif "FlipMilhas" in c: sty3 = style_heatmap_discrete(sty3, c, SCALE_YELLOW)
        elif "Capo Viagens" in c or "Capoviagens" in c: sty3 = style_heatmap_discrete(sty3, c, SCALE_PINK)
        else: sty3 = style_heatmap_discrete(sty3, c, SCALE_BLUE)

    st.dataframe(sty3, use_container_width=True)

    # % comparativo vs Cia
    def pct_vs_base(base, x):
        if _pd.isna(base) or base == 0 or _pd.isna(x): 
            return _np.nan
        return (x - base) / base * 100

    t4 = t3[["Trecho", "Cia Menor Preço", "Preço Menor Valor"]].copy()
    for label, col_cia, col_preco in [
        ("Maxmilhas", "Cia Maxmilhas", "Preço Maxmilhas"),
        ("123milhas", "Cia 123milhas", "Preço 123milhas"),
        ("FlipMilhas", "Cia FlipMilhas", "Preço FlipMilhas"),
        ("Capo Viagens", "Cia Capo Viagens", "Preço Capo Viagens"),
    ]:
        t4[f"Cia {label}"]   = t3[col_cia]
        t4[f"% Dif {label}"] = [pct_vs_base(b, x) for b, x in zip(t3["Preço Menor Valor"], t3[col_preco])]

    t4.index = _np.arange(1, len(t4) + 1); t4.index.name = "#"

    st.markdown("**%Comparativo Menor Preço Cia × Agências de Milhas**")
    pct_cols_t4 = [c for c in t4.columns if c.startswith("% Dif ")]
    fmt_map_t4  = {"Preço Menor Valor": "{:,.0f}"} | {c: "{:.2f}%" for c in pct_cols_t4}
    sty4 = t4.style.format(fmt_map_t4, na_rep="-", decimal=",", thousands=".") \
                   .set_table_styles([{"selector":"tbody td, th","props":[("border","1px solid #EEE")]}]) \
                   .set_properties(**{"background-color":"#ffffff","color":"#111111"})
    sty4 = style_heatmap_discrete(sty4, "Preço Menor Valor", SCALE_BLUE)
    for c in pct_cols_t4:
        if "Maxmilhas"   in c: sty4 = style_heatmap_discrete(sty4, c, SCALE_GREEN)
        elif "123milhas" in c: sty4 = style_heatmap_discrete(sty4, c, SCALE_ORANGE)
        elif "FlipMilhas" in c: sty4 = style_heatmap_discrete(sty4, c, SCALE_YELLOW)
        elif "Capo Viagens" in c: sty4 = style_heatmap_discrete(sty4, c, SCALE_PINK)
        else: sty4 = style_heatmap_discrete(sty4, c, SCALE_BLUE)

    st.dataframe(sty4, use_container_width=True)
# ───────────────────────── ABA: Top 3 Agências (END) ─────────────────────────





# ──────────────────── ABA: Top 3 Preços Mais Baratos (START) ─────────────────
@register_tab("Top 3 Preços Mais Baratos")
def tab3_top3_precos(df_raw: pd.DataFrame):
    """
    Pódio por Trecho → ADVP (mesma pesquisa):
      • Para CADA (Trecho, ADVP), usa a ÚLTIMA IDPESQUISA daquele par.
      • Cards Top1/2/3 só dessa pesquisa.
      • Ícone "?" discreto mostra/copía o ID da pesquisa.
    """
    import re
    import numpy as _np
    import pandas as _pd

    # ====== Filtros do app ======
    df = render_filters(df_raw, key_prefix="t3")
    st.subheader("Pódio por Trecho → ADVP (última pesquisa de cada par)")
    if df.empty:
        st.info("Sem dados para os filtros."); 
        return

    # ====== UI ======
    c1, c2, _ = st.columns([0.28, 0.18, 0.54])
    agencia_foco = c1.selectbox("Agência alvo", ["Todos", "123MILHAS", "MAXMILHAS"], index=0)
    posicao_foco = c2.selectbox("Ranking", ["Todas", 1, 2, 3], index=0)

    # ====== Helpers ======
    def fmt_moeda_br(x):
        try:
            xv = float(x)
            if not _np.isfinite(xv): return "R$ -"
            return "R$ " + f"{xv:,.0f}".replace(",", ".")
        except: return "R$ -"

    def fmt_pct_plus(x):
        try:
            v = float(x)
            if not _np.isfinite(v): return "—"
            return f"+{round(v):.0f}%"
        except: return "—"

    def _canon(s: str) -> str: return re.sub(r"[^A-Z0-9]+", "", str(s).upper())
    def _brand_tag(s: str) -> str | None:
        cs = _canon(s)
        if cs.startswith("123MILHAS") or cs == "123": return "123MILHAS"
        if cs.startswith("MAXMILHAS") or cs == "MAX":  return "MAXMILHAS"
        return None

    def parse_hora_text(val) -> str | None:
        s = str(val).strip()
        if s == "" or s.lower() in {"nan", "none", "null"}: return None
        try:
            ts = _pd.to_datetime(s, dayfirst=True, errors="raise")
            return ts.strftime("%H:%M:%S")
        except: pass
        digs = "".join(ch for ch in s if ch.isdigit())
        h = m = sec = None
        try:
            if len(digs) >= 6:   h, m, sec = int(digs[-6:-4]), int(digs[-4:-2]), int(digs[-2:])
            elif len(digs) == 4: h, m, sec = int(digs[:2]),    int(digs[2:4]),    0
            elif len(digs) in (1,2): h, m, sec = int(digs), 0, 0
        except: pass
        if h is None:
            parts = [p for p in s.replace(".", ":").split(":") if p]
            try:
                if len(parts) == 2: h, m, sec = int(parts[0]), int(parts[1]), 0
                elif len(parts) == 3: h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
            except: return None
        if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= sec <= 59): return None
        return f"{h:02d}:{m:02d}:{sec:02d}"

    # ====== CSS do ícone do ID (discreto) ======
    BADGE_POP_CSS = """
    <style>
    .idp-wrap{position:relative; display:inline-flex; align-items:center; margin-left:6px;}
    .idp-badge{display:inline-flex; align-items:center; justify-content:center; width:16px; height:16px;
      border:1px solid #cbd5e1; border-radius:50%; font-size:11px; font-weight:900; color:#64748b; background:#fff;
      user-select:none; cursor:default; line-height:1;}
    .idp-pop{position:absolute; top:18px; right:0; background:#fff; color:#0f172a; border:1px solid #e5e7eb;
      border-radius:8px; padding:6px 8px; font-size:12px; font-weight:700; box-shadow:0 6px 16px rgba(0,0,0,.08);
      display:none; z-index:9999; white-space:nowrap;}
    .idp-wrap:hover .idp-pop{ display:block; }
    .idp-idbox{border:1px solid #e5e7eb; background:#f8fafc; border-radius:6px; padding:2px 6px; font-weight:800; font-size:12px;
      min-width:60px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; user-select:text; cursor:text;}
    </style>
    """
    st.markdown(BADGE_POP_CSS, unsafe_allow_html=True)

    def _popover_html(id_val) -> str:
        if id_val is None or (isinstance(id_val, float) and _np.isnan(id_val)): return ""
        sid = str(id_val).replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")
        return ("<span class='idp-wrap'>"
                "<span class='idp-badge' title='Passe o mouse para ver o ID'>?</span>"
                f"<span class='idp-pop'>ID:&nbsp;<input class='idp-idbox' type='text' value='{sid}' readonly></span>"
                "</span>")

    # ====== Normalização base ======
    dfp = df.copy()
    advp_col = "ADVP_CANON" if "ADVP_CANON" in dfp.columns else "ADVP"

    dfp = dfp.assign(
        __TRECHO__ = dfp["TRECHO"].astype(str),
        __ADVP__   = dfp[advp_col].astype(str),
        __AG__     = dfp["AGENCIA_NORM"].astype(str),
        __PRECO__  = _pd.to_numeric(dfp["PRECO"], errors="coerce"),
        __DTKEY__  = _pd.to_datetime(dfp.get("DATAHORA_BUSCA"), errors="coerce"),
    )
    hora_raw = dfp.get("HORA_BUSCA", _pd.Series([None]*len(dfp), index=dfp.index))
    dfp["__HORA__"] = hora_raw.apply(parse_hora_text)
    dfp["_H_"]      = _pd.to_datetime(dfp["__HORA__"], format="%H:%M:%S", errors="coerce")

    dfp = dfp[dfp["__PRECO__"].notna()]
    if dfp.empty:
        st.info("Sem preços válidos no recorte."); 
        return

    # ====== Última ID por (Trecho, ADVP) ======
    tmp_sorted = dfp.sort_values(["__TRECHO__", "__ADVP__", "__DTKEY__", "_H_"], kind="mergesort")
    last_rows = (
        tmp_sorted.groupby(["__TRECHO__", "__ADVP__"], as_index=False)
                  .tail(1)[["__TRECHO__", "__ADVP__", "IDPESQUISA", "__DTKEY__", "__HORA__"]]
    )
    last_rows_idx = last_rows.set_index(["__TRECHO__", "__ADVP__"])

    # ====== Estilos dos cards ======
    GRID   = "display:grid;grid-auto-flow:column;grid-auto-columns:260px;gap:10px;overflow-x:auto;padding:6px 2px 10px 2px;scrollbar-width:thin;"
    BOX    = "border:1px solid #e5e7eb;border-radius:12px;background:#fff;"
    HEAD   = "padding:8px 10px;border-bottom:1px solid #f1f5f9;font-weight:700;color:#111827;"
    STACK  = "display:grid;gap:8px;padding:8px;"
    CARD   = "position:relative;border:1px solid #e5e7eb;border-radius:10px;padding:8px;background:#fff;min-height:78px;"
    DTWRAP = "position:absolute;right:8px;top:6px;display:flex;align-items:center;gap:6px;"
    DTTXT  = "font-size:10px;color:#94a3b8;font-weight:800;"
    RANK   = "font-weight:900;font-size:11px;color:#6b7280;letter-spacing:.3px;text-transform:uppercase;"
    AG     = "font-weight:800;font-size:15px;color:#111827;margin-top:2px;"
    PR     = "font-weight:900;font-size:18px;color:#111827;margin-top:2px;"
    SUB    = "font-weight:700;font-size:12px;color:#374151;"
    NOBOX  = "padding:22px 12px;color:#6b7280;font-weight:800;text-align:center;border:1px dashed #e5e7eb;border-radius:10px;background:#fafafa;"
    FOOT   = "border-top:1px dashed #e5e7eb;margin:6px 8px 8px 8px;padding-top:6px;display:flex;gap:6px;flex-wrap:wrap;"
    CHIP   = "background:#f3f4f6;border:1px solid #e5e7eb;border-radius:10px;padding:2px 6px;font-weight:700;font-size:11px;color:#111827;white-space:nowrap;line-height:1.1;"

    # ====== Cálculo do pódio por subgrupo ======
    def build_rank(sub: _pd.DataFrame) -> _pd.DataFrame:
        rk = (sub.groupby("__AG__", as_index=False)["__PRECO__"].min()
                .sort_values("__PRECO__").reset_index(drop=True))
        if not rk.empty:
            rk["_CAN"]   = rk["__AG__"].apply(_canon)
            rk["_BRAND"] = rk["__AG__"].apply(_brand_tag)
        return rk

    # ====== Render ======
    for trecho in sorted(dfp["__TRECHO__"].dropna().unique(), key=str):
        df_t = dfp[dfp["__TRECHO__"] == trecho]

        def _advp_key(v):
            m = re.search(r"\d+", str(v)); return (0, int(m.group())) if m else (1, str(v))
        advps = sorted(df_t["__ADVP__"].dropna().unique(), key=_advp_key)

        boxes = []
        for advp in advps:
            key = (trecho, str(advp))
            if key not in last_rows_idx.index:
                continue
            last_id  = last_rows_idx.loc[key, "IDPESQUISA"]
            dt_last  = last_rows_idx.loc[key, "__DTKEY__"]
            hh_last  = last_rows_idx.loc[key, "__HORA__"]
            dt_label = (dt_last.strftime("%d/%m/%Y") if _pd.notna(dt_last) else "") + (f" {hh_last}" if isinstance(hh_last, str) and hh_last else "")

            rows = df_t[(df_t["__ADVP__"] == str(advp)) & (df_t["IDPESQUISA"] == last_id)].copy()
            rank = build_rank(rows)

            if not rank.empty and agencia_foco != "Todos":
                rk_map = {row["__AG__"]: i+1 for i, row in rank.head(3).iterrows()}
                found = any(_canon(ag) == _canon(agencia_foco) and (posicao_foco == "Todas" or posicao_foco == i)
                            for ag, i in rk_map.items())
                if not found: 
                    continue

            box = [f"<div style='{BOX}'>", f"<div style='{HEAD}'>ADVP: <b>{advp}</b></div>"]

            if rank.empty:
                box.append(f"<div style='{NOBOX}'>Sem ofertas</div>")
                box.append("</div>"); boxes.append("".join(box)); continue

            box.append(f"<div style='{STACK}'>")
            for i in range(min(3, len(rank))):
                r = rank.iloc[i]
                preco_i = float(r["__PRECO__"]); ag_i = r["__AG__"]

                if i == 0 and len(rank) >= 2:
                    p2 = float(rank.iloc[1]["__PRECO__"])
                    subtxt = f"−{int(round((p2 - preco_i) / p2 * 100.0))}% vs 2º" if (_np.isfinite(p2) and p2 != 0) else "—"
                else:
                    p1 = float(rank.iloc[0]["__PRECO__"])
                    subtxt = fmt_pct_plus((preco_i - p1) / p1 * 100.0) + " vs 1º" if (_np.isfinite(p1) and p1 != 0) else "—"

                stripe = "#D4AF37" if i == 0 else ("#9CA3AF" if i == 1 else "#CD7F32")
                stripe_div = f"<div style='position:absolute;left:0;top:0;bottom:0;width:6px;border-radius:10px 0 0 10px;background:{stripe};'></div>"

                box.append(
                    "<div style='" + CARD + "'>"
                    + stripe_div +
                    f"<div style='{DTWRAP}'><span style='{DTTXT}'>{dt_label}</span>{_popover_html(last_id)}</div>"
                    f"<div style='{RANK}'>{i+1}º</div>"
                    f"<div style='{AG}'>{ag_i}</div>"
                    f"<div style='{PR}'>{fmt_moeda_br(preco_i)}</div>"
                    f"<div style='{SUB}'>{subtxt}</div>"
                    "</div>"
                )
            box.append("</div>")  # stack

            # chips 123/MAX se fora do Top3 (nessa pesquisa)
            podium_tags = {_canon(r['__AG__']) for _, r in rank.head(3).iterrows()}
            p1 = float(rank.iloc[0]["__PRECO__"])
            extras = []
            for target in ["123MILHAS", "MAXMILHAS"]:
                if _canon(target) in podium_tags: 
                    continue
                m = rank[rank["_BRAND"] == _brand_tag(target)]
                if m.empty:
                    extras.append(f"<span style='{CHIP}'>{target}: Não apareceu</span>")
                else:
                    pos = int(m.index[0]) + 1
                    px  = float(m.iloc[0]["__PRECO__"])
                    delta = ((px - p1) / p1 * 100.0) if (_np.isfinite(px) and _np.isfinite(p1) and p1 != 0) else None
                    extras.append(f"<span style='{CHIP}'>{pos}º {target}: {fmt_moeda_br(px)}{(' ' + fmt_pct_plus(delta)) if delta is not None else ''}</span>")

            if extras:
                box.append(f"<div style='{FOOT}'>" + "".join(extras) + "</div>")

            box.append("</div>")  # box
            boxes.append("".join(box))

        if boxes:
            st.markdown(
                f"<div style='margin:14px 0 10px 0;padding:10px 12px;border-left:4px solid #0B5FFF;background:#ECF3FF;border-radius:8px;font-weight:800;color:#0A2A6B;'>Trecho: <b>{trecho}</b></div>",
                unsafe_allow_html=True
            )
            st.markdown("<div style='" + GRID + "'>" + "".join(boxes) + "</div>", unsafe_allow_html=True)
# ───────────────────── ABA: Top 3 Preços Mais Baratos (END) ──────────────────



# ───────────────────── ABA 4: Ranking por Agências (START) ───────────────────
@register_tab("Ranking por Agências")
def tab4_ranking_agencias(df_raw: pd.DataFrame):
    """
    Ranking por posição (1..15) com 3 visões:
      1) Quantidade de ofertas por ranking (com Total)
      2) % participação do ranking DENTRO da agência (linha)
      3) % participação da agência DENTRO do ranking (coluna)
    Observações:
      - Usa colunas do app: AGENCIA_NORM, RANKING, TRECHO (opcional), DATAHORA_BUSCA
      - Heatmap sem matplotlib (css via Styler.apply), com destaque para 123MILHAS e MAXMILHAS
    """
    import numpy as np
    import pandas as pd
    # ... resto da função ...


    # ------------- Filtros globais (do app) -------------
    df = render_filters(df_raw, key_prefix="t4")
    st.subheader("Ranking por Agências (1º ao 15º)")
    if df.empty:
        st.info("Sem dados para os filtros."); 
        return

    # ------------- Normalização mínima -------------
    if "AGENCIA_NORM" not in df.columns or "RANKING" not in df.columns:
        st.error("Colunas obrigatórias ausentes: AGENCIA_NORM e/ou RANKING."); 
        return

    # RANKING como inteiro 1..15 (ignora nulos e fora de faixa)
    rk = pd.to_numeric(df["RANKING"], errors="coerce").astype("Int64")
    df = df[rk.notna()].copy()
    df["RANKING"] = rk.astype(int)
    RANKS = list(range(1, 16))
    df = df[df["RANKING"].isin(RANKS)]
    if df.empty:
        st.info("Sem posições de ranking entre 1 e 15 no recorte."); 
        return

    # ============ Núcleo: pivot com contagem ============
    # (compatível com o script de referência)
    counts = (df.groupby(["AGENCIA_NORM", "RANKING"], as_index=False)
                .agg(OFERTAS=("AGENCIA_NORM", "size")))

    pv = (counts.pivot(index="AGENCIA_NORM", columns="RANKING", values="OFERTAS")
                 .reindex(columns=RANKS, fill_value=0)
                 .fillna(0).astype(int))
    pv.index.name = "Agência/Companhia"

    # total por agência + linha Total
    if 1 not in pv.columns:  # salvaguarda
        pv[1] = 0
    pv["Total"] = pv.sum(axis=1)
    pv = pv.sort_values(by=1, ascending=False)

    total_row = pv.sum(axis=0).to_frame().T
    total_row.index = ["Total"]
    total_row.index.name = "Agência/Companhia"
    pv2 = pd.concat([pv, total_row], axis=0)

    # ============ Tabelas de % ============
    mat = pv[RANKS].copy()

    # 1) % dentro da agência (linha)
    row_sum = mat.sum(axis=1).replace(0, np.nan)
    pct_linha = (mat.div(row_sum, axis=0) * 100).fillna(0)
    pct_linha = pct_linha.sort_values(by=1, ascending=False)

    # 2) % dentro do ranking (coluna)
    col_sum = mat.sum(axis=0).replace(0, np.nan)
    pct_coluna = (mat.div(col_sum, axis=1) * 100).fillna(0)
    pct_coluna = pct_coluna.sort_values(by=1, ascending=False)

    # ============ Estilização (sem matplotlib) ============
    HL_MAP = {"123MILHAS": "#FFD8A8", "MAXMILHAS": "#D3F9D8"}  # linhas especiais

    def _fmt_percent(v):
        return "-" if pd.isna(v) else f"{v:.2f}%".replace(".", ",")

    def _heat_col(col: pd.Series):
        # gradiente branco -> azul (sem mpl)
        vmin = pd.to_numeric(col, errors="coerce").min()
        vmax = pd.to_numeric(col, errors="coerce").max()
        rng = (vmax - vmin) if pd.notna(vmax) and pd.notna(vmin) and vmax != vmin else 1.0

        def to_css(x):
            try:
                x = float(x); 
                if not np.isfinite(x): 
                    return ""
                t = (x - vmin) / rng
                # interp linear entre #ffffff e #1E40AF
                r = int(255 + t * (30 - 255))
                g = int(255 + t * (64 - 255))
                b = int(255 + t * (175 - 255))
                return f"background-color: rgb({r},{g},{b}); color:#0A2A6B;"
            except Exception:
                return ""
        return col.apply(to_css)

    def _apply_row_highlight(df_show: pd.DataFrame, hl_map: dict):
        first_col = df_show.columns[0]
        idx_upper = df_show[first_col].astype(str).str.upper()
        styles = pd.DataFrame("", index=df_show.index, columns=df_show.columns)
        for k, color in hl_map.items():
            mask = idx_upper.eq(k)
            styles.loc[mask, :] = f"background-color:{color}; color:#0A2A6B; font-weight:bold;"
        return styles

    def _style_table(df_show: pd.DataFrame, percent_cols=None, highlight_total_row=False, highlight_total_col=None,
                     highlight_rows_map=None):
        df_disp = df_show.reset_index(drop=True)
        df_disp.index = np.arange(1, len(df_disp) + 1)

        sty = df_disp.style

        # formatação de percentuais
        if percent_cols:
            fmt_map = {c: _fmt_percent for c in percent_cols if c in df_disp.columns}
            sty = sty.format(fmt_map, na_rep="-")

        # heatmap por coluna (numéricas)
        for c in df_disp.columns:
            if pd.api.types.is_numeric_dtype(df_disp[c]) and (not percent_cols or c not in percent_cols):
                sty = sty.apply(_heat_col, subset=[c])

        # destacar coluna "Total"
        if highlight_total_col and highlight_total_col in df_disp.columns:
            def _hl_total(col):
                return ["background-color:#E6F0FF; font-weight:bold; color:#0A2A6B;" for _ in col]
            sty = sty.apply(_hl_total, subset=[highlight_total_col])

        # destacar linhas especiais (123 / MAX)
        if highlight_rows_map:
            sty = sty.apply(lambda _: _apply_row_highlight(df_disp, highlight_rows_map), axis=None)

        # destacar última linha (Total)
        if highlight_total_row and len(df_disp) > 0:
            def _hl_last_row(row):
                return ["background-color:#E6F0FF; color:#0A2A6B; font-weight:bold;"
                        if row.name == df_disp.index.max() else "" for _ in row]
            sty = sty.apply(_hl_last_row, axis=1)

        return sty

    def _show_table(df_in: pd.DataFrame, percent_cols=None, highlight_total_row=False,
                    highlight_total_col=None, highlight_rows_map=None, height=440):
        st.dataframe(
            _style_table(df_in, percent_cols=percent_cols, highlight_total_row=highlight_total_row,
                         highlight_total_col=highlight_total_col, highlight_rows_map=highlight_rows_map),
            use_container_width=True, height=height
        )

    # ========= Tabela 1 — Quantidades =========
    t_qtd = pv2.reset_index()
    if t_qtd.columns[0] != "Agência/Companhia":
        t_qtd = t_qtd.rename(columns={t_qtd.columns[0]: "Agência/Companhia"})

    st.subheader("Quantidade de Ofertas por Ranking (Ofertas)")
    _show_table(
        t_qtd[["Agência/Companhia"] + RANKS + ["Total"]],
        highlight_total_row=True,
        highlight_total_col="Total",
        highlight_rows_map=HL_MAP,
        height=480
    )

    # ========= Tabela 2 — % dentro da Agência (linha) =========
    t_pct_linha = pct_linha.reset_index()
    if t_pct_linha.columns[0] != "Agência/Companhia":
        t_pct_linha = t_pct_linha.rename(columns={t_pct_linha.columns[0]: "Agência/Companhia"})

    st.subheader("Participação do Ranking dentro da Agência")
    _show_table(
        t_pct_linha[["Agência/Companhia"] + RANKS],
        percent_cols=set(RANKS),
        highlight_rows_map=HL_MAP,
        height=440
    )

    # ========= Tabela 3 — % dentro do Ranking (coluna) =========
    t_pct_coluna = pct_coluna.reset_index()
    if t_pct_coluna.columns[0] != "Agência/Companhia":
        t_pct_coluna = t_pct_coluna.rename(columns={t_pct_coluna.columns[0]: "Agência/Companhia"})

    st.subheader("Participação da Agência dentro do Ranking")
    _show_table(
        t_pct_coluna[["Agência/Companhia"] + RANKS],
        percent_cols=set(RANKS),
        highlight_rows_map=HL_MAP,
        height=440
    )
# ───────────────────── ABA 4: Ranking por Agências (END) ─────────────────────

# ──────────────── ABA 4: Melhor Preço por Período do Dia (START) ───────────────
@register_tab("Melhor Preço por Período do Dia")
def tab4_melhor_preco_por_periodo(df_raw: pd.DataFrame):
    """
    Adaptação da página "04 Melhor Preço por Hora":
      • Usa df filtrado via render_filters (app).
      • Hora do VOO: tenta HORA_PARTIDA; se faltar, tenta HORA_CHEGADA; senão usa HORA_HH (hora da busca).
      • Séries por hora (0–23): Melhor Preço (concorrente vencedor), 123MILHAS, MAXMILHAS.
      • 4 seções: Madrugada / Manhã / Tarde / Noite (Y invertido: menor em cima).
      • Top3 por hora com rótulos; 123 e MAX viram “Grupo 123” se diferença ≤ R$0,01.
    """
    import re
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st

    # ================== 1) Base com filtros globais ==================
    df = render_filters(df_raw, key_prefix="t4_new")
    st.subheader("Ranking de Melhor Preço por Período do Dia")
    if df.empty:
        st.info("Sem resultados para os filtros selecionados."); 
        return

    # ================== 2) Colunas-alvo & normalizações ==================
    # Agência normalizada já existe no app como AGENCIA_NORM
    if "AGENCIA_NORM" not in df.columns:
        st.error("Coluna AGENCIA_NORM não encontrada."); 
        return

    # Preço: já está numérico no app, mas deixo robusto se vier texto
    def parse_price_cell(x) -> float | None:
        if pd.isna(x): return None
        try:
            # se já for número, vai direto
            if isinstance(x, (int, float, np.number)):
                v = float(x)
                return v if np.isfinite(v) and v > 0 else None
            s = str(x)
            m = re.search(
                r"(?:R\$\s*)?("                       # opcional "R$"
                r"\d{1,3}(?:\.\d{3})*,\d{2}"          # 1.234,56
                r"|\d+,\d{2}"                         # 1234,56
                r"|\d{1,3}(?:\.\d{3})+"               # 1.234
                r"|\d+"                               # 1234
                r")", s
            )
            if not m: return None
            num = m.group(1).replace(".", "").replace(",", ".")
            v = float(num)
            return v if np.isfinite(v) and v > 0 else None
        except:
            return None

    price_series = (
        df["PRECO"] if "PRECO" in df.columns else df.get("VALOR", pd.Series([np.nan]*len(df)))
    )
    df["_PRICE_NUM"] = pd.to_numeric(price_series, errors="coerce")
    # fallback de parse se veio tudo NaN (ex.: strings)
    if df["_PRICE_NUM"].isna().all():
        df["_PRICE_NUM"] = price_series.apply(parse_price_cell)

    df = df[df["_PRICE_NUM"].notna() & (df["_PRICE_NUM"] > 0)].copy()
    if df.empty:
        st.info("Sem preços válidos no recorte atual."); 
        return

    # Hora do voo → 0..23
    def to_hour_any(v) -> float | None:
        try:
            if pd.isna(v): return np.nan
            s = str(v).strip()
            if ":" in s:
                # pega HH dos formatos HH:MM[:SS]
                hh = s.split(":")[0]
                return float(int(hh))
            return float(int(float(s)))
        except:
            return np.nan

    # preferência: HORA_PARTIDA > HORA_CHEGADA > HORA_HH (hora da busca)
    if "HORA_PARTIDA" in df.columns:
        hour_src = df["HORA_PARTIDA"].apply(to_hour_any)
    elif "HORA_CHEGADA" in df.columns:
        hour_src = df["HORA_CHEGADA"].apply(to_hour_any)
    else:
        # HORA_HH já é int 0..23 calculado no app; converto p/ float
        hour_src = pd.to_numeric(df.get("HORA_HH", pd.Series([np.nan]*len(df))), errors="coerce").astype(float)

    df["HORA"] = hour_src
    df = df[df["HORA"].between(0, 23, inclusive="both")].copy()
    if df.empty:
        st.info("Não há horas do voo (0–23) válidas no recorte."); 
        return

    # ConcorrenteNome = Agência + cia (se existir)
    cia_col = "CIA" if "CIA" in df.columns else None

    # ================== 3) Séries por hora (Melhor / 123 / MAX) ==================
    A_123 = "123MILHAS"
    A_MAX = "MAXMILHAS"

    g_123 = (df[df["AGENCIA_NORM"].eq(A_123)]
             .groupby("HORA", as_index=False)["_PRICE_NUM"].min()
             .rename(columns={"_PRICE_NUM": "Preco_123"}))

    g_max = (df[df["AGENCIA_NORM"].eq(A_MAX)]
             .groupby("HORA", as_index=False)["_PRICE_NUM"].min()
             .rename(columns={"_PRICE_NUM": "Preco_MAX"}))

    idxmin_all = df.groupby("HORA")["_PRICE_NUM"].idxmin()
    cols = ["HORA", "_PRICE_NUM", "AGENCIA_NORM"]
    if cia_col: cols.append(cia_col)
    melhor = df.loc[idxmin_all, cols].copy()
    melhor.rename(columns={"_PRICE_NUM": "Preco_MELHOR"}, inplace=True)
    if cia_col:
        melhor["ConcorrenteNome"] = melhor["AGENCIA_NORM"].astype(str) + " • " + melhor[cia_col].astype(str)
    else:
        melhor["ConcorrenteNome"] = melhor["AGENCIA_NORM"].astype(str)

    base = pd.DataFrame({"HORA": list(range(24))})
    base = (base.merge(g_123, how="left", on="HORA")
                .merge(g_max,  how="left", on="HORA")
                .merge(melhor[["HORA", "Preco_MELHOR", "ConcorrenteNome"]], how="left", on="HORA"))

    long_main = base.melt(
        id_vars=["HORA","ConcorrenteNome"],
        value_vars=["Preco_MELHOR","Preco_123","Preco_MAX"],
        var_name="Serie", value_name="Preco"
    )
    legend_main = {"Preco_MELHOR":"Melhor Preço", "Preco_123":"123MILHAS", "Preco_MAX":"MAXMILHAS"}
    long_main["Agência"] = long_main["Serie"].map(legend_main)
    long_main.drop(columns=["Serie"], inplace=True)

    def periodo(h):
        h = int(h)
        if   0 <= h <= 5:  return "Madrugada"
        elif 6 <= h <= 11: return "Manhã"
        elif 12 <= h <= 17:return "Tarde"
        else:              return "Noite"

    long_main["Período"] = long_main["HORA"].apply(periodo)

    COLORS_MAIN = {
        "Melhor Preço":"#2962FF",  # azul
        "123MILHAS":   "#FF8A00",  # laranja
        "MAXMILHAS":   "#00C853",  # verde
    }

    secoes = [
        ("Madrugada", [0, 1, 2, 3, 4, 5]),
        ("Manhã",     [6, 7, 8, 9, 10, 11]),
        ("Tarde",     [12, 13, 14, 15, 16, 17]),
        ("Noite",     [18, 19, 20, 21, 22, 23]),
    ]

    def desenha_secao_main(titulo: str, horas_fixas: list[int]):
        dfp = long_main[(long_main["HORA"].isin(horas_fixas)) & (~long_main["Preco"].isna())].copy()
        if dfp.empty:
            st.info(f"Sem dados para {titulo}."); return

        fig = px.line(
            dfp.sort_values(["Agência","HORA"]),
            x="HORA", y="Preco",
            color="Agência", markers=True,
            color_discrete_map=COLORS_MAIN,
            category_orders={"Agência":["Melhor Preço","123MILHAS","MAXMILHAS"]},
            labels={"HORA":"Hora do Voo", "Preco":"Preço (R$)"},
            title=titulo
        )
        fig.update_traces(line=dict(width=3))
        # hovers
        for tr in fig.data:
            name = tr.name
            sub = dfp[dfp["Agência"]==name].sort_values("HORA")
            if name == "Melhor Preço":
                nomes = sub["ConcorrenteNome"].fillna("").tolist()
                tr.customdata = np.array(nomes, dtype=object).reshape(-1,1)
                tr.hovertemplate = "Hora %{x} • %{customdata[0]}<br>Preço R$ %{y:,.0f}<extra></extra>"
            else:
                tr.hovertemplate = "Hora %{x}<br>%{fullData.name}<br>Preço R$ %{y:,.0f}<extra></extra>"

        # padding + eixo Y invertido
        x0, x1 = min(horas_fixas), max(horas_fixas)
        fig.update_xaxes(
            tickmode="array",
            tickvals=horas_fixas, ticktext=[str(h) for h in horas_fixas],
            range=[x0 - 0.35, x1 + 0.35],
            showgrid=False, showline=False, zeroline=False, ticks="", ticklen=0
        )
        fig.update_yaxes(
            autorange="reversed",  # menor em cima
            tickprefix="R$ ", separatethousands=True,
            showgrid=True, gridcolor="rgba(0,0,0,0.06)",
            showline=False, zeroline=False, ticks="", ticklen=0
        )
        fig.update_layout(
            template="simple_white",
            legend_title_text="Agência",
            legend=dict(orientation="v", x=1.02, y=1.0, bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=20, r=28, t=45, b=10),
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    for titulo, horas in secoes:
        desenha_secao_main(titulo, horas)

    # ================== 4) Agências campeãs (TOP3) por hora ==================
    st.subheader("Agências Campeãs (TOP3)")
    st.caption("123MILHAS e MAXMILHAS viram “Grupo 123” quando a diferença entre eles for ≤ R$0,01 no mesmo horário.")

    df_min_ag_h = (
        df.groupby(["HORA","AGENCIA_NORM"], as_index=False)["_PRICE_NUM"]
          .min()
          .rename(columns={"_PRICE_NUM":"Preco"})
    )

    # Colapsa 123/MAX em "Grupo 123" quando diferença ≤ R$0,01 no MESMO horário
    piv = df_min_ag_h.pivot(index="HORA", columns="AGENCIA_NORM", values="Preco")
    s123 = piv.get(A_123)
    smax = piv.get(A_MAX)
    eq_hours = []
    if s123 is not None and smax is not None:
        mask_eq = s123.notna() & smax.notna() & (s123.sub(smax).abs() <= 0.01)
        eq_hours = piv.index[mask_eq].tolist()

    df_group = pd.DataFrame({
        "HORA": eq_hours,
        "AGENCIA_NORM": "Grupo 123",
        "Preco": s123.loc[eq_hours].values if s123 is not None and len(eq_hours)>0 else []
    })

    mask_remove = (df_min_ag_h["HORA"].isin(eq_hours)) & (df_min_ag_h["AGENCIA_NORM"].isin([A_123, A_MAX]))
    df_min_ag_h2 = pd.concat([df_min_ag_h.loc[~mask_remove], df_group], ignore_index=True)

    # Top 3 por hora
    top3 = (
        df_min_ag_h2.sort_values(["HORA","Preco"])
                    .groupby("HORA", as_index=False, group_keys=False)
                    .apply(lambda g: g.head(3))
                    .reset_index(drop=True)
    )
    top3["Rank"] = top3.groupby("HORA").cumcount() + 1
    top3["RankLabel"] = top3["Rank"].map({1:"Top 1", 2:"Top 2", 3:"Top 3"})

    def _apply_label_jitter(d: pd.DataFrame) -> pd.DataFrame:
        if d.empty:
            d = d.copy(); d["Preco_text"] = d.get("Preco", pd.Series(dtype=float)); return d
        d = d.copy()
        d["dup_idx"]  = d.groupby(["HORA","Preco"]).cumcount()
        d["dup_size"] = d.groupby(["HORA","Preco"])["Preco"].transform("size")
        y_min, y_max = float(d["Preco"].min()), float(d["Preco"].max())
        y_range = max(1.0, y_max - y_min)
        step = max(y_range * 0.003, 0.3)
        center = (d["dup_size"] - 1) / 2.0
        d["Preco_text"] = d["Preco"] + (d["dup_idx"] - center) * step
        return d

    def _offset_for_reversed(d: pd.DataFrame, horas_fixas: list[int]) -> pd.DataFrame:
        if d.empty:
            d = d.copy(); d["x_text"]=d.get("HORA", pd.Series(dtype=float)); d["y_text"]=d.get("Preco_text", pd.Series(dtype=float)); d["textpos"]="middle left"; return d
        d = d.copy()
        x_min, x_max = min(horas_fixas), max(horas_fixas)
        x_span = max(1.0, x_max - x_min)
        dx = max(x_span * 0.002, 0.02)
        y_min, y_max = float(d["Preco_text"].min()), float(d["Preco_text"].max())
        y_range = max(1.0, y_max - y_min)
        dy = max(y_range * 0.003, 0.3)
        d["x_text"] = d["HORA"] + dx
        d["y_text"] = d["Preco_text"] - dy   # invertido: “acima” é menor
        d["textpos"] = "middle left"
        is_first = d["HORA"] <= x_min
        is_last  = d["HORA"] >= x_max
        d.loc[is_first, "x_text"] = np.maximum(d.loc[is_first, "HORA"] + dx, x_min + 0.06)
        d.loc[is_last,  "x_text"] = d.loc[is_last,  "HORA"] - dx
        d.loc[is_last,  "textpos"] = "middle right"
        return d

    LABEL_COLORS = {
        "Grupo 123": "#0D47A1",  # azul forte
        "MAXMILHAS": "#00C853",  # verde
        "123MILHAS": "#FF8A00",  # laranja
    }
    RANK_COLORS = {"Top 1":"#FFD700", "Top 2":"#9E9E9E", "Top 3":"#CD7F32"}
    BOLD_FAMILY = "Arial Black, DejaVu Sans, sans-serif"

    def desenha_secao_top3(titulo: str, horas_fixas: list[int]):
        dfp = top3[top3["HORA"].isin(horas_fixas)].copy()
        if dfp.empty:
            st.info(f"Sem dados para {titulo}."); return

        dfp["EmpresaRotulo"] = dfp["AGENCIA_NORM"].astype(str)

        fig = px.line(
            dfp.sort_values(["Rank","HORA"]),
            x="HORA", y="Preco",
            color="RankLabel", markers=True, text=None,
            color_discrete_map=RANK_COLORS,
            category_orders={"RankLabel":["Top 1","Top 2","Top 3"]},
            labels={"HORA":"Hora do Voo", "Preco":"Preço (R$)"},
            title=titulo
        )
        fig.update_traces(line=dict(width=3))
        for tr in fig.data:
            sub = dfp[dfp["RankLabel"]==tr.name].sort_values("HORA")
            tr.customdata = np.array(sub["AGENCIA_NORM"], dtype=object).reshape(-1,1)
            tr.hovertemplate = "Hora %{x} • %{customdata[0]}<br>Preço R$ %{y:,.0f}<extra></extra>"

        # Rótulos colados (com Y invertido)
        dfp_lbl = _apply_label_jitter(dfp)
        dfp_lbl = _offset_for_reversed(dfp_lbl, horas_fixas)

        # Especiais em negrito/cor
        for key in ["Grupo 123", "123MILHAS", "MAXMILHAS"]:
            sub = dfp_lbl[dfp_lbl["EmpresaRotulo"]==key].sort_values(["Rank","HORA"])
            if sub.empty: continue
            fig.add_trace(go.Scatter(
                x=sub["x_text"], y=sub["y_text"], mode="text",
                text=sub["EmpresaRotulo"], textposition=sub["textpos"].tolist(),
                textfont=dict(size=14, color=LABEL_COLORS[key], family=BOLD_FAMILY),
                showlegend=False, hoverinfo="skip", cliponaxis=False
            ))
        # Demais em preto
        sub_n = dfp_lbl[~dfp_lbl["EmpresaRotulo"].isin(LABEL_COLORS.keys())].sort_values(["Rank","HORA"])
        if not sub_n.empty:
            fig.add_trace(go.Scatter(
                x=sub_n["x_text"], y=sub_n["y_text"], mode="text",
                text=sub_n["EmpresaRotulo"], textposition=sub_n["textpos"].tolist(),
                textfont=dict(size=14, color="black"),
                showlegend=False, hoverinfo="skip", cliponaxis=False
            ))

        # Padding + Y invertido
        x0, x1 = min(horas_fixas), max(horas_fixas)
        pad_x = 0.40
        y_min = float(dfp["Preco"].min()); y_max = float(dfp["Preco"].max())
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y_min - 1.0, y_max + 1.0],
            mode="markers", marker=dict(size=0, opacity=0),
            showlegend=False, hoverinfo="skip"
        ))
        fig.update_xaxes(
            tickmode="array",
            tickvals=horas_fixas, ticktext=[str(h) for h in horas_fixas],
            range=[x0 - pad_x, x1 + pad_x],
            showgrid=False, showline=False, zeroline=False, ticks="", ticklen=0
        )
        fig.update_yaxes(
            autorange="reversed",
            tickprefix="R$ ", separatethousands=True,
            showgrid=True, gridcolor="rgba(0,0,0,0.06)",
            showline=False, zeroline=False, ticks="", ticklen=0
        )
        fig.update_layout(
            template="simple_white",
            legend_title_text="Ranking",
            legend=dict(orientation="v", x=1.02, y=1.0, bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=20, r=28, t=45, b=10),
            height=440
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    for titulo, horas in secoes:
        desenha_secao_top3(titulo, horas)
# ──────────────── ABA 4: Melhor Preço por Período do Dia (END) ────────────────

# ─────────────────────── ABA: Buscas x Ofertas (START) ────────────────────────
@register_tab("Qtde de Buscas x Ofertas")
def tab6_buscas_vs_ofertas(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t6")
    st.subheader("Quantidade de Buscas x Ofertas")
    searches = df["IDPESQUISA"].nunique(); offers = len(df)
    c1,c2 = st.columns(2)
    c1.metric("Pesquisas únicas", fmt_int(searches))
    c2.metric("Ofertas (linhas)", fmt_int(offers))
    t = pd.DataFrame({"Métrica":["Pesquisas","Ofertas"], "Valor":[searches, offers]})
    st.altair_chart(make_bar(t, "Valor", "Métrica"), use_container_width=True)
# ─────────────────────────── ABA: Buscas x Ofertas (END) ──────────────────────


# ────────────────────────── ABA: Comportamento Cias (START) ───────────────────
@register_tab("Comportamento Cias")
def tab7_comportamento_cias(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t7")
    st.subheader("Comportamento Cias (share por Trecho)")
    base = df.groupby(["TRECHO","AGENCIA_NORM"]).size().rename("Qtde").reset_index()
    if base.empty:
        st.info("Sem dados."); return
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
# ──────────────────────────── ABA: Comportamento Cias (END) ───────────────────


# ──────────────────────────── ABA: Competitividade (START) ────────────────────
@register_tab("Competitividade")
def tab8_competitividade(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t8")
    st.subheader("Competitividade (Δ mediano vs melhor preço por pesquisa)")
    best = df.groupby("IDPESQUISA")["PRECO"].min().rename("BEST").reset_index()
    t = df.merge(best, on="IDPESQUISA", how="left")
    t["DELTA"] = t["PRECO"] - t["BEST"]
    agg = t.groupby("AGENCIA_NORM", as_index=False)["DELTA"].median().rename(columns={"DELTA":"Δ Mediano"})
    st.altair_chart(make_bar(agg, "Δ Mediano", "AGENCIA_NORM"), use_container_width=True)
# ───────────────────────────── ABA: Competitividade (END) ─────────────────────


# ─────────────────────────── ABA: Melhor Preço Diário (START) ─────────────────
@register_tab("Melhor Preço Diário")
def tab9_melhor_preco_diario(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t9")
    st.subheader("Melhor Preço Diário (col. H - Data da busca)")
    t = df.groupby(df["DATAHORA_BUSCA"].dt.date, as_index=False)["PRECO"].min().rename(
        columns={"DATAHORA_BUSCA":"Data","PRECO":"Melhor Preço"}
    )
    if t.empty:
        st.info("Sem dados."); return
    t["Data"] = pd.to_datetime(t["Data"], dayfirst=True)
    st.altair_chart(make_line(t, "Data", "Melhor Preço"), use_container_width=True)
# ────────────────────────────── ABA: Melhor Preço Diário (END) ────────────────


# ─────────────────────────────── ABA: Exportar (START) ────────────────────────
@register_tab("Exportar")
def tab10_exportar(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t10")
    st.subheader("Exportar dados filtrados")
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Baixar CSV (filtro aplicado)",
                       data=csv_bytes, file_name="OFERTAS_filtrado.csv", mime="text/csv")
# ───────────────────────────────── ABA: Exportar (END) ────────────────────────

# ================================ ABAS (FIM) ==================================


# =================================== MAIN =====================================
def main():
    df_raw = load_base(DATA_PATH)

    # banner no topo (1ª imagem da raiz)
    for ext in ("*.png","*.jpg","*.jpeg","*.gif","*.webp"):
        imgs = list(APP_DIR.glob(ext))
        if imgs:
            st.image(imgs[0].as_posix(), use_container_width=True)
            break

    # monta as tabs dinamicamente a partir do registro
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
