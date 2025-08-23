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

# ========= Normalização / Parsing =========
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

    # horas -> "HH:MM:SS" e HH
    for c in ["HORA_BUSCA", "HORA_PARTIDA", "HORA_CHEGADA"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c].astype(str).str.strip(), errors="coerce").dt.strftime("%H:%M:%S")
    df["HORA_HH"] = pd.to_datetime(df["HORA_BUSCA"], errors="coerce").dt.hour

    # datas
    for c in ["DATA_EMBARQUE", "DATAHORA_BUSCA"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)

    # preço
    if "PRECO" in df.columns:
        df["PRECO"] = (df["PRECO"].astype(str)
                        .str.replace(r"[^\d,.-]", "", regex=True)
                        .str.replace(",", ".", regex=False))
        df["PRECO"] = pd.to_numeric(df["PRECO"], errors="coerce")

    # ranking
    if "RANKING" in df.columns:
        df["RANKING"] = pd.to_numeric(df["RANKING"], errors="coerce").astype("Int64")

    # normalizações
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

# ========= Estilos (3 cards por linha; cor só nos TOP 3) =========
CARD_CSS = """
<style>
.cards-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}
@media (max-width: 1100px) { .cards-grid { grid-template-columns: repeat(2, minmax(0,1fr)); } }
@media (max-width: 700px)  { .cards-grid { grid-template-columns: 1fr; } }

.card {
  border:1px solid #e9e9ee; border-radius:14px; padding:10px 12px;
  background:#fff; box-shadow:0 1px 2px rgba(0,0,0,.04);
}
.card .title { font-weight:650; font-size:15px; margin-bottom:8px; }

.goldcard   { background:#FFF9E5; border-color:#D4AF37; } /* ouro */
.silvercard { background:#F7F7FA; border-color:#C0C0C0; } /* prata */
.bronzecard { background:#FFF1E8; border-color:#CD7F32; } /* bronze */

.row  { display:flex; gap:8px; }
.item { flex:1; display:flex; align-items:center; justify-content:space-between;
        gap:8px; padding:8px 10px; border-radius:10px; border:1px solid #e3e3e8; background:#fafbfc; }
.pos  { font-weight:700; font-size:12px; opacity:.85; }
.pct  { font-size:16px; font-weight:650; }
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

# ========= Gráficos utilitários (sem use_container_width aqui) =========
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

# ========= Estado global dos filtros =========
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
        dt_ini = st.date_input("Data inicial (col. H)", key=f"{key_prefix}_dtini",
                               value=st.session_state["flt"]["dt_ini"],
                               min_value=dmin_abs, max_value=dmax_abs, format="DD/MM/YYYY")
    with c2:
        dt_fim = st.date_input("Data final (col. H)", key=f"{key_prefix}_dtfim",
                               value=st.session_state["flt"]["dt_fim"],
                               min_value=dmin_abs, max_value=dmax_abs, format="DD/MM/YYYY")
    with c3:
        advp_all = sorted(set(pd.to_numeric(df_raw["ADVP_CANON"], errors="coerce").dropna().astype(int).tolist()))
        advp_sel = st.multiselect("ADVP (col. L)", options=advp_all,
                                  default=st.session_state["flt"]["advp"], key=f"{key_prefix}_advp")
    with c4:
        trechos_all = sorted([t for t in df_raw["TRECHO"].dropna().unique().tolist() if str(t).strip()!=""])
        tr_sel = st.multiselect("Trechos (col. K)", trechos_all,
                                default=st.session_state["flt"]["trechos"], key=f"{key_prefix}_trechos")
    with c5:
        hh_sel = st.multiselect("Hora da busca HH (col. C)", list(range(24)),
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

# ========= Abas =========
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

def tab2_top3_agencias(df_raw: pd.DataFrame):
    # === Filtros padrão da sua app ===
    df = render_filters(df_raw, key_prefix="t2")
    st.subheader("Top 3 Agências (por menor preço no trecho)")
    if df.empty:
        st.info("Sem dados para os filtros."); 
        return

    # === Helpers locais (independentes) ===
    import numpy as _np
    import pandas as _pd

    def _is_null_like(v) -> bool:
        if v is None: return True
        if isinstance(v, float) and _np.isnan(v): return True
        if isinstance(v, str) and v.strip().lower() in {"none", "nan", ""}: return True
        return False

    def style_smart_colwise(df_show: _pd.DataFrame, fmt_map: dict, grad_cols: list[str]):
        """Tabela com fundo branco, formatação PT-BR e gradiente por coluna (fallback para colormaps nativos)."""
        sty = df_show.style
        sty = sty.set_properties(**{"background-color": "#FFFFFF", "color": "#111111"})
        if fmt_map:
            sty = sty.format(fmt_map, na_rep="-", decimal=",", thousands=".")
        # gradiente coluna-a-coluna (só onde houver número)
        for c in grad_cols:
            if c in df_show.columns:
                s = _pd.to_numeric(df_show[c], errors="coerce")
                if s.notna().any():
                    # usa colormaps básicos (sem depender de matplotlib no deploy)
                    cmap_name = "Blues"
                    if "Maxmilhas" in c:   cmap_name = "Greens"
                    if "123milhas" in c:   cmap_name = "Oranges"
                    if "FlipMilhas" in c:  cmap_name = "YlOrBr"
                    if "Capo Viagens" in c: cmap_name = "PuRd"
                    sty = sty.background_gradient(cmap=cmap_name, subset=[c])
        # força branco para nulos/None
        sty = sty.applymap(lambda v: "background-color: #FFFFFF; color: #111111" if _is_null_like(v) else "")
        # borda sutil
        sty = sty.set_table_styles([{
            "selector": "tbody td, th",
            "props": [("border", "1px solid #EEE")]
        }])
        return sty

    def pct_diff(base, other):
        if _pd.isna(base) or base == 0 or _pd.isna(other): 
            return _np.nan
        return (other - base) / base * 100

    # === Padronização de nomes que você já usa (AGENCIA_NORM) ===
    # No seu dataset AGENCIA_NORM é algo como 'MAXMILHAS', '123MILHAS', 'FLIPMILHAS', 'CAPOVIAGENS'
    A_MAX   = "MAXMILHAS"
    A_123   = "123MILHAS"
    A_FLIP  = "FLIPMILHAS"
    A_CAPO  = "CAPOVIAGENS"

    # ============ Tabela 1 — Ranking Top 3 por Trecho (menor preço) ============
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

        # rótulos de colunas amigáveis (só display)
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
    for c in ["Preço Top 1","Preço Top 2","Preço Top 3","123milhas","Maxmilhas","FlipMilhas","Capo Viagens"]:
        t1[c] = _pd.to_numeric(t1[c], errors="coerce")

    t1.index = _np.arange(1, len(t1) + 1); t1.index.name = "#"

    st.markdown("**Ranking Top 3 (Agências)**")
    cols_preco_t1 = ["Preço Top 1","Preço Top 2","Preço Top 3","123milhas","Maxmilhas","FlipMilhas","Capo Viagens"]
    fmt_map_t1    = {c: "{:,.0f}" for c in cols_preco_t1}
    sty1 = style_smart_colwise(t1, fmt_map_t1, grad_cols=cols_preco_t1)
    st.dataframe(sty1, use_container_width=True)

    # ============ Tabela 2 — % Diferença (base: Top1) ============
    rows2 = []
    for _, r in t1.reset_index().iterrows():
        base = r["Preço Top 1"]
        rows2.append({
            "#": r["#"],
            "Trecho": r["Trecho"],
            "Agencia Top 1": r["Agencia Top 1"],
            "Preço Top 1": r["Preço Top 1"],
            "Agencia Top 2": r["Agencia Top 2"],
            "% Dif Top2 vs Top1": pct_diff(base, r["Preço Top 2"]),
            "Agencia Top 3": r["Agencia Top 3"],
            "% Dif Top3 vs Top1": pct_diff(base, r["Preço Top 3"]),
            "123milhas": pct_diff(base, r["123milhas"]),
            "Maxmilhas": pct_diff(base, r["Maxmilhas"]),
            "FlipMilhas": pct_diff(base, r["FlipMilhas"]),
            "Capo Viagens": pct_diff(base, r["Capo Viagens"]),
        })
    t2 = _pd.DataFrame(rows2).set_index("#"); t2.index.name = "#"

    st.markdown("**% Diferença entre Agências (base: Top 1)**")
    pct_cols   = ["% Dif Top2 vs Top1","% Dif Top3 vs Top1","123milhas","Maxmilhas","FlipMilhas","Capo Viagens"]
    fmt_map_t2 = {"Preço Top 1": "{:,.0f}"} | {c: "{:.2f}%" for c in pct_cols}
    grad_cols  = ["Preço Top 1"] + pct_cols
    sty2 = style_smart_colwise(t2, fmt_map_t2, grad_cols=grad_cols)
    st.dataframe(sty2, use_container_width=True)

    # ============ Tabela 3/4 — Comparativo com Cia Aérea (se existir col 'CIA') ============
    if "CIA" not in df.columns:
        st.info("Coluna de Cia Aérea ('CIA') não encontrada. As tabelas 3/4 dependem dela.")
        return

    by_air = (
        df.groupby(["TRECHO","CIA"], as_index=False)
          .agg(PRECO_AIR_MIN=("PRECO", "min"))
    )
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
    t3.index = _np.arange(1, len(t3) + 1); t3.index.name = "#"

    st.markdown("**Comparativo Menor Preço Cia × Agências de Milhas**")
    preco_cols_t3 = [c for c in t3.columns if c.startswith("Preço")] + ["Preço Menor Valor"]
    preco_cols_t3 = list(dict.fromkeys(preco_cols_t3))
    fmt_map_t3 = {c: "{:,.0f}" for c in preco_cols_t3}
    sty3 = style_smart_colwise(t3, fmt_map_t3, grad_cols=preco_cols_t3)
    st.dataframe(sty3, use_container_width=True)

    # Tabela 4 — % comparativo vs menor preço da Cia
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
    grad_cols_t4 = ["Preço Menor Valor"] + pct_cols_t4
    sty4 = style_smart_colwise(t4, fmt_map_t4, grad_cols=grad_cols_t4)
    st.dataframe(sty4, use_container_width=True)


def tab3_top3_precos(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t3")
    st.subheader("Top 3 Preços Mais Baratos (geral filtrado)")
    t = df.sort_values("PRECO").head(3)[["AGENCIA_NORM","TRECHO","PRECO","DATAHORA_BUSCA"]]
    if t.empty:
        st.info("Sem dados."); return
    cols = st.columns(len(t))
    for i, (_,row) in enumerate(t.iterrows()):
        with cols[i]:
            st.metric(f"{row['AGENCIA_NORM']} • {row['TRECHO']}",
                      value=f"R$ {row['PRECO']:,.2f}".replace(",", "X").replace(".", ",").replace("X","."),
                      delta=f"{row['DATAHORA_BUSCA']:%d/%m/%Y}")

def tab4_ranking_agencias(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t4")
    st.subheader("Ranking por Agências")
    W = winners_by_position(df)
    wins = W["R1"].value_counts().rename_axis("Agência/Cia").reset_index(name="Vitórias 1º")
    vol  = df["AGENCIA_NORM"].value_counts().rename_axis("Agência/Cia").reset_index(name="Ofertas")
    rt = vol.merge(wins, on="Agência/Cia", how="left").fillna(0)
    rt["Taxa Vitória (%)"] = (rt["Vitórias 1º"]/rt["Ofertas"]*100).round(2)
    c1,c2 = st.columns(2)
    with c1: st.altair_chart(make_bar(rt[["Agência/Cia","Vitórias 1º"]], "Vitórias 1º", "Agência/Cia"), use_container_width=True)
    with c2: st.altair_chart(make_bar(rt[["Agência/Cia","Ofertas"]], "Ofertas", "Agência/Cia"), use_container_width=True)

def tab5_preco_periodo(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t5")
    st.subheader("Preço por Período do Dia (HH da busca)")
    t = df.groupby("HORA_HH", as_index=False)["PRECO"].median().rename(columns={"PRECO":"Preço Mediano"})
    st.altair_chart(make_line(t, "HORA_HH", "Preço Mediano"), use_container_width=True)

def tab6_buscas_vs_ofertas(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t6")
    st.subheader("Quantidade de Buscas x Ofertas")
    searches = df["IDPESQUISA"].nunique(); offers = len(df)
    c1,c2 = st.columns(2)
    c1.metric("Pesquisas únicas", fmt_int(searches))
    c2.metric("Ofertas (linhas)", fmt_int(offers))
    t = pd.DataFrame({"Métrica":["Pesquisas","Ofertas"], "Valor":[searches, offers]})
    st.altair_chart(make_bar(t, "Valor", "Métrica"), use_container_width=True)

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

def tab8_competitividade(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t8")
    st.subheader("Competitividade (Δ mediano vs melhor preço por pesquisa)")
    best = df.groupby("IDPESQUISA")["PRECO"].min().rename("BEST").reset_index()
    t = df.merge(best, on="IDPESQUISA", how="left")
    t["DELTA"] = t["PRECO"] - t["BEST"]
    agg = t.groupby("AGENCIA_NORM", as_index=False)["DELTA"].median().rename(columns={"DELTA":"Δ Mediano"})
    st.altair_chart(make_bar(agg, "Δ Mediano", "AGENCIA_NORM"), use_container_width=True)

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

def tab10_exportar(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t10")
    st.subheader("Exportar dados filtrados")
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Baixar CSV (filtro aplicado)",
                       data=csv_bytes, file_name="OFERTAS_filtrado.csv", mime="text/csv")

# ========= Main =========
def main():
    df_raw = load_base(DATA_PATH)

    # banner no topo (1ª imagem da raiz)
    for ext in ("*.png","*.jpg","*.jpeg","*.gif","*.webp"):
        imgs = list(APP_DIR.glob(ext))
        if imgs:
            st.image(imgs[0].as_posix(), use_container_width=True)
            break

    labels = [
        "Painel","Top 3 Agências","Top 3 Preços Mais Baratos","Ranking por Agências",
        "Preço por Período do Dia","Qtde de Buscas x Ofertas","Comportamento Cias",
        "Competitividade","Melhor Preço Diário","Exportar"
    ]
    t = st.tabs(labels)

    with t[0]:
        try: tab1_painel(df_raw)
        except Exception as e: st.error("Erro na aba Painel"); st.exception(e)
    with t[1]:
        try: tab2_top3_agencias(df_raw)
        except Exception as e: st.error("Erro na aba Top 3 Agências"); st.exception(e)
    with t[2]:
        try: tab3_top3_precos(df_raw)
        except Exception as e: st.error("Erro na aba Top 3 Preços"); st.exception(e)
    with t[3]:
        try: tab4_ranking_agencias(df_raw)
        except Exception as e: st.error("Erro na aba Ranking por Agências"); st.exception(e)
    with t[4]:
        try: tab5_preco_periodo(df_raw)
        except Exception as e: st.error("Erro na aba Preço por Período do Dia"); st.exception(e)
    with t[5]:
        try: tab6_buscas_vs_ofertas(df_raw)
        except Exception as e: st.error("Erro na aba Qtde de Buscas x Ofertas"); st.exception(e)
    with t[6]:
        try: tab7_comportamento_cias(df_raw)
        except Exception as e: st.error("Erro na aba Comportamento Cias"); st.exception(e)
    with t[7]:
        try: tab8_competitividade(df_raw)
        except Exception as e: st.error("Erro na aba Competitividade"); st.exception(e)
    with t[8]:
        try: tab9_melhor_preco_diario(df_raw)
        except Exception as e: st.error("Erro na aba Melhor Preço Diário"); st.exception(e)
    with t[9]:
        try: tab10_exportar(df_raw)
        except Exception as e: st.error("Erro na aba Exportar"); st.exception(e)

if __name__ == "__main__":
    main()
