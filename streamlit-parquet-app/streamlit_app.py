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
    try: v = float(str(x).replace(",", "."))
    except: v = np.nan
    if np.isnan(v): v = 1
    return min([1,5,11,17,30], key=lambda k: abs(v-k))

@st.cache_data(show_spinner=True)
def load_base(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Arquivo obrigatório não encontrado: `{path.as_posix()}`"); st.stop()
    df = pd.read_parquet(path)

    # mapear colunas pela posição se necessário
    colmap = {
        0:"IDPESQUISA", 1:"CIA", 2:"HORA_BUSCA", 3:"HORA_PARTIDA", 4:"HORA_CHEGADA",
        5:"TIPO_VOO", 6:"DATA_EMBARQUE", 7:"DATAHORA_BUSCA", 8:"AGENCIA_COMP",
        9:"PRECO", 10:"TRECHO", 11:"ADVP", 12:"RANKING"
    }
    if list(df.columns[:13]) != list(colmap.values()):
        rename = {df.columns[i]: colmap[i] for i in range(min(13, df.shape[1]))}
        df = df.rename(columns=rename)

    # horas texto -> HH e "HH:MM:SS"
    for c in ["HORA_BUSCA", "HORA_PARTIDA", "HORA_CHEGADA"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c].astype(str).str.strip(), errors="coerce").dt.strftime("%H:%M:%S")
    df["HORA_HH"] = pd.to_datetime(df["HORA_BUSCA"], errors="coerce").dt.hour

    # datas dd/mm/aaaa (G/H)
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
    df["AGENCIA_NORM"]  = df["AGENCIA_COMP"].apply(std_agencia)
    df["AGENCIA_GRUPO"] = df["AGENCIA_NORM"].replace({"MAXMILHAS":"GRUPO 123", "123MILHAS":"GRUPO 123"})
    df["ADVP_CANON"]    = df["ADVP"].apply(advp_nearest)
    return df

def winners_by_position(df: pd.DataFrame) -> pd.DataFrame:
    base = pd.DataFrame({"IDPESQUISA": df["IDPESQUISA"].unique()})
    for r in (1,2,3):
        s = (df[df["RANKING"]==r]
             .sort_values(["IDPESQUISA"])
             .drop_duplicates(subset=["IDPESQUISA"]))
        base = base.merge(s[["IDPESQUISA","AGENCIA_NORM"]].rename(columns={"AGENCIA_NORM":f"R{r}"}),
                          on="IDPESQUISA", how="left")
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

# ========= Gráficos (com coerção de tipos) =========
def make_bar(df: pd.DataFrame, x_col: str, y_col: str, sort_y_desc: bool = True):
    d = df[[y_col, x_col]].copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = d[y_col].astype(str)
    d = d.dropna(subset=[x_col])
    if sort_y_desc: d = d.sort_values(x_col, ascending=False)
    if d.empty: return alt.Chart(pd.DataFrame({x_col: [], y_col: []})).mark_bar()
    return (alt.Chart(d).mark_bar().encode(
        x=alt.X(f"{x_col}:Q", title=x_col),
        y=alt.Y(f"{y_col}:N", sort="-x", title=y_col),
        tooltip=[alt.Tooltip(f"{y_col}:N", title=y_col),
                 alt.Tooltip(f"{x_col}:Q", title=x_col)],
    ).properties(height=320, use_container_width=True))

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
    enc = dict(x=x_enc, y=alt.Y(f"{y_col}:Q", title=y_col),
               tooltip=[alt.Tooltip(f"{x_col}", title=x_col),
                        alt.Tooltip(f"{y_col}:Q", title=y_col)])
    if color: enc["color"] = alt.Color(f"{color}:N", title=color)
    return alt.Chart(d).mark_line(point=True).encode(**enc).properties(height=320, use_container_width=True)

# ========= Isolador de abas (cada aba independente) =========
def render_tab(tab, fn, df: pd.DataFrame, title: str):
    """Executa uma aba isoladamente; erros não derrubam as outras."""
    with tab:
        try:
            fn(df.copy())  # cada aba recebe uma cópia
        except Exception as e:
            st.error(f"Erro nesta aba **{title}**: {type(e).__name__}")
            st.exception(e)

# ========= Abas (funções 100% independentes) =========
def tab1_painel(df: pd.DataFrame):
    st.subheader("Painel")
    st.markdown("**Pesquisas únicas**")
    st.markdown(f"<h2 style='margin-top:-10px;'>{fmt_int(df['IDPESQUISA'].nunique())}</h2>", unsafe_allow_html=True)

    total_pesq = df["IDPESQUISA"].nunique() or 1
    cov = {r: df.loc[df["RANKING"].eq(r), "IDPESQUISA"].nunique() for r in (1,2,3)}
    st.markdown(
        f"<div style='opacity:.85'>Cobertura por Ranking "
        f"<b>1º</b>: {fmt_int(cov[1])} ({cov[1]/total_pesq*100:.1f}%) • "
        f"<b>2º</b>: {fmt_int(cov[2])} ({cov[2]/total_pesq*100:.1f}%) • "
        f"<b>3º</b>: {fmt_int(cov[3])} ({cov[3]/total_pesq*100:.1f}%)</div>",
        unsafe_allow_html=True
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    W  = winners_by_position(df)
    Wg = W.replace({"R1":{"MAXMILHAS":"GRUPO 123","123MILHAS":"GRUPO 123"},
                    "R2":{"MAXMILHAS":"GRUPO 123","123MILHAS":"GRUPO 123"},
                    "R3":{"MAXMILHAS":"GRUPO 123","123MILHAS":"GRUPO 123"}})

    def bloco_percent(title: str, base: pd.DataFrame, target: str):
        p1 = (base["R1"] == target).mean()*100
        p2 = (base["R2"] == target).mean()*100
        p3 = (base["R3"] == target).mean()*100
        st.markdown(f"### {title}")
        c1,c2,c3 = st.columns(3)
        for i,(lab,val) in enumerate([("1º",p1),("2º",p2),("3º",p3)]):
            with (c1 if i==0 else c2 if i==1 else c3):
                st.caption(lab); st.markdown(f"<h3 style='margin-top:-8px;'>{val:.2f}%</h3>", unsafe_allow_html=True)
                st.progress(int(round(val)))

    bloco_percent("GRUPO 123", Wg, "GRUPO 123")
    st.markdown("<hr>", unsafe_allow_html=True)
    bloco_percent("FLIPMILHAS", W, "FLIPMILHAS")

def tab2_top3_agencias(df: pd.DataFrame):
    st.subheader("Top 3 Agências (mais vitórias em 1º)")
    W = winners_by_position(df)
    vc = W["R1"].value_counts()
    t3 = vc.head(3).rename_axis("Agência/Cia").reset_index(name="Vitórias")
    if t3.empty: st.info("Sem dados para os filtros."); return
    total = int(vc.sum()) or 1
    cols = st.columns(len(t3))
    for i, (_, row) in enumerate(t3.iterrows()):
        nome = str(row["Agência/Cia"]); v = int(row["Vitórias"]); pct = round(v/total*100, 2)
        with cols[i]:
            st.markdown(f"**{nome}**")
            st.markdown(f"<h3 style='margin-top:-8px;'>{pct:.2f}%</h3>", unsafe_allow_html=True)
            st.progress(int(round(pct)))

def tab3_top3_precos(df: pd.DataFrame):
    st.subheader("Top 3 Preços Mais Baratos (geral filtrado)")
    t = df.sort_values("PRECO").head(3)[["AGENCIA_NORM","TRECHO","PRECO","DATAHORA_BUSCA"]]
    if t.empty: st.info("Sem dados."); return
    cols = st.columns(len(t))
    for i, (_,row) in enumerate(t.iterrows()):
        with cols[i]:
            st.metric(f"{row['AGENCIA_NORM']} • {row['TRECHO']}",
                      value=f"R$ {row['PRECO']:,.2f}".replace(",", "X").replace(".", ",").replace("X","."),
                      delta=f"{row['DATAHORA_BUSCA']:%d/%m/%Y}")

def tab4_ranking_agencias(df: pd.DataFrame):
    st.subheader("Ranking por Agências")
    W = winners_by_position(df)
    wins = W["R1"].value_counts().rename_axis("Agência/Cia").reset_index(name="Vitórias 1º")
    vol  = df["AGENCIA_NORM"].value_counts().rename_axis("Agência/Cia").reset_index(name="Ofertas")
    rt   = vol.merge(wins, on="Agência/Cia", how="left").fillna(0)
    rt["Taxa Vitória (%)"] = (rt["Vitórias 1º"]/rt["Ofertas"]*100).round(2)
    c1,c2 = st.columns(2)
    with c1: st.altair_chart(make_bar(rt[["Agência/Cia","Vitórias 1º"]], "Vitórias 1º", "Agência/Cia"))
    with c2: st.altair_chart(make_bar(rt[["Agência/Cia","Ofertas"]], "Ofertas", "Agência/Cia"))

def tab5_preco_periodo(df: pd.DataFrame):
    st.subheader("Preço por Período do Dia (HH da busca)")
    t = df.groupby("HORA_HH", as_index=False)["PRECO"].median().rename(columns={"PRECO":"Preço Mediano"})
    st.altair_chart(make_line(t, "HORA_HH", "Preço Mediano"))

def tab6_buscas_vs_ofertas(df: pd.DataFrame):
    st.subheader("Quantidade de Buscas x Ofertas")
    searches = df["IDPESQUISA"].nunique(); offers = len(df)
    c1,c2 = st.columns(2)
    c1.metric("Pesquisas únicas", fmt_int(searches))
    c2.metric("Ofertas (linhas)", fmt_int(offers))
    t = pd.DataFrame({"Métrica":["Pesquisas","Ofertas"], "Valor":[searches, offers]})
    st.altair_chart(make_bar(t, "Valor", "Métrica"))

def tab7_comportamento_cias(df: pd.DataFrame):
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
    ).properties(height=360, use_container_width=True)
    st.altair_chart(chart)

def tab8_competitividade(df: pd.DataFrame):
    st.subheader("Competitividade (Δ mediano vs melhor preço por pesquisa)")
    best = df.groupby("IDPESQUISA")["PRECO"].min().rename("BEST").reset_index()
    t = df.merge(best, on="IDPESQUISA", how="left")
    t["DELTA"] = t["PRECO"] - t["BEST"]
    agg = t.groupby("AGENCIA_NORM", as_index=False)["DELTA"].median().rename(columns={"DELTA":"Δ Mediano"})
    st.altair_chart(make_bar(agg, "Δ Mediano", "AGENCIA_NORM"))

def tab9_melhor_preco_diario(df: pd.DataFrame):
    st.subheader("Melhor Preço Diário (col. H - Data da busca)")
    t = df.groupby(df["DATAHORA_BUSCA"].dt.date, as_index=False)["PRECO"].min().rename(
        columns={"DATAHORA_BUSCA":"Data","PRECO":"Melhor Preço"}
    )
    if t.empty: st.info("Sem dados."); return
    t["Data"] = pd.to_datetime(t["Data"], dayfirst=True)
    st.altair_chart(make_line(t, "Data", "Melhor Preço"))

def tab10_exportar(df: pd.DataFrame):
    st.subheader("Exportar dados filtrados")
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Baixar CSV (filtro aplicado)", data=csv_bytes,
                       file_name="OFERTAS_filtrado.csv", mime="text/csv")

# ========= Main =========
def main():
    df_raw = load_base(DATA_PATH)

    # banner no topo (1ª imagem da raiz)
    banner = None
    for ext in ("*.png","*.jpg","*.jpeg","*.gif","*.webp"):
        imgs = list(APP_DIR.glob(ext))
        if imgs: banner = imgs[0]; break
    if banner: st.image(banner.as_posix(), use_container_width=True)

    # abas (criadas ANTES dos filtros)
    labels = [
        "1. Painel","2. Top 3 Agências","3. Top 3 Preços Mais Baratos","4. Ranking por Agências",
        "5. Preço por Período do Dia","6. Qtde de Buscas x Ofertas","7. Comportamento Cias",
        "8. Competitividade","9. Melhor Preço Diário","10. Exportar"
    ]
    tabs = st.tabs(labels)

    # filtros IMEDIATAMENTE abaixo das abas (horizontais)
    with st.container():
        c1, c2, c3, c4, c5 = st.columns([1.2,1.2,1,2,1.2])
        dmin = pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce").min()
        dmax = pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce").max()
        dmin = dmin.date() if pd.notna(dmin) else date(2000,1,1)
        dmax = dmax.date() if pd.notna(dmax) else date.today()

        with c1:
            dt_ini = st.date_input("Data inicial (col. H)", value=dmin, min_value=dmin, max_value=dmax, format="DD/MM/YYYY")
        with c2:
            dt_fim = st.date_input("Data final (col. H)", value=dmax, min_value=dmin, max_value=dmax, format="DD/MM/YYYY")
        with c3:
            advp_sel = st.multiselect("ADVP (col. L)", [1,5,11,17,30], default=[1,5,11,17,30])
        with c4:
            trechos = sorted([t for t in df_raw["TRECHO"].dropna().unique().tolist() if str(t).strip()!=""])
            tr_sel = st.multiselect("Trechos (col. K)", trechos, default=[])
        with c5:
            hh_sel = st.multiselect("Hora da busca HH (col. C)", list(range(24)), default=[])

    # aplica filtros globais
    mask = pd.Series(True, index=df_raw.index)
    mask &= (pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce") >= pd.Timestamp(dt_ini))
    mask &= (pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce") <= pd.Timestamp(dt_fim))
    mask &= df_raw["ADVP_CANON"].isin(advp_sel)
    if tr_sel:  mask &= df_raw["TRECHO"].isin(tr_sel)
    if hh_sel:  mask &= df_raw["HORA_HH"].isin(hh_sel)
    df = df_raw[mask].copy()

    # info global discreta (como no layout desejado)
    st.caption(f"Linhas após filtros: {fmt_int(len(df))} • Última atualização: {last_update_from_cols(df)}")

    # renderização ISOLADA das abas
    render_tab(tabs[0],  tab1_painel,            df, "Painel")
    render_tab(tabs[1],  tab2_top3_agencias,     df, "Top 3 Agências")
    render_tab(tabs[2],  tab3_top3_precos,       df, "Top 3 Preços Mais Baratos")
    render_tab(tabs[3],  tab4_ranking_agencias,  df, "Ranking por Agências")
    render_tab(tabs[4],  tab5_preco_periodo,     df, "Preço por Período do Dia")
    render_tab(tabs[5],  tab6_buscas_vs_ofertas, df, "Qtde de Buscas x Ofertas")
    render_tab(tabs[6],  tab7_comportamento_cias,df, "Comportamento Cias")
    render_tab(tabs[7],  tab8_competitividade,   df, "Competitividade")
    render_tab(tabs[8],  tab9_melhor_preco_diario, df, "Melhor Preço Diário")
    render_tab(tabs[9],  tab10_exportar,         df, "Exportar")

if __name__ == "__main__":
    main()
