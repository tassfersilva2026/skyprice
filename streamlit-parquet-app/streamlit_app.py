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

# =========================
# Helpers
# =========================
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
    except Exception: v = np.nan
    if np.isnan(v): v = 1
    return min([1,5,11,17,30], key=lambda k: abs(v-k))

@st.cache_data(show_spinner=True)
def load_base(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Arquivo obrigatório não encontrado: `{path.as_posix()}`"); st.stop()
    df = pd.read_parquet(path)

    # Posições → nomes
    colmap = {
        0:"IDPESQUISA", 1:"CIA", 2:"HORA_BUSCA", 3:"HORA_PARTIDA", 4:"HORA_CHEGADA",
        5:"TIPO_VOO", 6:"DATA_EMBARQUE", 7:"DATAHORA_BUSCA", 8:"AGENCIA_COMP",
        9:"PRECO", 10:"TRECHO", 11:"ADVP", 12:"RANKING"
    }
    if list(df.columns[:13]) != list(colmap.values()):
        rename = {df.columns[i]: colmap[i] for i in range(min(13, df.shape[1]))}
        df = df.rename(columns=rename)

    # Horas texto → HH e time string
    for c in ["HORA_BUSCA", "HORA_PARTIDA", "HORA_CHEGADA"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c].astype(str).str.strip(), errors="coerce").dt.strftime("%H:%M:%S")
    df["HORA_HH"] = pd.to_datetime(df["HORA_BUSCA"], errors="coerce").dt.hour

    # Datas (dd/mm/aaaa)
    for c in ["DATA_EMBARQUE", "DATAHORA_BUSCA"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)

    # Preço
    if "PRECO" in df.columns:
        df["PRECO"] = (df["PRECO"].astype(str)
                       .str.replace(r"[^\d,.-]", "", regex=True)
                       .str.replace(",", ".", regex=False))
        df["PRECO"] = pd.to_numeric(df["PRECO"], errors="coerce")

    # Ranking
    if "RANKING" in df.columns:
        df["RANKING"] = pd.to_numeric(df["RANKING"], errors="coerce").astype("Int64")

    # Normalizações
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

def make_bar(df, x, y, sort_y_desc=True):
    if sort_y_desc: df = df.sort_values(x, ascending=False)
    return alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{x}:Q"), y=alt.Y(f"{y}:N", sort="-x"), tooltip=list(df.columns)
    ).properties(height=320, use_container_width=True)

def make_line(df, x, y, color=None):
    enc = dict(x=alt.X(f"{x}:T"), y=alt.Y(f"{y}:Q"), tooltip=list(df.columns))
    if color: enc["color"] = alt.Color(f"{color}:N")
    return alt.Chart(df).mark_line(point=True).encode(**enc).properties(height=320, use_container_width=True)

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
# =========================
# Load + banner
# =========================
df_raw = load_base(DATA_PATH)

banner = None
for ext in ("*.png","*.jpg","*.jpeg","*.gif","*.webp"):
    imgs = list(APP_DIR.glob(ext))
    if imgs: banner = imgs[0]; break
if banner: st.image(banner.as_posix(), use_container_width=True)

# =========================
# Abas (título → filtros logo abaixo)
# =========================
tab_labels = [
    "1. Painel",
    "2. Top 3 Agências",
    "3. Top 3 Preços Mais Baratos",
    "4. Ranking por Agências",
    "5. Preço por Período do Dia",
    "6. Qtde de Buscas x Ofertas",
    "7. Comportamento Cias",
    "8. Competitividade",
    "9. Melhor Preço Diário",
    "10. Exportar"
]
tabs = st.tabs(tab_labels)

# =========================
# Filtros (ABAIXO das abas) — dd/mm/aaaa
# =========================
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

# Linha discreta — igual à imagem
st.caption(f"Linhas após filtros: {fmt_int(len(df))} • Última atualização: {last_update_from_cols(df)}")

# =========================
# 1) PAINEL — sem tabelas
# =========================
with tabs[0]:
    st.subheader("Painel")

    # Pesquisas únicas
    st.markdown("**Pesquisas únicas**")
    st.markdown(f"<h2 style='margin-top:-10px;'>{fmt_int(df['IDPESQUISA'].nunique())}</h2>", unsafe_allow_html=True)

    # Cobertura por Ranking (1º/2º/3º)
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

    # Blocos de % por posição (GRUPO 123 e FLIPMILHAS)
    W = winners_by_position(df)
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
                st.caption(lab)
                st.markdown(f"<h3 style='margin-top:-8px;'>{val:.2f}%</h3>", unsafe_allow_html=True)
                st.progress(int(round(val)))

    bloco_percent("GRUPO 123", Wg, "GRUPO 123")
    st.markdown("<hr>", unsafe_allow_html=True)
    bloco_percent("FLIPMILHAS", W, "FLIPMILHAS")

# =========================
# 2) TOP 3 AGÊNCIAS — barras
# =========================
with tabs[1]:
    st.subheader("Top 3 Agências (mais vitórias em 1º)")
    if df.empty:
        st.info("Sem dados.")
    else:
        W = winners_by_position(df)
        vc = W["R1"].value_counts()
        t3 = vc.head(3).rename_axis("Agência/Cia").reset_index(name="Vitórias")
        st.altair_chart(make_bar(t3, "Vitórias", "Agência/Cia"))

# =========================
# 3) TOP 3 PREÇOS MAIS BARATOS — cards
# =========================
with tabs[2]:
    st.subheader("Top 3 Preços Mais Baratos (geral filtrado)")
    if df.empty:
        st.info("Sem dados.")
    else:
        t = df.sort_values("PRECO").head(3)[["AGENCIA_NORM","TRECHO","PRECO","DATAHORA_BUSCA"]]
        cols = st.columns(3)
        for i, (_,row) in enumerate(t.iterrows()):
            with cols[i]:
                st.metric(f"{row['AGENCIA_NORM']} • {row['TRECHO']}",
                          value=f"R$ {row['PRECO']:,.2f}".replace(",", "X").replace(".", ",").replace("X","."),
                          delta=f"{row['DATAHORA_BUSCA']:%d/%m/%Y}")

# =========================
# 4) RANKING POR AGÊNCIAS — barras
# =========================
with tabs[3]:
    st.subheader("Ranking por Agências")
    if df.empty:
        st.info("Sem dados.")
    else:
        W = winners_by_position(df)
        wins = W["R1"].value_counts().rename_axis("Agência/Cia").reset_index(name="Vitórias 1º")
        vol  = df["AGENCIA_NORM"].value_counts().rename_axis("Agência/Cia").reset_index(name="Ofertas")
        rt   = vol.merge(wins, on="Agência/Cia", how="left").fillna(0)
        rt["Taxa Vitória (%)"] = (rt["Vitórias 1º"]/rt["Ofertas"]*100).round(2)
        c1,c2 = st.columns(2)
        with c1: st.altair_chart(make_bar(rt[["Agência/Cia","Vitórias 1º"]], "Vitórias 1º", "Agência/Cia"))
        with c2: st.altair_chart(make_bar(rt[["Agência/Cia","Ofertas"]], "Ofertas", "Agência/Cia"))

# =========================
# 5) PREÇO POR PERÍODO DO DIA — linha
# =========================
with tabs[4]:
    st.subheader("Preço por Período do Dia (HH da busca)")
    if df.empty:
        st.info("Sem dados.")
    else:
        t = df.groupby("HORA_HH", as_index=False)["PRECO"].median().rename(columns={"PRECO":"Preço Mediano"})
        st.altair_chart(make_line(t, "HORA_HH", "Preço Mediano"))

# =========================
# 6) QTDE DE BUSCAS x OFERTAS — métricas + barra
# =========================
with tabs[5]:
    st.subheader("Quantidade de Buscas x Ofertas")
    if df.empty:
        st.info("Sem dados.")
    else:
        searches = df["IDPESQUISA"].nunique()
        offers   = len(df)
        c1,c2 = st.columns(2)
        c1.metric("Pesquisas únicas", fmt_int(searches))
        c2.metric("Ofertas (linhas)", fmt_int(offers))
        t = pd.DataFrame({"Métrica":["Pesquisas","Ofertas"], "Valor":[searches, offers]})
        st.altair_chart(make_bar(t, "Valor", "Métrica"))

# =========================
# 7) COMPORTAMENTO CIAS — share (top 10 trechos) empilhado 100%
# =========================
with tabs[6]:
    st.subheader("Comportamento Cias (share por Trecho)")
    if df.empty:
        st.info("Sem dados.")
    else:
        base = df.groupby(["TRECHO","AGENCIA_NORM"]).size().rename("Qtde").reset_index()
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

# =========================
# 8) COMPETITIVIDADE — Δ mediano
# =========================
with tabs[7]:
    st.subheader("Competitividade (Δ mediano vs melhor preço por pesquisa)")
    if df.empty:
        st.info("Sem dados.")
    else:
        best = df.groupby("IDPESQUISA")["PRECO"].min().rename("BEST").reset_index()
        t = df.merge(best, on="IDPESQUISA", how="left")
        t["DELTA"] = t["PRECO"] - t["BEST"]
        agg = t.groupby("AGENCIA_NORM", as_index=False)["DELTA"].median().rename(columns={"DELTA":"Δ Mediano"})
        st.altair_chart(make_bar(agg, "Δ Mediano", "AGENCIA_NORM"))

# =========================
# 9) MELHOR PREÇO DIÁRIO — linha
# =========================
with tabs[8]:
    st.subheader("Melhor Preço Diário (col. H - Data da busca)")
    if df.empty:
        st.info("Sem dados.")
    else:
        t = df.groupby(df["DATAHORA_BUSCA"].dt.date, as_index=False)["PRECO"].min().rename(columns={"DATAHORA_BUSCA":"Data","PRECO":"Melhor Preço"})
        t["Data"] = pd.to_datetime(t["Data"], dayfirst=True)
        st.altair_chart(make_line(t, "Data", "Melhor Preço"))

# =========================
# 10) EXPORTAR — sem tabela, só download
# =========================
with tabs[9]:
    st.subheader("Exportar dados filtrados")
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Baixar CSV (filtro aplicado)", data=csv_bytes,
                       file_name="OFERTAS_filtrado.csv", mime="text/csv")
