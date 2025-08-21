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

# =========================================================
# Helpers de normalização e carregamento
# =========================================================
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
    if np.isnan(v): v = 1
    return min([1,5,11,17,30], key=lambda k: abs(v-k))

@st.cache_data(show_spinner=True)
def load_base(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Arquivo obrigatório não encontrado: `{path.as_posix()}`"); st.stop()
    df = pd.read_parquet(path)

    # Mapa por posição (A..M) caso os nomes venham diferentes
    colmap = {
        0:"IDPESQUISA", 1:"CIA", 2:"HORA_BUSCA", 3:"HORA_PARTIDA", 4:"HORA_CHEGADA",
        5:"TIPO_VOO", 6:"DATA_EMBARQUE", 7:"DATAHORA_BUSCA", 8:"AGENCIA_COMP",
        9:"PRECO", 10:"TRECHO", 11:"ADVP", 12:"RANKING"
    }
    if list(df.columns[:13]) != list(colmap.values()):
        rename = {df.columns[i]: colmap[i] for i in range(min(13, df.shape[1]))}
        df = df.rename(columns=rename)

    # Horas texto -> "HH:MM:SS" + coluna HH (somente hora)
    for c in ["HORA_BUSCA", "HORA_PARTIDA", "HORA_CHEGADA"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c].astype(str).strip(), errors="coerce").dt.strftime("%H:%M:%S")
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
    """Retorna por IDPESQUISA a agência vencedora em R1, R2 e R3 (preenche 'SEM OFERTAS' se não houver)."""
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
    """Última atualização = maior data (H) e, nessa data, maior HORA_BUSCA (C)."""
    if df.empty: return "—"
    max_d = pd.to_datetime(df["DATAHORA_BUSCA"], errors="coerce").max()
    if pd.isna(max_d): return "—"
    same_day = df[pd.to_datetime(df["DATAHORA_BUSCA"], errors="coerce").dt.date == max_d.date()]
    hh = pd.to_datetime(same_day["HORA_BUSCA"], errors="coerce").dt.time
    max_h = max([h for h in hh if pd.notna(h)], default=None)
    if isinstance(max_h, dtime):
        return f"{max_d.strftime('%d/%m/%Y')} - {max_h.strftime('%H:%M:%S')}"
    return f"{max_d.strftime('%d/%m/%Y')}"

# =========================================================
# Helpers de gráfico (coerção → evita SchemaValidationError)
# =========================================================
def make_bar(df: pd.DataFrame, x_col: str, y_col: str, sort_y_desc: bool = True):
    d = df[[y_col, x_col]].copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = d[y_col].astype(str)
    d = d.dropna(subset=[x_col])
    if sort_y_desc: d = d.sort_values(x_col, ascending=False)
    if d.empty:
        return alt.Chart(pd.DataFrame({x_col: [], y_col: []})).mark_bar()
    return (
        alt.Chart(d)
        .mark_bar()
        .encode(
            x=alt.X(f"{x_col}:Q", title=x_col),
            y=alt.Y(f"{y_col}:N", sort="-x", title=y_col),
            tooltip=[alt.Tooltip(f"{y_col}:N", title=y_col),
                     alt.Tooltip(f"{x_col}:Q", title=x_col)],
        ).properties(height=320, use_container_width=True)
    )

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
    if d.empty:
        return alt.Chart(pd.DataFrame({x_col: [], y_col: []})).mark_line()
    enc = dict(x=x_enc, y=alt.Y(f"{y_col}:Q", title=y_col),
               tooltip=[alt.Tooltip(f"{x_col}", title=x_col),
                        alt.Tooltip(f"{y_col}:Q", title=y_col)])
    if color: enc["color"] = alt.Color(f"{color}:N", title=color)
    return alt.Chart(d).mark_line(point=True).encode(**enc).properties(height=320, use_container_width=True)

# =========================================================
# Load base + banner
# =========================================================
df_raw = load_base(DATA_PATH)

banner = None
for ext in ("*.png","*.jpg","*.jpeg","*.gif","*.webp"):
    imgs = list(APP_DIR.glob(ext))
    if imgs: banner = imgs[0]; break
if banner: st.image(banner.as_posix(), use_container_width=True)

# =========================================================
# Abas
# =========================================================
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

# =========================================================
# Filtros (IMEDIATAMENTE abaixo das abas) — dd/mm/aaaa
# =========================================================
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

# Aplica filtros globais
mask = pd.Series(True, index=df_raw.index)
mask &= (pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce") >= pd.Timestamp(dt_ini))
mask &= (pd.to_datetime(df_raw["DATAHORA_BUSCA"], errors="coerce") <= pd.Timestamp(dt_fim))
mask &= df_raw["ADVP_CANON"].isin(advp_sel)
if tr_sel:  mask &= df_raw["TRECHO"].isin(tr_sel)
if hh_sel:  mask &= df_raw["HORA_HH"].isin(hh_sel)
df = df_raw[mask].copy()

# Linha discreta (igual ao print) — fica depois dos filtros e ANTES dos conteúdos
st.caption(f"Linhas após filtros: {fmt_int(len(df))} • Última atualização: {last_update_from_cols(df)}")

# =========================================================
# 1) PAINEL (sem tabelas)
# =========================================================
with tabs[0]:
    st.subheader("Painel")

    # Pesquisas únicas
    st.markdown("**Pesquisas únicas**")
    st.markdown(f"<h2 style='margin-top:-10px;'>{fmt_int(df['IDPESQUISA'].nunique())}</h2>", unsafe_allow_html=True)

    # Cobertura por Ranking
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

    # Blocos: GRUPO 123 e FLIPMILHAS
    def winners_by_position_local(df_local: pd.DataFrame) -> pd.DataFrame:
        base = pd.DataFrame({"IDPESQUISA": df_local["IDPESQUISA"].unique()})
        for r in (1,2,3):
            s = (df_local[df_local["RANKING"]==r]
                 .sort_values(["IDPESQUISA"])
                 .drop_duplicates(subset=["IDPESQUISA"]))
            base = base.merge(s[["IDPESQUISA","AGENCIA_NORM"]].rename(columns={"AGENCIA_NORM":f"R{r}"}),
                              on="IDPESQUISA", how="left")
        for r in (1,2,3):
            base[f"R{r}"] = base[f"R{r}"].fillna("SEM OFERTAS")
        return base

    W = winners_by_position_local(df)
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

# =========================================================
# 2) Top 3 Agências (mais vitórias em 1º) — SEM Altair (cards + progresso)
# =========================================================
with tabs[1]:
    st.subheader("Top 3 Agências (mais vitórias em 1º)")
    if df.empty:
        st.info("Sem dados.")
    else:
        W = winners_by_position(df)
        vc = W["R1"].value_counts()
        t3 = vc.head(3).rename_axis("Agência/Cia").reset_index(name="Vitórias")
        total = int(vc.sum()) or 1
        cols = st.columns(3)
        for i, (_, row) in enumerate(t3.iterrows()):
            nome = str(row["Agência/Cia"])
            v = int(row["Vitórias"])
            pct = round(v/total*100, 2)
            with cols[i]:
                st.markdown(f"**{nome}**")
                st.markdown(f"<h3 style='margin-top:-8px;'>{pct:.2f}%</h3>", unsafe_allow_html=True)
                st.progress(int(round(pct)))

# =========================================================
# 3) Top 3 Preços Mais Baratos — cards
# =========================================================
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

# =========================================================
# 4) Ranking por Agências — barras (vitórias e volume)
# =========================================================
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

# =========================================================
# 5) Preço por Período do Dia (HH) — linha
# =========================================================
with tabs[4]:
    st.subheader("Preço por Período do Dia (HH da busca)")
    if df.empty:
        st.info("Sem dados.")
    else:
        t = df.groupby("HORA_HH", as_index=False)["PRECO"].median().rename(columns={"PRECO":"Preço Mediano"})
        st.altair_chart(make_line(t, "HORA_HH", "Preço Mediano"))

# =========================================================
# 6) Qtde de Buscas x Ofertas — métricas + barra
# =========================================================
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

# =========================================================
# 7) Comportamento Cias — share top 10 trechos empilhado 100%
# =========================================================
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

# =========================================================
# 8) Competitividade — Δ mediano vs melhor preço por pesquisa
# =========================================================
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

# =========================================================
# 9) Melhor Preço Diário — linha
# =========================================================
with tabs[8]:
    st.subheader("Melhor Preço Diário (col. H - Data da busca)")
    if df.empty:
        st.info("Sem dados.")
    else:
        t = df.groupby(df["DATAHORA_BUSCA"].dt.date, as_index=False)["PRECO"].min().rename(
            columns={"DATAHORA_BUSCA":"Data","PRECO":"Melhor Preço"}
        )
        t["Data"] = pd.to_datetime(t["Data"], dayfirst=True)
        st.altair_chart(make_line(t, "Data", "Melhor Preço"))

# =========================================================
# 10) Exportar — sem tabela, só download do filtrado
# =========================================================
with tabs[9]:
    st.subheader("Exportar dados filtrados")
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Baixar CSV (filtro aplicado)", data=csv_bytes,
                       file_name="OFERTAS_filtrado.csv", mime="text/csv")
