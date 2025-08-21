
import os
import io
from datetime import datetime
from typing import Optional, Tuple, List

import pandas as pd
import streamlit as st

st.set_page_config(page_title="OFERTAS.parquet — Viewer", layout="wide", initial_sidebar_state="expanded")

# =============== Helpers ===============
@st.cache_data(show_spinner=False)
def load_parquet_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(data))

@st.cache_data(show_spinner=True)
def load_parquet_path(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def try_find_default_path() -> Optional[str]:
    # Tenta data/OFERTAS.parquet primeiro, depois raiz
    candidates = [
        os.path.join("data", "OFERTAS.parquet"),
        "OFERTAS.parquet",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def detect_dt_columns(df: pd.DataFrame) -> List[str]:
    # Tenta detectar colunas de data/datetime
    dt_cols = []
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            dt_cols.append(c)
        else:
            # tenta converter rapidamente strings para datetime (sem erro)
            sample = df[c].dropna().astype(str).head(20)
            ok = False
            for s in sample:
                try:
                    # formatos comuns
                    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
                        try:
                            datetime.strptime(s, fmt)
                            ok = True
                            break
                        except Exception:
                            pass
                    if ok:
                        break
                except Exception:
                    pass
            if ok:
                try:
                    df[c] = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True, dayfirst=True)
                    if pd.api.types.is_datetime64_any_dtype(df[c]):
                        dt_cols.append(c)
                except Exception:
                    pass
    return dt_cols

def low_card_cols(df: pd.DataFrame, max_unique: int = 50, max_cols: int = 8) -> List[str]:
    cols = []
    for c in df.columns:
        try:
            nunique = int(df[c].nunique(dropna=True))
        except Exception:
            continue
        if nunique <= max_unique:
            cols.append(c)
        if len(cols) >= max_cols:
            break
    return cols

def format_last_update(df: pd.DataFrame, dt_cols: List[str]) -> str:
    if dt_cols:
        try:
            mx = None
            for c in dt_cols:
                val = pd.to_datetime(df[c], errors="coerce").max()
                if pd.notna(val) and (mx is None or val > mx):
                    mx = val
            if mx is not None:
                return mx.strftime("%d/%m/%Y %H:%M:%S")
        except Exception:
            pass
    return "—"

# =============== UI ===============
st.markdown("### 📦 Leitor de `OFERTAS.parquet`")
st.caption("Básico, direto, sem frescura. Carrega e filtra seus dados do Parquet.")

with st.sidebar:
    st.subheader("Arquivo de entrada")
    default_path = try_find_default_path()
    if default_path:
        st.success(f"Encontrado: `{default_path}`")
    else:
        st.info("Coloque **data/OFERTAS.parquet** no repo ou envie abaixo.")
    uploaded = st.file_uploader("Ou envie o arquivo `.parquet`", type=["parquet"], accept_multiple_files=False)

    row_limit = st.number_input("Limite de linhas (visualização)", 1000, 2_000_000, value=100_000, step=10_000)
    st.caption("Dica: limite maior = mais memória/tempo.")

# Carregamento
df = None
src = None

try:
    if uploaded is not None:
        df = load_parquet_bytes(uploaded.read())
        src = f"upload: {uploaded.name}"
    elif default_path is not None:
        df = load_parquet_path(default_path)
        src = default_path
except Exception as e:
    st.error(f"Falha ao carregar o Parquet: {e}")

if df is None:
    st.stop()

# Preview e metadados
n_rows, n_cols = df.shape
dt_cols = detect_dt_columns(df.copy())
last_up = format_last_update(df, dt_cols)

m1, m2, m3 = st.columns(3)
m1.metric("Linhas", f"{n_rows:,}".replace(",", "."))
m2.metric("Colunas", f"{n_cols:,}".replace(",", "."))
m3.metric("Última atualização (detectada)", last_up)

st.caption(f"Fonte: **{src}**")

# Filtros rápidos com baixa cardinalidade
st.markdown("#### 🔍 Filtros rápidos (baixa cardinalidade)")
fcols = low_card_cols(df)
if fcols:
    with st.expander("Abrir filtros", expanded=True):
        filters = {}
        for c in fcols:
            vals = sorted([v for v in df[c].dropna().unique().tolist() if str(v).strip() != "" ])
            sel = st.multiselect(f"{c}", vals, default=[])
            filters[c] = sel

        # aplica filtros
        for c, sel in filters.items():
            if sel:
                df = df[df[c].isin(sel)]
else:
    st.info("Nenhuma coluna com baixa cardinalidade encontrada (≤ 50 valores).")

# Busca de texto livre
st.markdown("#### 🧭 Filtro por texto (todas as colunas)")
q = st.text_input("Contém (case-insensitive):", value="")
if q.strip():
    q_lower = q.strip().lower()
    df = df[df.apply(lambda r: q_lower in " ".join(map(str, r.values)).lower(), axis=1)]

# Limitador para visualização
if len(df) > row_limit:
    st.warning(f"Exibindo apenas as primeiras {row_limit:,} linhas de {len(df):,}. Ajuste o limite na barra lateral.".replace(",", "."))
df_view = df.head(row_limit)

st.markdown("#### 📋 Dados")
st.dataframe(df_view, use_container_width=True, height=480)

# Download dos dados filtrados
csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
st.download_button("⬇️ Baixar CSV (filtro aplicado)", csv_bytes, file_name="OFERTAS_filtrado.csv", mime="text/csv")

st.caption("Cache: use o menu de três pontos no topo direito para 'Rerun' se trocar o arquivo.")
