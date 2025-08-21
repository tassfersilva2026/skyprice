# streamlit_app.py
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st

st.set_page_config(page_title="OFERTAS.parquet — Viewer", layout="wide", initial_sidebar_state="expanded")

# Caminho fixo: repo_root/data/OFERTAS.parquet
DATA_PATH = Path(__file__).resolve().parent / "data" / "OFERTAS.parquet"

if not DATA_PATH.exists():
    st.error(f"Arquivo obrigatório não encontrado:\n{DATA_PATH}\n\n"
             "Garanta que o arquivo exista no repositório exatamente nesse caminho.")
    st.stop()

@st.cache_data(show_spinner=True)
def load_parquet(path: Path) -> pd.DataFrame:
    # Usa pyarrow por padrão (requirements já cobre)
    return pd.read_parquet(path)

def detect_last_update(df: pd.DataFrame) -> str:
    # tenta achar a "maior" data em qualquer coluna datetime; se não houver, retorna “—”
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if not dt_cols:
        # tentativa leve de converter strings comuns
        for c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True, dayfirst=True)
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    dt_cols.append(c)
            except Exception:
                pass
    if dt_cols:
        try:
            mx = max([pd.to_datetime(df[c], errors="coerce").max() for c in dt_cols])
            if pd.notna(mx):
                return mx.strftime("%d/%m/%Y %H:%M:%S")
        except Exception:
            pass
    return "—"

# ===== Carrega =====
df = load_parquet(DATA_PATH)

# ===== Header =====
st.markdown("### 📦 Leitor fixo de `data/OFERTAS.parquet`")
st.caption(f"Caminho: `{DATA_PATH.as_posix()}`")

# ===== Métricas =====
n_rows, n_cols = df.shape
last_up = detect_last_update(df.copy())

m1, m2, m3 = st.columns(3)
m1.metric("Linhas", f"{n_rows:,}".replace(",", "."))
m2.metric("Colunas", f"{n_cols:,}".replace(",", "."))
m3.metric("Última atualização (detectada)", last_up)

# ===== Controles simples =====
row_limit = st.number_input("Limite de linhas para exibir", 1000, 2_000_000, value=100_000, step=10_000)
q = st.text_input("Filtro por texto (todas as colunas):", "")

df_view = df
if q.strip():
    ql = q.lower().strip()
    df_view = df_view[df_view.apply(lambda r: ql in " ".join(map(str, r.values)).lower(), axis=1)]

if len(df_view) > row_limit:
    st.warning(f"Exibindo apenas as primeiras {row_limit:,} de {len(df_view):,} linhas."
               .replace(",", "."))

st.dataframe(df_view.head(row_limit), use_container_width=True, height=480)

# ===== Download filtrado =====
csv_bytes = df_view.to_csv(index=False).encode("utf-8-sig")
st.download_button("⬇️ Baixar CSV (com filtro aplicado)", csv_bytes, file_name="OFERTAS_filtrado.csv", mime="text/csv")
