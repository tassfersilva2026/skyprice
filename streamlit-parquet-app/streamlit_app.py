# writer.py
from __future__ import annotations
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from datetime import datetime

ADVP_BUCKETS = [1,5,11,17,30]

def bucket_advp(x) -> int:
    x = int(x)
    return min(ADVP_BUCKETS, key=lambda b: abs(x - b))

def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    # Ajuste nomes conforme seu schema real (A..M -> nomes semânticos)
    # Exemplo mínimo:
    # H = DATA_BUSCA, C = HORA_BUSCA, L = ADVP, K = TRECHO, I = AGENCIA, J = PRECO
    df = df.copy()
    df["DATA_BUSCA"] = pd.to_datetime(df["DATA_BUSCA"]).dt.date.astype(str)
    # Hora cheia
    df["HORA_BUSCA"] = pd.to_datetime(df["HORA_BUSCA"], format="%H:%M:%S", errors="coerce").dt.hour.astype("Int16")
    # ADVP bucketizado
    df["ADVP"] = df["ADVP"].astype("Int16").map(bucket_advp)
    # Categorias
    for col in ["AGENCIA","TRECHO"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    # Numéricos compactos
    if "PRECO" in df.columns:
        df["PRECO"] = pd.to_numeric(df["PRECO"], errors="coerce").astype("float32")
    return df

def write_daily(df_raw: pd.DataFrame, dia: str,
                out_snap="data/snapshots",
                out_ds="data/silver/ofertas_dataset"):
    df = normalize_types(df_raw)
    # 1) Snapshot do dia (auditoria)
    df_dia = df[df["DATA_BUSCA"] == dia]
    if not df_dia.empty:
        pq.write_table(
            pa.Table.from_pandas(df_dia, preserve_index=False),
            f"{out_snap}/OFERTAS_{dia}.parquet",
            compression="zstd", use_dictionary=True
        )
    # 2) Dataset particionado (leitura rápida no app)
    pq.write_to_dataset(
        pa.Table.from_pandas(df, preserve_index=False),
        root_path=out_ds,
        partition_cols=["ADVP","DATA_BUSCA"],
        compression="zstd", use_dictionary=True,
        row_group_size=50_000, data_page_size=1<<16
    )
