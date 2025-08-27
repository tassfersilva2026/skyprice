# loader.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional
import hashlib
import polars as pl
import pyarrow as pa

DATA_DIR = Path(__file__).resolve().parent / "data"
LEGACY = DATA_DIR / "OFERTAS.parquet"                # arquivo pesadão
INCS   = sorted(DATA_DIR.glob("OFERTAS_*.parquet"))  # incrementais (se existirem)

# --- util: fingerprint p/ cache (muda quando arquivos mudam) ---
def _fingerprint_files(files: Iterable[Path]) -> str:
    h = hashlib.md5()
    for p in files:
        if p.exists():
            st = p.stat()
            h.update(f"{p.name}|{int(st.st_mtime)}|{st.st_size}".encode())
    return h.hexdigest()

def _list_files() -> list[Path]:
    files = []
    if LEGACY.exists(): files.append(LEGACY)
    files.extend(INCS)
    return files

def scan() -> pl.LazyFrame:
    files = _list_files()
    if not files:
        # evita exceção quando vazio
        return pl.LazyFrame(schema={"SEM_DADOS": pl.Int32}).limit(0)
    # leitura lazy com predicate/column pushdown
    return pl.scan_parquet([str(p) for p in files])

def file_fingerprint() -> str:
    return _fingerprint_files(_list_files())

# ---------- APIs principais ----------
def load_slice(
    dt_col: str,                     # ex: "DATA DA BUSCA" (date ou string)
    start_date: str,                 # "2025-08-20"
    end_date: str,                   # "2025-08-27"
    cols: Optional[list[str]] = None,
    filtros: Optional[dict] = None,  # {"CIA DO VOO": ["AZUL","GOL"], "TRECHO": ["GRU-REC"]}
    limit_rows: Optional[int] = None,
) -> pl.DataFrame:
    lf = scan()

    # garante tipo date (funciona se vier string ou date)
    if lf.schema.get(dt_col) != pl.Date:
        lf = lf.with_columns(
            pl.col(dt_col)
            .str.strptime(pl.Date, strict=False, fmt=None)   # tenta inferir
            .alias(dt_col)
        )

    lf = lf.filter(pl.col(dt_col).is_between(start_date, end_date, closed="both"))

    if filtros:
        for k, v in filtros.items():
            if v:
                lf = lf.filter(pl.col(k).is_in(v))

    if cols:
        lf = lf.select([pl.col(c) for c in cols])

    if limit_rows:
        lf = lf.limit(limit_rows)

    # coleta em modo streaming (economiza memória para groupbys e scans longos)
    return lf.collect(streaming=True)

def load_grouped(
    dt_col: str,
    start_date: str,
    end_date: str,
    group_cols: list[str],           # ex: ["CIA DO VOO"]
    aggs: dict[str, list[str]],      # {"TOTAL": ["sum"], "ID": ["count"]}
    filtros: Optional[dict] = None,
) -> pl.DataFrame:
    lf = scan()

    if lf.schema.get(dt_col) != pl.Date:
        lf = lf.with_columns(
            pl.col(dt_col).str.strptime(pl.Date, strict=False).alias(dt_col)
        )

    lf = lf.filter(pl.col(dt_col).is_between(start_date, end_date, closed="both"))

    if filtros:
        for k, v in filtros.items():
            if v:
                lf = lf.filter(pl.col(k).is_in(v))

    agg_exprs = []
    for col, fns in aggs.items():
        for fn in fns:
            if fn == "sum":
                agg_exprs.append(pl.col(col).sum().alias(f"{col}_sum"))
            elif fn == "count":
                agg_exprs.append(pl.len().alias(f"{col}_count"))  # contagem de linhas
            elif fn == "mean":
                agg_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
            elif fn == "min":
                agg_exprs.append(pl.col(col).min().alias(f"{col}_min"))
            elif fn == "max":
                agg_exprs.append(pl.col(col).max().alias(f"{col}_max"))

    lf = lf.group_by(group_cols).agg(agg_exprs)
    return lf.collect(streaming=True)

# modo amostra para pré-visualização rápida (sem ORDER BY random pesado)
def load_sample(
    dt_col: str,
    start_date: str,
    end_date: str,
    sample_rows: int = 100_000,
    cols: Optional[list[str]] = None,
    filtros: Optional[dict] = None,
) -> pl.DataFrame:
    lf = scan()

    if lf.schema.get(dt_col) != pl.Date:
        lf = lf.with_columns(
            pl.col(dt_col).str.strptime(pl.Date, strict=False).alias(dt_col)
        )

    lf = lf.filter(pl.col(dt_col).is_between(start_date, end_date, closed="both"))
    if filtros:
        for k, v in filtros.items():
            if v:
                lf = lf.filter(pl.col(k).is_in(v))
    if cols:
        lf = lf.select([pl.col(c) for c in cols])

    # sample por reservoir em streaming
    return lf.sample(n=sample_rows, shuffle=True).collect(streaming=True)
