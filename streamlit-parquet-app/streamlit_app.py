# ──────────────────── ABA: Top 3 Preços Mais Baratos (START) ─────────────────
@register_tab("Top 3 Preços Mais Baratos")
def tab3_top3_precos(df_raw: pd.DataFrame):
    """
    Pódio por Trecho → ADVP (mesma pesquisa):
    • Para cada (Trecho, ADVP), opcionalmente isola a ÚLTIMA pesquisa.
    • Renderiza cards Top 1/2/3 com preço, delta e badge "?" que mostra/copIa o ID.
    • Filtros: Agência alvo (Todos/123/MAX) e Ranking (Todas/1/2/3).
    """
    import re
    import numpy as np
    import pandas as pd
    import streamlit as st

    # ======== FILTROS (usa os do app + filtros extras desta aba) ========
    df = render_filters(df_raw, key_prefix="t3")
    st.subheader("Pódio por Trecho → ADVP (última pesquisa de cada par)")

    top_row = st.container()
    with top_row:
        c1, c2, c3 = st.columns([0.28, 0.18, 0.54])
        agencia_foco = c1.selectbox("Agência alvo", ["Todos", "123MILHAS", "MAXMILHAS"], index=0)
        posicao_foco = c2.selectbox("Ranking", ["Todas", 1, 2, 3], index=0)
        por_pesquisa = c3.checkbox("Isolar última pesquisa por Trecho×ADVP", value=True)

    if df.empty:
        st.info("Sem resultados para os filtros selecionados."); return

    # ===================== Helpers =====================
    def fmt_moeda_br(x) -> str:
        try:
            xv = float(x)
            if not np.isfinite(xv): return "R$ -"
            return "R$ " + f"{xv:,.0f}".replace(",", ".")
        except Exception:
            return "R$ -"

    def fmt_pct_plus(x) -> str:
        try:
            v = float(x)
            if not np.isfinite(v): return "—"
            return f"+{round(v):.0f}%"
        except Exception:
            return "—"

    def _canon(s: str) -> str:
        return re.sub(r"[^A-Z0-9]+", "", str(s).upper())

    def _brand_tag(s: str) -> str | None:
        cs = _canon(s)
        if cs.startswith("123MILHAS") or cs == "123": return "123MILHAS"
        if cs.startswith("MAXMILHAS") or cs == "MAX":  return "MAXMILHAS"
        return None

    # ====== detectar a coluna de ID da pesquisa ======
    def _find_id_col(df_: pd.DataFrame) -> str | None:
        cands = ["IDPESQUISA","ID_PESQUISA","ID BUSCA","IDBUSCA","ID","NOME_ARQUIVO_STD",
                 "NOME_ARQUIVO","NOME DO ARQUIVO","ARQUIVO"]
        norm = { _canon(c): c for c in df_.columns }
        for nm in cands:
            key = _canon(nm)
            if key in norm: return norm[key]
        # fallback: primeira coluna (coluna A real)
        return df_.columns[0] if len(df_.columns) else None

    # ===================== Layout / CSS =====================
    GRID_STYLE    = "display:grid;grid-auto-flow:column;grid-auto-columns:260px;gap:10px;overflow-x:auto;padding:6px 2px 10px 2px;scrollbar-width:thin;"
    BOX_STYLE     = "border:1px solid #e5e7eb;border-radius:12px;background:#fff;"
    HEAD_STYLE    = "padding:8px 10px;border-bottom:1px solid #f1f5f9;font-weight:700;color:#111827;"
    STACK_STYLE   = "display:grid;gap:8px;padding:8px;"
    CARD_BASE     = "position:relative;border:1px solid #e5e7eb;border-radius:10px;padding:8px;background:#fff;min-height:78px;"
    DT_WRAP_STYLE = "position:absolute;right:8px;top:6px;display:flex;align-items:center;gap:6px;"
    DT_TXT_STYLE  = "font-size:10px;color:#94a3b8;font-weight:800;"
    RANK_STYLE    = "font-weight:900;font-size:11px;color:#6b7280;letter-spacing:.3px;text-transform:uppercase;"
    AG_STYLE      = "font-weight:800;font-size:15px;color:#111827;margin-top:2px;"
    PR_STYLE      = "font-weight:900;font-size:18px;color:#111827;margin-top:2px;"
    SUB_STYLE     = "font-weight:700;font-size:12px;color:#374151;"
    NO_STYLE      = "padding:22px 12px;color:#6b7280;font-weight:800;text-align:center;border:1px dashed #e5e7eb;border-radius:10px;background:#fafafa;"
    TRE_HDR_STYLE = "margin:14px 0 10px 0;padding:10px 12px;border-left:4px solid #0B5FFF;background:#ECF3FF;border-radius:8px;font-weight:800;color:#0A2A6B;"
    EXTRAS_STYLE  = "border-top:1px dashed #e5e7eb;margin:6px 8px 8px 8px;padding-top:6px;display:flex;gap:6px;flex-wrap:wrap;"
    CHIP_STYLE    = "background:#f3f4f6;border:1px solid #e5e7eb;border-radius:10px;padding:2px 6px;font-weight:700;font-size:11px;color:#111827;white-space:nowrap;line-height:1.1;"

    BADGE_POP_CSS = """
    <style>
    .idp-wrap{position:relative; display:inline-flex; align-items:center;}
    .idp-badge{
      display:inline-flex; align-items:center; justify-content:center;
      width:16px; height:16px; border:1px solid #cbd5e1; border-radius:50%;
      font-size:11px; font-weight:900; color:#64748b; background:#fff;
      user-select:none; cursor:default; line-height:1;
    }
    .idp-pop{
      position:absolute; top:18px; right:0;
      background:#fff; color:#0f172a; border:1px solid #e5e7eb;
      border-radius:8px; padding:6px 8px; font-size:12px; font-weight:700;
      box-shadow:0 6px 16px rgba(0,0,0,.08); display:none; z-index:9999; white-space:nowrap;
    }
    .idp-wrap:hover .idp-pop{ display:block; }
    .idp-idbox{
      border:1px solid #e5e7eb; background:#f8fafc; border-radius:6px;
      padding:2px 6px; font-weight:800; font-size:12px; min-width:60px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      user-select:text; cursor:text;
    }
    </style>
    """
    st.markdown(BADGE_POP_CSS, unsafe_allow_html=True)

    def _js_escape(s: str) -> str:
        return str(s).replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")

    def _popover_html(id_val: str | None) -> str:
        if not id_val:
            return ""
        sid = _js_escape(_normalize_id(id_val))
        return (
            f"<span class='idp-wrap'>"
            f"  <span class='idp-badge' title='Duplo clique no ID para selecionar'>?</span>"
            f"  <span class='idp-pop'>ID:&nbsp;<input class='idp-idbox' type='text' value='{sid}' readonly></span>"
            f"</span>"
        )

    def _card_html(rank: int, agencia: str, preco: float, subtxt: str, dt: str, id_to_copy: str | None) -> str:
        stripe = "#D4AF37" if rank == 1 else ("#9CA3AF" if rank == 2 else "#CD7F32")
        stripe_div = f"<div style='position:absolute;left:0;top:0;bottom:0;width:6px;border-radius:10px 0 0 10px;background:{stripe};'></div>"
        badge = _popover_html(id_to_copy)
        return (
            f"<div style='{CARD_BASE}'>"
            f"{stripe_div}"
            f"<div style='{DT_WRAP_STYLE}'><span style='{DT_TXT_STYLE}'>{dt}</span>{badge}</div>"
            f"<div style='{RANK_STYLE}'>{rank}º</div>"
            f"<div style='{AG_STYLE}'>{agencia}</div>"
            f"<div style='{PR_STYLE}'>{fmt_moeda_br(preco)}</div>"
            f"<div style='{SUB_STYLE}'>{subtxt}</div>"
            f"</div>"
        )

    # ===================== Preparos: mapear colunas ao seu DF =====================
    dfp = df.copy()
    dfp["TRECHO_STD"] = dfp.get("TRECHO", "").astype(str)
    dfp["AGENCIA_UP"] = dfp.get("AGENCIA_NORM", "").astype(str)
    dfp["ADVP"]       = (dfp.get("ADVP_CANON").fillna(dfp.get("ADVP"))).astype(str)
    dfp["__PRECO__"]  = pd.to_numeric(dfp.get("PRECO"), errors="coerce")
    dfp["__DTKEY__"]  = pd.to_datetime(dfp.get("DATAHORA_BUSCA"), errors="coerce")

    # Coluna de ID (para badge ?)
    ID_COL = _find_id_col(dfp)

    def _normalize_id(val) -> str | None:
        if val is None or (isinstance(val, float) and np.isnan(val)): return None
        s = str(val)
        try:
            f = float(s.replace(",", "."))
            if f.is_integer(): return str(int(f))
        except Exception:
            pass
        return s

    # limpar linhas sem preço
    dfp = dfp[dfp["__PRECO__"].notna()].copy()
    if dfp.empty:
        st.info("Sem preços válidos no recorte atual."); return

    # ========= Última pesquisa por Trecho×ADVP (quando marcado) =========
    pesq_por_trecho_advp: dict[tuple[str, str], str] = {}
    PESQ_COL = ID_COL  # usa IDPESQUISA (ou similar) como “identificador” da busca
    if por_pesquisa and PESQ_COL and PESQ_COL in dfp.columns:
        tmp = dfp.dropna(subset=["TRECHO_STD", "ADVP", PESQ_COL, "__DTKEY__"]).copy()
        g = tmp.groupby(["TRECHO_STD", "ADVP", PESQ_COL], as_index=False)["__DTKEY__"].max()
        if not g.empty:
            idx = g.groupby(["TRECHO_STD", "ADVP"])["__DTKEY__"].idxmax()
            idx = idx[idx.notna()].astype(int)
            if len(idx):
                last_by_ta = g.loc[idx.values]
                pesq_por_trecho_advp = {
                    (str(r["TRECHO_STD"]), str(r["ADVP"])): str(r[PESQ_COL])
                    for _, r in last_by_ta.iterrows()
                }

    # ===================== Funções (rank, presença, dt/id) =====================
    def build_rank(df_subset: pd.DataFrame) -> pd.DataFrame:
        tmp = df_subset.copy()
        tmp["AGENCIA_UP"] = tmp["AGENCIA_UP"].astype(str)
        rank = (
            tmp.groupby("AGENCIA_UP", as_index=False)["__PRECO__"].min()
               .sort_values("__PRECO__")
               .reset_index(drop=True)
        )
        if not rank.empty:
            rank["_CAN"] = rank["AGENCIA_UP"].apply(_canon)
            rank["_BRAND"] = rank["AGENCIA_UP"].apply(_brand_tag)
        return rank

    def presence_flags(df_all_rows: pd.DataFrame, label: str) -> dict:
        sub = df_all_rows[df_all_rows["AGENCIA_UP"].astype(str).apply(_brand_tag) == _brand_tag(label)]
        present_any = not sub.empty
        present_with_price = (present_any and np.isfinite(sub["__PRECO__"]).any())
        return {"present_any": present_any, "present_with_price": present_with_price}

    def dt_and_id_for(trecho: str, advp: str, agencia: str, preco: float, pesquisa_id: str | None) -> tuple[str, str | None]:
        sub = dfp[
            (dfp["TRECHO_STD"].astype(str) == str(trecho)) &
            (dfp["ADVP"].astype(str) == str(advp)) &
            (dfp["AGENCIA_UP"].astype(str).apply(_canon) == _canon(agencia))
        ].copy()
        if pesquisa_id and PESQ_COL:
            sub = sub[sub[PESQ_COL].astype(str) == str(pesquisa_id)]
        sub_price = sub[np.isclose(sub["__PRECO__"], float(preco), rtol=0, atol=1)]
        if sub_price.empty and not sub.empty:
            min_price = sub["__PRECO__"].min()
            if np.isfinite(min_price) and np.isclose(min_price, float(preco), rtol=0, atol=1):
                sub_price = sub[np.isclose(sub["__PRECO__"], min_price, rtol=0, atol=1)]
        if sub_price.empty:
            return "", None

        ts = pd.to_datetime(sub_price["__DTKEY__"], errors="coerce").max()
        dt_lbl = ts.strftime("%d/%m %H:%M:%S") if pd.notna(ts) else ""

        id_val = None
        if ID_COL and ID_COL in sub_price.columns:
            row_latest = sub_price.loc[sub_price["__DTKEY__"].idxmax()]
            id_val = _normalize_id(row_latest[ID_COL])
        return dt_lbl, id_val

    # ===================== Render =====================
    trechos_sorted = sorted(dfp["TRECHO_STD"].dropna().astype(str).unique(), key=lambda x: str(x))
    for trecho in trechos_sorted:
        df_t = dfp[dfp["TRECHO_STD"] == trecho]
        # ADVPs ordenados por número quando existir
        advps = sorted(
            df_t["ADVP"].dropna().astype(str).unique(),
            key=lambda v: (0, int(re.search(r"\d+", str(v)).group())) if re.search(r"\d+", str(v)) else (1, str(v)),
        )

        boxes = []
        for advp in advps:
            df_ta = df_t[df_t["ADVP"].astype(str) == str(advp)].copy()
            if por_pesquisa and pesq_por_trecho_advp:
                pesq_id = pesq_por_trecho_advp.get((trecho, advp))
                all_rows = df_ta[df_ta[ID_COL].astype(str) == pesq_id] if (pesq_id is not None and ID_COL) else df_ta.iloc[0:0]
            else:
                all_rows = df_ta
                pesq_id = None

            base_rank = build_rank(all_rows)

            # filtro Agência alvo / Ranking
            if not base_rank.empty and agencia_foco != "Todos":
                rk_map = {row["AGENCIA_UP"]: i+1 for i, row in base_rank.head(3).iterrows()}
                found_target = False
                for ag_up, rank_val in rk_map.items():
                    if _canon(ag_up) == _canon(agencia_foco):
                        if posicao_foco == "Todas" or rank_val == int(posicao_foco):
                            found_target = True
                            break
                if not found_target:
                    continue

            # caixa do ADVP
            box_content = []
            box_content.append(f"<div style='{BOX_STYLE}'>")
            box_content.append(f"<div style='{HEAD_STYLE}'>ADVP: <b>{advp}</b></div>")

            if base_rank.empty:
                box_content.append(f"<div style='{NO_STYLE}'>Sem ofertas</div>")
                extras = []
                for target_label in ["123MILHAS", "MAXMILHAS"]:
                    pres = presence_flags(all_rows, target_label)
                    if not pres["present_any"]:
                        extras.append(f"<span style='{CHIP_STYLE}'>{target_label}: Não apareceu</span>")
                    elif not pres["present_with_price"]:
                        extras.append(f"<span style='{CHIP_STYLE}'>{target_label}: Sem ofertas</span>")
                if extras:
                    box_content.append(f"<div style='{EXTRAS_STYLE}'>" + "".join(extras) + "</div>")
                box_content.append("</div>")
                boxes.append("".join(box_content))
                continue

            podium = base_rank.head(3).copy()

            # cards 1/2/3
            box_content.append(f"<div style='{STACK_STYLE}'>")
            for i in range(len(podium)):
                current_rank_row = podium.iloc[i]
                preco_current = float(current_rank_row["__PRECO__"])
                ag_current = current_rank_row["AGENCIA_UP"]

                dt_current, id_to_copy = dt_and_id_for(trecho, advp, ag_current, preco_current, pesq_id)

                subtxt = "—"
                if i == 0 and len(podium) >= 2:
                    p2 = float(podium.iloc[1]["__PRECO__"])
                    if np.isfinite(p2) and p2 != 0:
                        # quanto o 1º está abaixo do 2º
                        diff_pct = (p2 - preco_current) / p2 * 100.0
                        subtxt = f"−{int(round(diff_pct))}% vs 2º"
                elif i > 0:
                    p1 = float(podium.iloc[0]["__PRECO__"])
                    if np.isfinite(p1) and p1 != 0 and np.isfinite(preco_current):
                        diff_pct = (preco_current - p1) / p1 * 100.0
                        subtxt = f"{fmt_pct_plus(diff_pct)} vs 1º"

                box_content.append(_card_html(i + 1, ag_current, preco_current, subtxt, dt_current, id_to_copy))
            box_content.append("</div>")  # stack

            # chips adicionais (presença e delta de marcas)
            extras_chips = []
            p1 = float(podium.iloc[0]["__PRECO__"])
            podium_brands_canon = {_canon(row['AGENCIA_UP']) for _, row in podium.iterrows()}

            for target_label in ["123MILHAS", "MAXMILHAS"]:
                if _canon(target_label) in podium_brands_canon:
                    continue
                pos, preco_val = None, None
                matching_rows = base_rank[base_rank["_BRAND"] == _brand_tag(target_label)]
                if not matching_rows.empty:
                    pos = matching_rows.index[0] + 1
                    preco_val = float(matching_rows.iloc[0]["__PRECO__"])

                if pos is None or not np.isfinite(preco_val):
                    pres = presence_flags(all_rows, target_label)
                    if not pres["present_any"]:
                        extras_chips.append(f"<span style='{CHIP_STYLE}'>{target_label}: Não apareceu</span>")
                    elif not pres["present_with_price"]:
                        extras_chips.append(f"<span style='{CHIP_STYLE}'>{target_label}: Sem ofertas</span>")
                else:
                    delta = ((preco_val - p1) / p1 * 100.0) if (np.isfinite(preco_val) and np.isfinite(p1) and p1 != 0) else None
                    ts_lbl, _ = dt_and_id_for(trecho, advp, target_label, preco_val, pesq_id)
                    pct_str = f" {fmt_pct_plus(delta)}" if delta is not None else ""
                    ts_part = f" | {ts_lbl}" if ts_lbl else ""
                    extras_chips.append(f"<span style='{CHIP_STYLE}'>{pos}º {target_label}: {fmt_moeda_br(preco_val)}{pct_str}{ts_part}</span>")

            if extras_chips:
                box_content.append(f"<div style='{EXTRAS_STYLE}'>" + "".join(extras_chips) + "</div>")

            box_content.append("</div>")  # box
            boxes.append("".join(box_content))

        if boxes:
            st.markdown(f"<div style='{TRE_HDR_STYLE}'>Trecho: <b>{trecho}</b></div>", unsafe_allow_html=True)
            st.markdown("<div style='" + GRID_STYLE + "'>" + "".join(boxes) + "</div>", unsafe_allow_html=True)
# ───────────────────── ABA: Top 3 Preços Mais Baratos (END) ──────────────────
