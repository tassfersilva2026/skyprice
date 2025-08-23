# ... (all other code remains the same)

def tab1_painel(df_raw: pd.DataFrame):
    df = render_filters(df_raw, key_prefix="t1") # FILTROS NO TOPO
    st.subheader("Painel de Performance")
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
    st.markdown("---",)

    W = winners_by_position(df)

    # Inject custom CSS for cards
    st.markdown("""
        <style>
            .stCard {
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .gold { color: #FFD700; }
            .silver { color: #C0C0C0; }
            .bronze { color: #CD7F32; }
            .stMetricValue { font-size: 2.5rem; }
        </style>
    """, unsafe_allow_html=True)


    def render_agency_card(title: str, base: pd.DataFrame, target: str):
        """Função para renderizar o card de uma agência."""
        if base.empty:
            p1, p2, p3 = 0.0, 0.0, 0.0
        else:
            p1 = (base["R1"] == target).mean() * 100
            p2 = (base["R2"] == target).mean() * 100
            p3 = (base["R3"] == target).mean() * 100
        
        p1, p2, p3 = np.nan_to_num(p1), np.nan_to_num(p2), np.nan_to_num(p3)

        with st.container():
            st.markdown(f"**{title}**")
            st.markdown(f"<div class='stCard'>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"<p class='gold'><b>1º lugar</b></p>", unsafe_allow_html=True)
                st.markdown(f"<h3>{p1:.2f}%</h3>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<p class='silver'><b>2º lugar</b></p>", unsafe_allow_html=True)
                st.markdown(f"<h3>{p2:.2f}%</h3>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<p class='bronze'><b>3º lugar</b></p>", unsafe_allow_html=True)
                st.markdown(f"<h3>{p3:.2f}%</h3>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


    # 1. Apresentar GRUPO 123 (Visão Consolidada)
    Wg = W.replace({
        "R1": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
        "R2": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"},
        "R3": {"MAXMILHAS": "GRUPO 123", "123MILHAS": "GRUPO 123"}
    })
    render_agency_card("GRUPO 123", Wg, "GRUPO 123")
    st.markdown("---")

    # 2. Apresentar TODAS as agências individualmente
    if not W.empty:
        top_agencies = W["R1"].value_counts().index.tolist()
        agencies_to_show = [agency for agency in top_agencies if agency != "SEM OFERTAS"]
        
        # Use st.columns to create a grid of cards
        for i in range(0, len(agencies_to_show), 4):
            cols = st.columns(4)
            for j, agency in enumerate(agencies_to_show[i:i+4]):
                with cols[j]:
                    render_agency_card(agency, W, agency)
    st.markdown("---")

    # 3. Apresentar o bloco de "% SEM OFERTAS"
    render_agency_card("SEM OFERTAS", W, "SEM OFERTAS")

# ... (all other code remains the same)
