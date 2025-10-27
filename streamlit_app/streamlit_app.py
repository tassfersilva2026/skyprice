import streamlit as st
import os

# Ler o arquivo de manutenção
caminho_manutencao = "data/MANUTENÇÃO.txt"

if os.path.exists(caminho_manutencao):
    with open(caminho_manutencao, "r", encoding="utf-8") as file:
        conteudo_manutencao = file.read()
    
    # Exibir no app
    st.header("📋 Manutenção")
    st.text(conteudo_manutencao)
else:
    st.warning(f"Arquivo não encontrado: {caminho_manutencao}")
