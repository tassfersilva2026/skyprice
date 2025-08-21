
# Streamlit Parquet App — Starter

App básico de Streamlit que **lê `OFERTAS.parquet`** e mostra os dados com filtros simples.
Perfeito para subir no GitHub e publicar no Streamlit Cloud.

## 🚀 Como usar (local)
1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Coloque seu arquivo **OFERTAS.parquet** na pasta `data/` (ou use o uploader no app).
3. Rode o app:
   ```bash
   streamlit run streamlit_app.py
   ```

## ☁️ Como publicar no Streamlit Cloud
1. Crie um repositório no GitHub com estes arquivos.
2. (Opcional) Faça commit também de `data/OFERTAS.parquet` — caso contrário, carregue pelo uploader na interface.
3. Na plataforma do Streamlit, aponte para o arquivo principal: `streamlit_app.py`.

## 📁 Estrutura
```
.
├─ .streamlit/
│  └─ config.toml
├─ data/
│  └─ (coloque aqui o OFERTAS.parquet)
├─ .gitignore
├─ requirements.txt
├─ streamlit_app.py
└─ README.md
```

## 💡 Observações
- O app tenta automaticamente carregar `data/OFERTAS.parquet`. Se não encontrar, aparece um **file uploader**.
- Cache habilitado com `st.cache_data()` para leituras rápidas.
- Detecta colunas com baixa cardinalidade para filtros rápidos (até 50 valores).
