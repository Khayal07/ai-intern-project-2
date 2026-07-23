"""Streamlit demo — "Sənədlərinlə Danış" RAG interfeysi.

Brauzerdə işləyən sadə sual-cavab səhifəsi. Bütün RAG məntiqi mövcud modullardan
(`ingestion`, `vectorstore`, `rag`) gəlir — bu fayl yalnız UI qatıdır.

İşə salmaq:  streamlit run app.py
"""

import streamlit as st

from src.config import DATA_DIR
from src.ingestion import load_and_chunk
from src.rag import answer_question, format_sources
from src.vectorstore import build_vectorstore


# @st.cache_resource — ağır, təkrar istifadə olunan obyektləri (məs. vektor bazası)
# bir dəfə qurub yaddaşda saxlayır. Belədə hər sualda embedding yenidən hesablanmır.
@st.cache_resource(show_spinner="Sənədlər yüklənir və vektor bazası qurulur...")
def get_vectorstore():
    chunks = load_and_chunk()
    store = build_vectorstore(chunks)
    return store, len(chunks)


st.set_page_config(page_title="Sənədlərinlə Danış", page_icon="📄")

st.title("📄 Sənədlərinlə Danış")
st.caption("Sənədlər haqqında sual ver — cavab yalnız sənəd məzmununa əsaslanır (RAG).")

store, chunk_count = get_vectorstore()

# Yan panel — hansı sənədlərin yükləndiyi barədə məlumat.
with st.sidebar:
    st.header("Məlumat")
    st.write(f"**Qovluq:** `{DATA_DIR.name}/`")
    st.write(f"**Chunk sayı:** {chunk_count}")
    st.write("**Model:** gpt-4o-mini")
    st.write("**Vektor DB:** Chroma")

query = st.text_input(
    "Sualınız:",
    placeholder="Məs: How many days do I have to request a refund?",
)

if st.button("Soruş", type="primary") and query.strip():
    with st.spinner("Cavab hazırlanır..."):
        answer, results = answer_question(store, query)

    st.subheader("Cavab")
    st.write(answer)

    # Mənbələri açılan bölmədə göstəririk (traceability — Checkpoint 5).
    with st.expander("Mənbələr (istifadə olunan chunk-lar)"):
        st.text(format_sources(results))
