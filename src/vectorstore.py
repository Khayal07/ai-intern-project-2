"""Checkpoint 3 — Vektor saxlama + oxşarlıq axtarışı (Chroma).

Bu modul embedding-ləri **Chroma** vektor bazasında saxlayır və verilən sual üçün
ən oxşar chunk-ları tapır (retrieval).

İş prinsipi:
  1) Hər chunk embedding modeli ilə vektora çevrilir və metadata ilə birlikdə saxlanılır.
  2) Sual da eyni modellə vektora çevrilir.
  3) Chroma vektor fəzasında suala ən yaxın chunk vektorlarını tapıb qaytarır.

Niyə vektor bazası (adi siyahı yox)?
  Böyük sənəd toplusunda milyonlarla chunk ola bilər. Vektor bazası oxşarlıq axtarışını
  effektiv (indeksləşdirilmiş) şəkildə edir və nəticələri diskdə saxlayır — hər dəfə
  yenidən embedding hesablamağa ehtiyac qalmır.
"""

import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import CHROMA_DIR, COLLECTION_NAME, TOP_K
from src.embeddings import get_embeddings_model


def build_vectorstore(chunks: list[Document], reset: bool = True) -> Chroma:
    """Chunk-lardan persistent Chroma kolleksiyası qurur.

    `reset=True` olduqda köhnə kolleksiyanı silirik — belədə eyni proqramı bir neçə
    dəfə işə saldıqda eyni chunk-lar təkrar-təkrar əlavə olunub dublikat yaratmır.

    Chroma `from_documents` daxildə hər chunk-ı embedding modeli ilə vektora çevirir
    və mətn + metadata ilə birlikdə saxlayır.
    """
    if reset and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings_model(),
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )


def load_vectorstore() -> Chroma:
    """Diskdə artıq mövcud olan Chroma kolleksiyasını yükləyir (yenidən qurmadan)."""
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings_model(),
    )


def similarity_search(
    store: Chroma,
    query: str,
    k: int = TOP_K,
) -> list[tuple[Document, float]]:
    """Suala ən yaxın `k` chunk-ı oxşarlıq balı ilə birlikdə qaytarır.

    Nəticə: (Document, distance) cütlərinin siyahısı. Chroma-da bu bal **məsafədir** —
    yəni dəyər NƏ QƏDƏR KİÇİKDİRSƏ, chunk suala bir o qədər YAXINDIR (oxşardır).
    """
    return store.similarity_search_with_score(query, k=k)
