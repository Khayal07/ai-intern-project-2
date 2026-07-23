"""Checkpoint 2 — Chunk-lar üçün embedding generasiyası.

Embedding = mətni ədədlərdən ibarət vektora çevirmək. Mənaca oxşar mətnlərin
vektorları vektor fəzasında bir-birinə yaxın olur. Bu, sonrakı checkpoint-də
oxşarlıq axtarışının (similarity search) əsasını təşkil edir: sualı da vektora
çevirib, ona ən yaxın chunk vektorlarını tapırıq.

Burada OpenAI-nin `text-embedding-3-small` modelindən istifadə edirik.
Model hər mətn parçasını sabit ölçülü (1536) vektora çevirir.
"""

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.config import EMBEDDING_MODEL, OPENAI_API_KEY


def get_embeddings_model() -> OpenAIEmbeddings:
    """OpenAI embedding modelini hazırlayıb qaytarır.

    Açar yoxdursa, aydın xəta veririk — belədə istifadəçi problemi dərhal başa düşür.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY tapılmadı. '.env.example'-i '.env' kimi kopyalayıb "
            "açarı doldurun."
        )
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)


def embed_chunks(
    chunks: list[Document],
    model: OpenAIEmbeddings | None = None,
) -> list[list[float]]:
    """Verilmiş chunk-ların hər biri üçün embedding vektoru generasiya edir.

    `embed_documents` bütün mətnləri bir sorğuda API-yə göndərir (batch) —
    bu, hər chunk üçün ayrıca sorğudan daha sürətli və ucuzdur.
    Nəticə: hər chunk üçün bir vektor (list[float]).
    """
    model = model or get_embeddings_model()
    texts = [chunk.page_content for chunk in chunks]
    return model.embed_documents(texts)
