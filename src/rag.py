"""Checkpoint 4 — RAG axını: retrieval → prompt → LLM cavabı.

Bu modul bütün hissələri birləşdirir:
  1) Sual üçün ən oxşar chunk-ları tapır (Checkpoint 3).
  2) Onları strukturlaşdırılmış prompt-a yığır (Checkpoint 4).
  3) LLM-i çağırıb cavab alır.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src.config import LLM_MODEL, OPENAI_API_KEY, TOP_K
from src.prompt import build_prompt
from src.vectorstore import similarity_search


def get_llm() -> ChatOpenAI:
    """Cavab generasiyası üçün LLM-i hazırlayır.

    `temperature=0` — cavabların determinist və faktlara sadiq olması üçün.
    RAG-da yaradıcılıq yox, dəqiqlik istəyirik.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY tapılmadı. '.env.example'-i '.env' kimi kopyalayıb "
            "açarı doldurun."
        )
    return ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY, temperature=0)


def answer_question(
    store: Chroma,
    query: str,
    k: int = TOP_K,
    llm: ChatOpenAI | None = None,
) -> tuple[str, list[tuple[Document, float]]]:
    """Suala kontekst əsasında cavab qaytarır.

    Nəticə: (cavab mətni, istifadə olunan chunk-lar). Chunk-ları da qaytarırıq
    ki, Checkpoint 5-də mənbə istinadını göstərə bilək.
    """
    results = similarity_search(store, query, k=k)
    messages = build_prompt(query, results)

    llm = llm or get_llm()
    response = llm.invoke(messages)

    return response.content, results
