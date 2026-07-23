"""Checkpoint 4 — Prompt qurulması (retrieved chunk-ların prompt-a inteqrasiyası).

RAG-da prompt-un quruluşu keyfiyyətə birbaşa təsir edir. Burada üç hissəni
**aydın şəkildə ayırırıq**:

  1) TƏLİMAT (system) — modelə "yalnız kontekstə əsaslan" deyir.
  2) KONTEKST      — retrieval-dən gələn nömrələnmiş chunk blokları.
  3) SUAL          — istifadəçinin sualı.

Niyə ayrım vacibdir?
  Model kontekstin harada bitib, sualın harada başladığını dəqiq başa düşməlidir.
  Aydın ayrım modelin öz "biliyindən" cavab uydurmasının qarşısını almağa kömək edir
  (bu, Checkpoint 6-nın — hallüsinasiya idarəsinin — təməlidir).
"""

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import NOT_FOUND_MESSAGE

# Model üçün əsas təlimat. Qısa, aydın və qəti olmalıdır.
# Checkpoint 6: kontekstdə cavab yoxdursa, model uydurmamalı, dəqiq NOT_FOUND_MESSAGE-i
# qaytarmalıdır. Bu, RAG-ın ən tanınmış uğursuzluğunun (hallüsinasiya) qarşısını alır.
SYSTEM_INSTRUCTION = (
    "Sən sənədlərə əsaslanan sual-cavab köməkçisisən. "
    "Cavabı YALNIZ aşağıda verilən KONTEKST əsasında ver. "
    "Kontekstdən kənar öz biliyindən istifadə etmə. "
    "Əgər cavab KONTEKST-də yoxdursa, heç nə uydurma və dəqiq bu cümləni qaytar: "
    f"\"{NOT_FOUND_MESSAGE}\""
)


def format_context(results: list[tuple[Document, float]]) -> str:
    """Retrieval nəticələrini nömrələnmiş kontekst bloklarına çevirir.

    Hər blok ayrıca nömrələnir ki, həm model, həm də biz (debug üçün) hansı
    chunk-ın istifadə olunduğunu izləyə bilək.
    """
    blocks = []
    for position, (document, _score) in enumerate(results, start=1):
        blocks.append(f"[Kontekst {position}]\n{document.page_content}")
    return "\n\n".join(blocks)


def build_prompt(
    query: str,
    results: list[tuple[Document, float]],
) -> list:
    """Təlimat + kontekst + sualdan ibarət chat mesajları siyahısı qurur.

    `SystemMessage` təlimatı, `HumanMessage` isə kontekst və sualı daşıyır.
    Bu ayrım LangChain-in chat modelləri üçün standart yanaşmasıdır.
    """
    context = format_context(results)
    human_content = f"KONTEKST:\n{context}\n\nSUAL: {query}"

    return [
        SystemMessage(content=SYSTEM_INSTRUCTION),
        HumanMessage(content=human_content),
    ]
