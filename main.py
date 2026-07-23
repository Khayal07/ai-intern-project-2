"""RAG — "Sənədlərinlə Danış" — CLI giriş nöqtəsi.

Checkpoint 1: sənədləri yükləyib chunk-lara bölür və nəticəni çap edir.
Növbəti checkpoint-lərdə bu fayl embedding, saxlama və sual-cavab addımları ilə genişlənəcək.

İşə salmaq:  python main.py
"""

import sys

from src.embeddings import embed_chunks
from src.ingestion import chunk_documents, load_documents
from src.rag import answer_question, format_sources
from src.vectorstore import build_vectorstore, similarity_search

# Windows-un standart konsolu (cp1252) Azərbaycan hərflərini çap edə bilmir.
# Çıxışı UTF-8-ə keçirərək bunun qarşısını alırıq.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    print("=== Checkpoint 1: Sənəd ingestion + chunking ===\n")

    documents = load_documents()
    print(f"Yüklənən sənəd (səhifə) sayı: {len(documents)}")
    for doc in documents:
        source = doc.metadata.get("source", "?")
        print(f"  - mənbə: {source} | uzunluq: {len(doc.page_content)} simvol")

    chunks = chunk_documents(documents)
    print(f"\nYaradılan chunk sayı: {len(chunks)}")

    # İlk üç chunk-ı nümunə kimi göstəririk ki, overlap və məzmun görünsün.
    print("\n--- İlk 3 chunk (nümunə) ---")
    for chunk in chunks[:3]:
        index = chunk.metadata.get("chunk_index")
        preview = chunk.page_content.replace("\n", " ")
        print(f"\n[chunk {index}] ({len(chunk.page_content)} simvol)")
        print(preview)

    # --- Checkpoint 2: embedding generasiyası ---
    print("\n\n=== Checkpoint 2: Embedding generasiyası ===\n")
    vectors = embed_chunks(chunks)
    print(f"Generasiya olunan vektor sayı: {len(vectors)} (hər chunk üçün bir vektor)")
    print(f"Hər vektorun ölçüsü (dimension): {len(vectors[0])}")
    print("\nİlk chunk-ın vektorunun ilk 8 dəyəri (nümunə):")
    print([round(value, 4) for value in vectors[0][:8]])

    # --- Checkpoint 3: vektor saxlama + oxşarlıq axtarışı ---
    print("\n\n=== Checkpoint 3: Vektor saxlama + oxşarlıq axtarışı ===\n")
    store = build_vectorstore(chunks)
    print(f"Chroma kolleksiyasına saxlanan chunk sayı: {len(chunks)}")

    query = "How many days do I have to request a refund?"
    print(f"\nSual: {query}")
    results = similarity_search(store, query)

    print(f"\nƏn oxşar {len(results)} chunk (bal kiçikdirsə = daha oxşar):")
    for document, score in results:
        index = document.metadata.get("chunk_index")
        preview = document.page_content.replace("\n", " ")[:120]
        print(f"\n[chunk {index}] məsafə={score:.4f}")
        print(f"  {preview}...")

    # --- Checkpoint 4: retrieval + prompt + LLM cavabı ---
    print("\n\n=== Checkpoint 4: Retrieval + prompt qurulması ===\n")
    answer, used_chunks = answer_question(store, query)
    print(f"Sual: {query}")
    print(f"\nCavab:\n{answer}")

    # --- Checkpoint 5: mənbə istinadı ---
    print("\n\n=== Checkpoint 5: Mənbə istinadı ilə cavab ===\n")
    print(f"Sual: {query}")
    print(f"\nCavab:\n{answer}")
    print(f"\nMənbələr:\n{format_sources(used_chunks)}")

    # --- Checkpoint 6: "sənədlərdə yoxdur" halının idarəsi ---
    print("\n\n=== Checkpoint 6: 'Sənədlərdə yoxdur' halının idarəsi ===\n")
    # Bu sual mövzuya yaxındır (məsafə kiçik), amma cavab sənəddə YOXDUR —
    # sistem uydurmamalı, dürüst şəkildə bilmədiyini deməlidir.
    missing_query = "Does Nimbus offer a mobile app for iPhone?"
    missing_answer, _ = answer_question(store, missing_query)
    print(f"Sual: {missing_query}")
    print(f"\nCavab:\n{missing_answer}")


if __name__ == "__main__":
    main()
