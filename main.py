"""RAG — "Sənədlərinlə Danış" — CLI giriş nöqtəsi.

Checkpoint 1: sənədləri yükləyib chunk-lara bölür və nəticəni çap edir.
Növbəti checkpoint-lərdə bu fayl embedding, saxlama və sual-cavab addımları ilə genişlənəcək.

İşə salmaq:  python main.py
"""

import sys

from src.ingestion import chunk_documents, load_documents

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


if __name__ == "__main__":
    main()
