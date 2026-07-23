# RAG — "Sənədlərinlə Danış"

İstifadəçinin sənədləri (PDF/mətn) haqqında sual verib, **yalnız həmin sənədlərə əsaslanan**
cavab ala biləcəyi RAG (Retrieval-Augmented Generation) pipeline-ı.

Pipeline: `sənəd → chunking → embedding → vektor bazası → retrieval → prompt → LLM cavabı`.

## Texnologiyalar

- **Python**
- **LangChain** — RAG komponentləri (loader, splitter, retriever)
- **OpenAI** — cavab generasiyası (`gpt-4o-mini`) və embedding (`text-embedding-3-small`)
- **Chroma** — vektor bazası (persistent)

## Quraşdırma

```bash
# 1. Asılılıqları quraşdır
pip install -r requirements.txt

# 2. API açarını hazırla
cp .env.example .env
# .env faylını açıb OPENAI_API_KEY dəyərini yaz
```

`.env` faylı `.gitignore`-dadır — real API açarı heç vaxt git-ə düşmür.

## İşə salmaq

```bash
python main.py
```

## Layihənin quruluşu

```
├── data/                    # mənbə sənədlər (məs. nimbus_handbook.txt)
├── src/
│   ├── config.py            # sabitlər (chunk ölçüsü, overlap, qovluqlar)
│   └── ingestion.py         # sənəd yükləmə + chunking
├── main.py                  # CLI giriş nöqtəsi
├── requirements.txt
└── .env.example
```

## Checkpoint-lərin gedişatı

### ✅ Checkpoint 1 — Sənəd ingestion + chunking

- `data/` qovluğundakı sənədlər yüklənir (`TextLoader` / `PyPDFLoader`).
- Mətn `RecursiveCharacterTextSplitter` ilə **overlap-lı** chunk-lara bölünür
  (`CHUNK_SIZE=500`, `CHUNK_OVERLAP=100`).
- **Overlap niyə vacibdir?** Bir fakt iki chunk sərhədinə düşəndə, overlap sayəsində
  hər iki chunk faktın bir hissəsini əhatə edir və məlumat itmir. Overlap olmayan
  sadə (fixed-size) bölmə isə belə faktları yarımçıq kəsə bilər.
- `RecursiveCharacterTextSplitter` mətni əvvəlcə böyük məntiqi sərhədlərdən
  (paraqraf → sətir → cümlə → söz) bölür, ona görə chunk-lar məna baxımından bütöv qalır.

**Nümunə çıxış:**

```
=== Checkpoint 1: Sənəd ingestion + chunking ===

Yüklənən sənəd (səhifə) sayı: 1
  - mənbə: .../data/nimbus_handbook.txt | uzunluq: 1939 simvol

Yaradılan chunk sayı: 5

--- İlk 3 chunk (nümunə) ---
[chunk 0] (319 simvol)
Nimbus Cloud Storage — Customer Handbook ...
```
