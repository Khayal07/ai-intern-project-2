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
│   ├── config.py            # sabitlər (chunk ölçüsü, overlap, model, açar)
│   ├── ingestion.py         # sənəd yükləmə + chunking
│   ├── embeddings.py        # chunk-lar üçün embedding generasiyası
│   ├── vectorstore.py       # Chroma saxlama + oxşarlıq axtarışı
│   ├── prompt.py            # prompt qurulması (kontekst/təlimat ayrımı)
│   └── rag.py               # retrieval → prompt → LLM axını
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

### ✅ Checkpoint 2 — Chunk-lar üçün embedding generasiyası

- Hər chunk `OpenAIEmbeddings` (`text-embedding-3-small`) ilə **1536 ölçülü vektora** çevrilir.
- **Embedding nədir?** Mətnin mənasını ədədlər massivi kimi ifadə etməkdir. Mənaca oxşar
  mətnlərin vektorları vektor fəzasında bir-birinə yaxın olur.
- **Niyə lazımdır?** Sonrakı oxşarlıq axtarışı üçün: sualı da vektora çevirib, ona ən yaxın
  chunk vektorlarını tapırıq.
- `embed_documents` bütün chunk-ları **bir sorğuda** (batch) göndərir — sürətli və ucuz.

**Nümunə çıxış:**

```
=== Checkpoint 2: Embedding generasiyası ===

Generasiya olunan vektor sayı: 5 (hər chunk üçün bir vektor)
Hər vektorun ölçüsü (dimension): 1536

İlk chunk-ın vektorunun ilk 8 dəyəri (nümunə):
[0.0031, 0.0374, 0.0208, -0.0159, 0.0098, -0.0427, -0.013, 0.0222]
```

### ✅ Checkpoint 3 — Vektor saxlama + oxşarlıq axtarışı

- Chunk-lar embedding-ləri və metadata-sı ilə birlikdə **Chroma** vektor bazasında
  (diskdə, `chroma_db/` qovluğunda) saxlanılır.
- `similarity_search` sualı vektora çevirib, vektor fəzasında ən yaxın chunk-ları tapır.
- Nəticə **məsafə (distance)** balı ilə qayıdır: **bal nə qədər kiçikdirsə, chunk suala
  bir o qədər oxşardır**.
- Metadata (mənbə fayl, `chunk_index`) saxlanılır — Checkpoint 5-də mənbə istinadı üçün lazımdır.

**Nümunə çıxış:**

```
=== Checkpoint 3: Vektor saxlama + oxşarlıq axtarışı ===

Chroma kolleksiyasına saxlanan chunk sayı: 5

Sual: How many days do I have to request a refund?

Ən oxşar 3 chunk (bal kiçikdirsə = daha oxşar):
[chunk 2] məsafə=1.0377
  Refund Policy Nimbus wants customers to feel confident when they upgrade...
[chunk 3] məsafə=1.3944
  File Recovery Every paid plan keeps deleted files...
```

> Sual "refund" haqqındadır və sistem düzgün olaraq **Refund Policy** chunk-ını (chunk 2)
> ən yaxın nəticə kimi qaytardı.

### ✅ Checkpoint 4 — Retrieval + prompt qurulması

- Tapılan chunk-lar **aydın kontekst/təlimat ayrımı** ilə prompt-a yığılır:
  `SystemMessage` (təlimat) + `HumanMessage` (nömrələnmiş KONTEKST blokları + SUAL).
- LLM (`gpt-4o-mini`, `temperature=0`) yalnız bu kontekstə əsaslanaraq cavab verir.
- `temperature=0` — determinist, faktlara sadiq cavablar üçün (RAG-da yaradıcılıq yox, dəqiqlik istəyirik).

**Nümunə çıxış:**

```
=== Checkpoint 4: Retrieval + prompt qurulması ===

Sual: How many days do I have to request a refund?

Cavab:
You have forty-five calendar days to request a refund from the date of the original purchase.
```

> Cavab tamamilə sənəddəki mətnə əsaslanır — model heç nə uydurmur.

### ✅ Checkpoint 5 — Mənbə istinadı ilə cavab

- Hər cavabın altında istifadə olunan chunk-ların **mənbəsi** göstərilir:
  `fayl adı + chunk indeksi + oxşarlıq məsafəsi`.
- İstinadlar **retrieval metadata-sından** götürülür, LLM-dən yox — belədə istinad
  100% dəqiqdir (LLM istinadı da uydura bilər).

**Nümunə çıxış:**

```
=== Checkpoint 5: Mənbə istinadı ilə cavab ===

Sual: How many days do I have to request a refund?

Cavab:
You have forty-five calendar days to request a refund from the date of the original purchase.

Mənbələr:
- nimbus_handbook.txt (chunk 2, məsafə=1.0377)
- nimbus_handbook.txt (chunk 3, məsafə=1.3942)
- nimbus_handbook.txt (chunk 4, məsafə=1.6703)
```
