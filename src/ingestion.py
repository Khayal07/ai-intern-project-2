"""Checkpoint 1 — Sənəd ingestion + chunking.

Bu modul iki işi görür:
  1) `load_documents`  — `data/` qovluğundakı sənədləri (PDF, mətn) yükləyir.
  2) `chunk_documents` — sənədləri overlap-lı, məntiqli parçalara (chunk) bölür.

Niyə chunking?
  LLM-in kontekst pəncərəsi məhduddur və embedding modelləri qısa mətnlərdə daha
  dəqiq işləyir. Böyük sənədi bütöv göndərmək əvəzinə onu kiçik parçalara bölürük,
  sonra yalnız suala uyğun parçaları LLM-ə veririk (retrieval).
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR

# Hansı fayl tiplərini oxuya bilirik.
SUPPORTED_TEXT_SUFFIXES = {".txt", ".md"}
SUPPORTED_PDF_SUFFIXES = {".pdf"}


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """`data_dir` qovluğundakı bütün dəstəklənən sənədləri yükləyir.

    Hər sənəd LangChain `Document` obyektinə çevrilir: `.page_content` (mətn) və
    `.metadata` (mənbə fayl, səhifə və s.). Metadata sonrakı checkpoint-lərdə
    mənbə istinadı (Checkpoint 5) üçün lazım olacaq.
    """
    documents: list[Document] = []

    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix in SUPPORTED_PDF_SUFFIXES:
            loader = PyPDFLoader(str(path))
        elif suffix in SUPPORTED_TEXT_SUFFIXES:
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            # Dəstəklənməyən faylları sakitcə ötürürük.
            continue

        documents.extend(loader.load())

    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Sənədləri overlap-lı chunk-lara bölür.

    `RecursiveCharacterTextSplitter` mətni əvvəlcə böyük məntiqi sərhədlərdən
    (paraqraf -> cümlə -> söz) bölməyə çalışır. Bu, mətni təsadüfi yerdən
    kəsməkdən daha yaxşıdır, çünki chunk-lar məna baxımından bütöv qalır.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Bölmə üçün prioritet ayırıcılar: əvvəl paraqraf, sonra sətir, cümlə, söz.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Hər chunk-a nömrə əlavə edirik — sonrakı checkpoint-lərdə istinad üçün faydalıdır.
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    return chunks


def load_and_chunk(data_dir: Path = DATA_DIR) -> list[Document]:
    """Rahatlıq üçün: yüklə + böl addımlarını bir çağırışda edir."""
    documents = load_documents(data_dir)
    return chunk_documents(documents)
