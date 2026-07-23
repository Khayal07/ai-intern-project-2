"""Layihə boyu istifadə olunan sabitlər və qovluq yolları.

Bütün konfiqurasiyanı bir yerdə saxlamaq üçün ayrıca modul yaratdıq —
belədə chunk ölçüsü kimi dəyərləri dəyişmək üçün bir neçə fayla baxmaq lazım gəlmir.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Layihənin kök qovluğu (bu fayldan iki səviyyə yuxarı: src/ -> layihə kökü).
BASE_DIR = Path(__file__).resolve().parent.parent

# .env faylını yükləyirik ki, OPENAI_API_KEY mühit dəyişəni kimi əlçatan olsun.
load_dotenv(BASE_DIR / ".env")

# Mənbə sənədlərinin saxlanıldığı qovluq.
DATA_DIR = BASE_DIR / "data"

# --- OpenAI parametrləri ---
# API açarı koda YAZILMIR — yalnız .env faylından oxunur (təhlükəsizlik).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Embedding modeli: mətni ~1536 ölçülü vektora çevirir. Ucuz və keyfiyyətlidir.
EMBEDDING_MODEL = "text-embedding-3-small"

# Cavab generasiyası üçün LLM modeli.
LLM_MODEL = "gpt-4o-mini"

# --- Vektor bazası (Chroma) parametrləri ---
# Chroma verilənləri bu qovluqda diskdə saxlayır (persistent) — proqram bağlansa da qalır.
CHROMA_DIR = BASE_DIR / "chroma_db"
# Kolleksiya = vektorların saxlanıldığı "cədvəl"in adı.
COLLECTION_NAME = "nimbus_docs"

# Oxşarlıq axtarışında qaytarılacaq ən yaxın chunk sayı (top-k).
TOP_K = 3

# --- Chunking parametrləri ---
# CHUNK_SIZE: hər chunk-ın təxmini maksimal simvol sayı.
# CHUNK_OVERLAP: qonşu chunk-ların bir-biri ilə üst-üstə düşən hissəsi.
#   Overlap sayəsində bir cümlə/fakt iki chunk sərhədinə düşəndə tam itmir —
#   hər iki chunk faktın bir hissəsini əhatə edir.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
