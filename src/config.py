"""Layihə boyu istifadə olunan sabitlər və qovluq yolları.

Bütün konfiqurasiyanı bir yerdə saxlamaq üçün ayrıca modul yaratdıq —
belədə chunk ölçüsü kimi dəyərləri dəyişmək üçün bir neçə fayla baxmaq lazım gəlmir.
"""

from pathlib import Path

# Layihənin kök qovluğu (bu fayldan iki səviyyə yuxarı: src/ -> layihə kökü).
BASE_DIR = Path(__file__).resolve().parent.parent

# Mənbə sənədlərinin saxlanıldığı qovluq.
DATA_DIR = BASE_DIR / "data"

# --- Chunking parametrləri ---
# CHUNK_SIZE: hər chunk-ın təxmini maksimal simvol sayı.
# CHUNK_OVERLAP: qonşu chunk-ların bir-biri ilə üst-üstə düşən hissəsi.
#   Overlap sayəsində bir cümlə/fakt iki chunk sərhədinə düşəndə tam itmir —
#   hər iki chunk faktın bir hissəsini əhatə edir.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
