from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "apple_chunks.json"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "apple_embeddings.npy"


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# CHECK INPUT
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Chunk file not found: {INPUT_FILE}\n"
        "Run chunk_filing.py first."
    )


# ============================================================
# LOAD CHUNKS
# ============================================================

print("Loading chunks...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("Total chunks:", len(chunks))


if not chunks:
    raise ValueError("No chunks found.")


# ============================================================
# EXTRACT TEXT
# ============================================================

texts = [
    chunk["text"]
    for chunk in chunks
]


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print(f"Loading embedding model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("Generating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)


# ============================================================
# DISPLAY SHAPE
# ============================================================

print("Embedding shape:", embeddings.shape)


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

np.save(OUTPUT_FILE, embeddings)

print("Saved embeddings to:", OUTPUT_FILE)