"""
Module 3a: RAG Index Builder
Chunks ICMR STW text files, embeds them, and builds a FAISS + BM25 hybrid
retrieval index. Run this once (and again whenever medical_guidelines_text/
changes) to produce the artifacts that rag.py loads at query time.
"""

import re
import json
import pickle
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ---- Config ----
TEXT_DIR = Path("medical_guidelines_text")
INDEX_DIR = Path("models/rag_index")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # general-purpose, fast; swap for a
                                        # biomedical sentence embedder later if desired
CHUNK_MAX_CHARS = 800
CHUNK_OVERLAP_CHARS = 120

KNOWN_DOMAINS = [
    "cardiology", "hypertension", "ent", "nephrology",
    "neurology", "orthopaedics", "orthopedics", "psychiatry",
]


def infer_domain(file_path: Path) -> str:
    """
    Infers domain from the parent folder name if the corpus is organized
    into domain subfolders (medical_guidelines_text/cardiology/...).
    Falls back to matching a known domain token inside the filename.
    """
    parent = file_path.parent.name.lower()
    if parent in KNOWN_DOMAINS:
        return "orthopaedics" if parent == "orthopedics" else parent

    fname = file_path.stem.lower()
    for domain in KNOWN_DOMAINS:
        if domain in fname:
            return "orthopaedics" if domain == "orthopedics" else domain

    return "general"


def chunk_text(text: str, max_chars=CHUNK_MAX_CHARS, overlap=CHUNK_OVERLAP_CHARS):
    """
    Paragraph-aware chunking. Oversized paragraphs are split on sentence
    boundaries (falling back to whitespace) instead of raw character
    slicing, so chunks never cut off mid-word/mid-sentence.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    def split_long_paragraph(para):
        # Prefer splitting on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", para)
        pieces, current = [], ""
        for sent in sentences:
            if len(current) + len(sent) + 1 <= max_chars:
                current = f"{current} {sent}".strip()
            else:
                if current:
                    pieces.append(current)
                if len(sent) <= max_chars:
                    current = sent
                else:
                    # Single sentence still too long -> split on whitespace only
                    words = sent.split()
                    current = ""
                    for w in words:
                        if len(current) + len(w) + 1 <= max_chars:
                            current = f"{current} {w}".strip()
                        else:
                            pieces.append(current)
                            current = w
        if current:
            pieces.append(current)
        return pieces

    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= max_chars:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            if len(para) <= max_chars:
                current = para
            else:
                chunks.extend(split_long_paragraph(para))
                current = ""

    if current:
        chunks.append(current)

    return chunks


def build_index():
    if not TEXT_DIR.exists():
        raise FileNotFoundError(f"'{TEXT_DIR}' not found. Run extract_pdfs.py first.")

    txt_files = list(TEXT_DIR.rglob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found under '{TEXT_DIR}'.")

    print(f"[*] Found {len(txt_files)} source files under '{TEXT_DIR}'.")

    records = []  # one dict per chunk: text, domain, source_file, chunk_id

    for file_path in txt_files:
        raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
        domain = infer_domain(file_path)
        chunks = chunk_text(raw_text)

        for i, chunk in enumerate(chunks):
            records.append({
                "text": chunk,
                "domain": domain,
                "source_file": file_path.name,
                "chunk_id": f"{file_path.stem}_{i}",
            })

    print(f"[*] Produced {len(records)} chunks across {len(txt_files)} files.")

    print(f"[*] Loading embedding model: {EMBED_MODEL_NAME}...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    texts = [r["text"] for r in records]
    print("[*] Encoding chunks (dense embeddings for FAISS)...")
    embeddings = embedder.encode(
        texts, show_progress_bar=True, normalize_embeddings=True
    ).astype("float32")

    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine sim
    faiss_index.add(embeddings)

    print("[*] Building BM25 sparse index...")
    tokenized_corpus = [t.lower().split() for t in texts]
    bm25_index = BM25Okapi(tokenized_corpus)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(faiss_index, str(INDEX_DIR / "faiss.index"))

    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(bm25_index, f)

    with open(INDEX_DIR / "records.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    with open(INDEX_DIR / "config.json", "w") as f:
        json.dump({"embed_model": EMBED_MODEL_NAME}, f)

    print(f"[+] Index built and saved to '{INDEX_DIR}/'.")
    print(f"    - faiss.index   ({faiss_index.ntotal} vectors)")
    print(f"    - bm25.pkl      ({len(tokenized_corpus)} docs)")
    print(f"    - records.json  (chunk text + metadata)")


if __name__ == "__main__":
    build_index()