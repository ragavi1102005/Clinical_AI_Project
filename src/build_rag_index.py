"""
Module 3a: RAG index builder (offline, run once whenever medical_guidelines_text/ changes).

Design: "small-to-big" (parent-document) retrieval
--------------------------------------------------
* PARENT = one page of an ICMR STW  (each page is one complete treatment sheet).
* CHILD  = small chunk (~500 chars) cut at bullet boundaries, never mid-sentence.
           Children are what we embed / keyword-search (small = precise matching).
* At query time rag.py finds the best child chunks, then returns their PARENT page,
  so the doctor sees the complete protocol, not a fragment.

Every child chunk is prefixed with its guideline title (e.g. "Acute Rhinosinusitis: ...")
so a chunk like "Oral antibiotics - Amoxycillin for 7-10 days" still knows which disease it belongs to.
"""

import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

TEXT_DIR = Path("medical_guidelines_text")
INDEX_DIR = Path("models/rag_index")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_MAX_CHARS = 500
MIN_PAGE_CHARS = 200
DUPLICATE_JACCARD = 0.85      # pages sharing >=85% of their words are treated as the same STW

# Manual escape hatch: {"1725952349_ent.txt": "ent"}  (only needed if a file is tagged 'general')
DOMAIN_OVERRIDES = {}

# file-name token (prefix) -> domain key used by classifier.py / rag.py
DOMAIN_RULES = [
    ("cardiology",   ("cardio", "cardiac")),
    ("hypertension", ("hypertens", "htn")),
    ("nephrology",   ("nephro", "renal", "kidney")),
    ("neurology",    ("neuro",)),
    ("orthopaedics", ("ortho",)),
    ("psychiatry",   ("psych",)),
]
ENT_TOKENS = {"ent", "otolaryngology"}   # matched as WHOLE tokens only (see infer_domain)


# --------------------------------------------------------------------------- helpers
def infer_domain(file_path: Path) -> str:
    """
    Tags a file with a domain using WHOLE WORDS of its file/folder name.
    (The old version used substring matching, so 'ent' wrongly matched words like
     'dementia', 'management', 'adolescent' and silently tagged them as ENT.)
    """
    if file_path.name in DOMAIN_OVERRIDES:
        return DOMAIN_OVERRIDES[file_path.name]

    tokens = re.findall(r"[a-z]+", f"{file_path.parent.name} {file_path.stem}".lower())
    if any(t in ENT_TOKENS for t in tokens):
        return "ent"
    for domain, prefixes in DOMAIN_RULES:
        if any(t.startswith(p) for t in tokens for p in prefixes):
            return domain
    return "general"


def word_set(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def extract_title(page_text: str, fallback: str) -> str:
    m = re.search(r"Management of\s*\n\s*([^\n]+)", page_text)
    if m:
        return m.group(1).strip().title()
    for ln in page_text.split("\n")[:8]:
        if ln.isupper() and len(ln) > 6 and not ln.startswith("---"):
            return ln.strip().title()
    return fallback


def parse_pages(raw_text: str):
    """Splits a .txt file into [(page_number, page_text), ...] using the '--- Page N ---' markers."""
    parts = re.split(r"^--- Page (\d+) ---\s*$", raw_text, flags=re.MULTILINE)
    if len(parts) == 1:                       # no markers at all
        return [(1, raw_text.strip())]
    return [(int(parts[i]), parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]


BULLET_RE = re.compile(r"^(?:[∙•·▪●○\-\*]|\d{1,2}[.)])\s*")


def split_items(text: str):
    """Groups lines into logical items: one bullet / numbered point / heading(+its plain lines)."""
    items = []
    for ln in (l.strip() for l in text.split("\n")):
        if not ln:
            continue
        new_item = (not items) or bool(BULLET_RE.match(ln)) or (ln.isupper() and len(ln) > 3)
        if new_item:
            items.append(ln)
        else:
            items[-1] += " " + ln
    return items


def chunk_page(text: str, max_chars: int = CHUNK_MAX_CHARS):
    """Packs whole items into chunks <= max_chars. Never cuts inside a bullet/sentence."""
    pieces = []
    for it in split_items(text):
        if len(it) <= 2 * max_chars:
            pieces.append(it)
        else:                                  # very long single item -> split on sentence ends
            cur = ""
            for s in re.split(r"(?<=[.!?])\s+", it):
                if cur and len(cur) + 1 + len(s) > max_chars:
                    pieces.append(cur)
                    cur = s
                else:
                    cur = f"{cur} {s}".strip()
            if cur:
                pieces.append(cur)

    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + 1 + len(p) > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n{p}" if cur else p
    if cur:
        chunks.append(cur)

    if len(chunks) > 1 and len(chunks[-1]) < 80:      # avoid a useless tiny last chunk
        chunks[-2] += "\n" + chunks.pop()
    return chunks


# --------------------------------------------------------------------------- main
def build_index():
    if not TEXT_DIR.exists():
        raise FileNotFoundError(f"'{TEXT_DIR}' not found. Run extract_pdfs.py first.")
    txt_files = sorted(TEXT_DIR.rglob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found under '{TEXT_DIR}'.")
    print(f"[*] Found {len(txt_files)} text files under '{TEXT_DIR}'.")

    # 1. read all pages; process short (single STW) files first so that, when the same STW also
    #    appears inside a big merged file, the standalone copy is the one we keep.
    loaded = []
    for fp in txt_files:
        pages = parse_pages(fp.read_text(encoding="utf-8", errors="ignore"))
        loaded.append((fp, pages))
    loaded.sort(key=lambda x: len(x[1]))

    # 2. build parents (one per page) with de-duplication
    parents, kept_sets, skipped = {}, [], []
    for fp, pages in loaded:
        domain = infer_domain(fp)
        for page_no, page_text in pages:
            if len(page_text) < MIN_PAGE_CHARS:
                continue
            ws = word_set(page_text)
            dup_of = next((pid for pid, s in kept_sets if jaccard(ws, s) >= DUPLICATE_JACCARD), None)
            if dup_of:
                skipped.append((f"{fp.name} p{page_no}", dup_of))
                continue
            pid = f"{fp.stem}_p{page_no}"
            kept_sets.append((pid, ws))
            parents[pid] = {
                "title": extract_title(page_text, fp.stem),
                "source_file": fp.name,
                "page": page_no,
                "domain": domain,
                "text": page_text,
            }

    print(f"[*] {len(parents)} unique guideline pages kept; {len(skipped)} duplicate pages skipped.")
    for a, b in skipped:
        print(f"    - skipped {a}  (duplicate of {b})")

    # 3. children = small chunks, each carrying its parent id and the guideline title
    chunks = []
    for pid, par in parents.items():
        for i, body in enumerate(chunk_page(par["text"])):
            chunks.append({
                "chunk_id": f"{pid}_c{i}",
                "parent_id": pid,
                "title": par["title"],
                "domain": par["domain"],
                "body": body,
            })
    print(f"[*] Produced {len(chunks)} chunks from {len(parents)} pages "
          f"(avg {sum(len(c['body']) for c in chunks) // max(len(chunks), 1)} chars).")

    # sanity table so wrongly tagged domains are visible immediately
    print("\n[*] Pages per domain (check this looks right!):")
    counts = {}
    for par in parents.values():
        counts[par["domain"]] = counts.get(par["domain"], 0) + 1
    for d, n in sorted(counts.items()):
        print(f"    {d:14s} {n}")
    for fp, _ in loaded:
        if infer_domain(fp) == "general":
            print(f"    [!] '{fp.name}' was tagged 'general' -> add it to DOMAIN_OVERRIDES")

    # 4. dense index (FAISS). BM25 is rebuilt in rag.py at load time (tiny corpus, instant).
    print(f"\n[*] Loading embedding model: {EMBED_MODEL_NAME}...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [f"{c['title']}: {c['body']}" for c in chunks]
    print("[*] Encoding chunks...")
    emb = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True).astype("float32")

    index = faiss.IndexFlatIP(emb.shape[1])     # inner product on normalised vectors = cosine similarity
    index.add(emb)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    (INDEX_DIR / "parents.json").write_text(json.dumps(parents, ensure_ascii=False, indent=1), encoding="utf-8")
    (INDEX_DIR / "config.json").write_text(json.dumps({"embed_model": EMBED_MODEL_NAME}), encoding="utf-8")

    print(f"\n[+] Saved to '{INDEX_DIR}/': faiss.index ({index.ntotal} vectors), chunks.json, parents.json")


if __name__ == "__main__":
    build_index()