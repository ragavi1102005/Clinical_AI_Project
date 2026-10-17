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
NEPHRO_SUBSTRINGS = ("kidney", "dialysis", "glomerul", "nephro", "renal")
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
    # kidney guidelines have run-together names ("hemodialysis", "acutekidneyinjury", "uti"), so use substrings
    stem = file_path.stem.lower()
    if "uti" in tokens or any(k in stem for k in NEPHRO_SUBSTRINGS):
        return "nephrology"
    for domain, prefixes in DOMAIN_RULES:
        if any(t.startswith(p) for t in tokens for p in prefixes):
            return domain
    return "general"


# --------------------------------------------------------------------------- noise removal
STRONG = re.compile(r"doi:|PMID|PMCID|Epub|Erratum|accessed online|\bet al\b", re.I)
MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
JOURNAL = re.compile(rf"\b(?:19|20)\d\d;\s?\d|\b(?:19|20)\d\d\s+{MONTHS}\b|\b[A-Z][a-z]+ J\. ")
AUTHOR = re.compile(r"\b[A-Z][a-z'\-]+ [A-Z]{1,3}\b[,;.]")
NUM_START = re.compile(r"^\s*\d{1,2}\s*[.)]\s*[A-Z]")
YEAR = re.compile(r"\b(?:19|20)\d\d\b")
ABBR_KEY = r"[A-Z][A-Za-z0-9/\-]*[A-Z0-9][A-Za-z0-9/\-]*"          # at least two capitals: AP, ESR, NSAIDs, HFrEF
ABBR_FIRST = re.compile(rf"^{ABBR_KEY}:\s+[A-Z]")
ABBR_ANY = re.compile(rf"(?<!\w){ABBR_KEY}:\s+[A-Z]")
BANNERS = re.compile(r"^(REFERENCES?|ABBREVIATIONS?|ABBREVATIONS?|KEEP A HIGH THRESHOLD FOR INVASIVE PROCEDURES|"
                     r"(January|February|March|April|May|June|July|August|September|October|November|December)/?\s*\d{4})$", re.I)


def _ref_like(ln: str) -> bool:
    return bool(STRONG.search(ln) or JOURNAL.search(ln) or len(AUTHOR.findall(ln)) >= 3
                or (NUM_START.match(ln) and YEAR.search(ln)))


def strip_noise(text: str) -> str:
    """Line-level removal of literature references, abbreviation lists and banner/heading leftovers.
    (Text-flow extraction scatters these anywhere on the page, so we cannot rely on a 'REFERENCES' heading.)"""
    lines = [l for l in text.split("\n") if not BANNERS.match(l.strip())]
    flag = [_ref_like(l) for l in lines]
    for _ in range(2):          # a title/wrapped line sandwiched between reference lines is part of the reference
        for k, l in enumerate(lines):
            if not flag[k] and any(flag[max(0, k - 2):k]) and any(flag[k + 1:k + 3]):
                flag[k] = True
    keep = []
    for k, l in enumerate(lines):
        if flag[k]:
            continue
        if ABBR_FIRST.match(l.strip()) or len(ABBR_ANY.findall(l)) >= 2:
            continue
        keep.append(l)
    return "\n".join(keep)


def word_set(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def is_clinical_page(page_text: str) -> bool:
    """True for real guideline sheets. Covers, tables of contents, introductions and committee lists
    carry neither the 'Standard Treatment Workflow (STW)' title line nor an ICD code."""
    return bool(re.search(r"Standard Treatment Workflow \(STW\)", page_text)) or bool(re.search(r"\bICD[- ]?\d", page_text))


def _nice_title(text: str) -> str:
    out = []
    for w in text.split():
        core = w.strip("()")
        out.append(w if (w.startswith("(") and core.isupper() and len(core) <= 4) else w.title())   # keep "(SD)"
    return " ".join(out)


def title_and_continued(page_text: str, fallback: str):
    """
    Finds the guideline title wherever the 'Standard Treatment Workflow (STW)' line sits on the page
    (text-flow order does not always put it first) and reads the capitalised lines that follow it.
    Also reports whether the sheet is a '(Continued)' second page.
    """
    lines = [l.strip() for l in page_text.split("\n") if l.strip()]
    for i, ln in enumerate(lines):
        if "Standard Treatment Workflow (STW)" in ln:
            parts, continued = [], False
            for nxt in lines[i + 1:i + 5]:
                if re.match(r"^\(?continued\)?$", nxt, re.IGNORECASE):
                    continued = True
                    break
                if re.match(r"^ICD", nxt):
                    break
                if parts and not re.search(r"\b(IN|OF|AND|FOR|WITH|THE|OR|TO|&)$", parts[-1]):
                    break                      # title is complete; next caps line is a banner, not part of it
                if nxt.isupper() or re.match(r"^Management of\b", nxt):
                    parts.append(nxt)
                else:
                    break
            title = re.sub(r"^Management of\s+", "", " ".join(parts), flags=re.IGNORECASE).strip()
            if title:
                return _nice_title(title), continued
    for ln in lines[:8]:
        if ln.isupper() and len(ln) > 6:
            return _nice_title(ln), False
    return fallback, False


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
    # run-on text without sentence ends (common in infographic tables): cut at word boundaries
    final = []
    for pc in pieces:
        while len(pc) > 2 * max_chars:
            cut = pc.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            final.append(pc[:cut]); pc = pc[cut:].strip()
        final.append(pc)
    pieces = final

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

    # 2. build parents (one per guideline sheet) with de-duplication
    parents, kept_sets, skipped, non_clinical = {}, [], [], 0
    for fp, pages in loaded:
        domain = infer_domain(fp)

        # 2a. keep guideline pages only, and glue a "(Continued)" page onto the sheet it continues
        sheets = []
        for page_no, page_text in pages:
            if len(page_text) < MIN_PAGE_CHARS or not is_clinical_page(page_text):
                non_clinical += 1
                continue
            page_text = strip_noise(page_text)
            title, continued = title_and_continued(page_text, fp.stem)
            if continued and sheets:
                sheets[-1]["text"] += "\n" + page_text
                sheets[-1]["last"] = page_no
            else:
                sheets.append({"first": page_no, "last": page_no, "title": title, "text": page_text})

        # 2b. drop sheets that already exist (same STW inside a merged file) and store the rest
        for sh in sheets:
            ws = word_set(sh["text"])
            dup_of = next((pid for pid, s_ in kept_sets if jaccard(ws, s_) >= DUPLICATE_JACCARD), None)
            label = sh["first"] if sh["first"] == sh["last"] else f"{sh['first']}-{sh['last']}"
            if dup_of:
                skipped.append((f"{fp.name} p{label} [{sh['title']}]", dup_of))
                continue
            pid = f"{fp.stem}_p{sh['first']}"
            kept_sets.append((pid, ws))
            parents[pid] = {
                "title": sh["title"],
                "source_file": fp.name,
                "page": label,
                "domain": domain,
                "text": sh["text"],
            }

    print(f"[*] {len(parents)} unique guideline sheets kept; {non_clinical} non-guideline pages skipped "
          f"(covers, contents, committee lists); {len(skipped)} duplicate sheets skipped.")
    for a_, b_ in skipped:
        print(f"    - skipped {a_}  (duplicate of {b_})")
    print("\n[*] Guidelines indexed:")
    for par in parents.values():
        print(f"    [{par['domain']:12s}] {par['title']}   ({par['source_file']} p{par['page']})")

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