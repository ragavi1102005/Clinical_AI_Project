"""
Module 3b: Hybrid RAG retriever (online, runs for every query).

Pipeline
--------
 query (+ NER entities)
   -> FAISS dense search  (meaning)           \
   -> BM25 sparse search  (exact words)        /-> Reciprocal Rank Fusion  (chunk level)
   -> group chunks by their parent PAGE, page score = its best chunk   (document level)
   -> domain prior from the classifier (soft boost, see ROUTING_MODE)
   -> return the top-K whole guideline pages  + the passages that matched
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

INDEX_DIR = Path("models/rag_index")
RRF_K = 60

# "soft": search ALL guidelines, then multiply scores of the predicted domain by DOMAIN_BOOST.
#         A wrong classifier prediction can then no longer hide the right guideline.
# "hard": only search guidelines of the predicted domain (original behaviour).
ROUTING_MODE = "soft"
DOMAIN_BOOST = 1.3

# a predicted domain that should also search related domains
DOMAIN_EXPANSION = {"cardiology": ["cardiology", "hypertension"]}


def tokenize(text: str):
    """lower-case, keep letters/digits only (so 'Amoxycillin/' == 'amoxycillin'), drop stop-words."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in ENGLISH_STOP_WORDS]


class HybridRetriever:
    def __init__(self, index_dir: Path = INDEX_DIR):
        print("[*] Loading hybrid RAG index...")
        config = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
        self.embedder = SentenceTransformer(config["embed_model"])
        self.faiss_index = faiss.read_index(str(index_dir / "faiss.index"))
        self.chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        self.parents = json.loads((index_dir / "parents.json").read_text(encoding="utf-8"))

        self.bm25 = BM25Okapi([tokenize(f"{c['title']} {c['body']}") for c in self.chunks])
        print(f"[+] Loaded {len(self.parents)} guideline pages / {len(self.chunks)} chunks "
              f"(routing mode: {ROUTING_MODE}).")

    # ------------------------------------------------------------------ query
    @staticmethod
    def _build_query(raw_text, entities=None):
        return f"{raw_text}\nKey clinical terms: {', '.join(entities)}" if entities else raw_text

    def _wanted_domains(self, domain):
        if not domain or domain == "general":
            return None
        return set(DOMAIN_EXPANSION.get(domain, [domain]))

    # ------------------------------------------------------------------ the two searchers
    def _dense_search(self, query, candidate_idxs, top_k):
        qv = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        if candidate_idxs is None:
            scores, idxs = self.faiss_index.search(qv, min(top_k, self.faiss_index.ntotal))
            return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]
        sub = np.array([self.faiss_index.reconstruct(i) for i in candidate_idxs]).astype("float32")
        sims = sub @ qv[0]
        order = np.argsort(-sims)[:top_k]
        return [(candidate_idxs[i], float(sims[i])) for i in order]

    def _sparse_search(self, query, candidate_idxs, top_k):
        scores = self.bm25.get_scores(tokenize(query))
        pool = enumerate(scores) if candidate_idxs is None else ((i, scores[i]) for i in candidate_idxs)
        ranked = sorted(pool, key=lambda x: x[1], reverse=True)[:top_k]
        return [(i, float(s)) for i, s in ranked if s > 0]      # ignore chunks with zero keyword overlap

    # ------------------------------------------------------------------ main entry
    def retrieve(self, query_text, entities=None, domain=None, top_k=2, fetch_k=40):
        query = self._build_query(query_text, entities)
        wanted = self._wanted_domains(domain)

        candidate_idxs = None
        if ROUTING_MODE == "hard" and wanted:
            ids = [i for i, c in enumerate(self.chunks) if c["domain"] in wanted]
            candidate_idxs = ids or None

        dense = self._dense_search(query, candidate_idxs, fetch_k)
        sparse = self._sparse_search(query, candidate_idxs, fetch_k)

        # 1) Reciprocal Rank Fusion on chunks
        rrf = defaultdict(float)
        for rank, (idx, _) in enumerate(dense):
            rrf[idx] += 1.0 / (RRF_K + rank + 1)
        for rank, (idx, _) in enumerate(sparse):
            rrf[idx] += 1.0 / (RRF_K + rank + 1)

        # 2) chunk scores -> parent (page) scores.
        #    Page score = score of its BEST chunk ("best passage wins"). Summing chunk scores would
        #    favour pages simply because they were split into more chunks.
        parent_score = {}
        parent_hits = defaultdict(list)
        for idx, s in sorted(rrf.items(), key=lambda x: x[1], reverse=True):
            pid = self.chunks[idx]["parent_id"]
            parent_score.setdefault(pid, s)          # first = highest, list is sorted
            parent_hits[pid].append((s, idx))

        # 3) soft domain prior
        if ROUTING_MODE == "soft" and wanted:
            for pid in parent_score:
                if self.parents[pid]["domain"] in wanted:
                    parent_score[pid] *= DOMAIN_BOOST

        # 4) top-K parents
        results = []
        for pid, score in sorted(parent_score.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            par = self.parents[pid]
            hits = sorted(parent_hits[pid], reverse=True)[:3]
            results.append({
                "parent_id": pid,
                "title": par["title"],
                "source_file": par["source_file"],
                "page": par["page"],
                "domain": par["domain"],
                "text": par["text"],                                   # the full guideline page
                "matched_passages": [self.chunks[i]["body"] for _, i in hits],
                "rrf_score": round(score, 4),
            })
        return results


if __name__ == "__main__":
    r = HybridRetriever()
    q = "Patient presents with severe acute rhinosinusitis, facial pain, and nasal obstruction."
    for res in r.retrieve(q, entities=["acute rhinosinusitis", "facial pain"], domain="ent", top_k=2):
        print(f"\n=== {res['title']}  ({res['source_file']} p{res['page']})  score {res['rrf_score']} ===")
        print("Matched passages:")
        for p in res["matched_passages"]:
            print("  *", p[:160].replace("\n", " "))
        print("\nFull page preview:\n", res["text"][:500])