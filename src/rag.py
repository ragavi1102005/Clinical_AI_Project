"""
Module 3b: Hybrid RAG Retriever
Combines FAISS dense retrieval and BM25 sparse retrieval via Reciprocal
Rank Fusion (RRF), with domain filtering (including related-domain
expansion, e.g. cardiology also searches hypertension), NER-based query
enrichment, and near-duplicate suppression.
"""

import json
import pickle
from pathlib import Path
from difflib import SequenceMatcher

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("models/rag_index")
RRF_K = 60
DUPLICATE_SIMILARITY_THRESHOLD = 0.85

# When a predicted domain also relates to another domain's documents,
# list them here so both get searched together.
DOMAIN_EXPANSION = {
    "cardiology": ["cardiology", "hypertension"],
}


def _is_near_duplicate(text_a: str, text_b: str, threshold: float = DUPLICATE_SIMILARITY_THRESHOLD) -> bool:
    return SequenceMatcher(None, text_a[:300], text_b[:300]).ratio() >= threshold


class HybridRetriever:
    def __init__(self, index_dir: Path = INDEX_DIR):
        print("[*] Loading hybrid RAG index...")

        with open(index_dir / "config.json") as f:
            config = json.load(f)

        self.embedder = SentenceTransformer(config["embed_model"])
        self.faiss_index = faiss.read_index(str(index_dir / "faiss.index"))

        with open(index_dir / "bm25.pkl", "rb") as f:
            self.bm25_index = pickle.load(f)

        with open(index_dir / "records.json", encoding="utf-8") as f:
            self.records = json.load(f)

        print(f"[+] Loaded {len(self.records)} chunks "
              f"({self.faiss_index.ntotal} dense vectors, BM25 ready).")

    def _build_query(self, raw_text: str, entities=None) -> str:
        if entities:
            return f"{raw_text}\nKey clinical terms: {', '.join(entities)}"
        return raw_text

    def _domain_candidate_indices(self, domain: str):
        if not domain or domain == "general":
            return None

        domains_to_match = DOMAIN_EXPANSION.get(domain, [domain])
        idxs = [i for i, r in enumerate(self.records) if r["domain"] in domains_to_match]
        return idxs if idxs else None

    def _dense_search(self, query: str, candidate_idxs, top_k: int):
        query_vec = self.embedder.encode([query], normalize_embeddings=True).astype("float32")

        if candidate_idxs is None:
            scores, idxs = self.faiss_index.search(query_vec, top_k)
            return list(zip(idxs[0].tolist(), scores[0].tolist()))

        sub_vecs = np.array([
            self.faiss_index.reconstruct(i) for i in candidate_idxs
        ]).astype("float32")
        sims = sub_vecs @ query_vec[0]
        order = np.argsort(-sims)[:top_k]
        return [(candidate_idxs[i], float(sims[i])) for i in order]

    def _sparse_search(self, query: str, candidate_idxs, top_k: int):
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)

        if candidate_idxs is not None:
            masked = [(i, scores[i]) for i in candidate_idxs]
        else:
            masked = list(enumerate(scores))

        masked.sort(key=lambda x: x[1], reverse=True)
        return masked[:top_k]

    def retrieve(self, query_text: str, entities=None, domain: str = None,
                 top_k: int = 5, fetch_k: int = 20):
        query = self._build_query(query_text, entities)
        candidate_idxs = self._domain_candidate_indices(domain)

        dense_results = self._dense_search(query, candidate_idxs, fetch_k)
        sparse_results = self._sparse_search(query, candidate_idxs, fetch_k)

        rrf_scores = {}
        for rank, (idx, _) in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (RRF_K + rank + 1)
        for rank, (idx, _) in enumerate(sparse_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (RRF_K + rank + 1)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in ranked:
            if len(results) >= top_k:
                break
            record = self.records[idx]
            if any(_is_near_duplicate(record["text"], r["text"]) for r in results):
                continue
            results.append({
                "text": record["text"],
                "domain": record["domain"],
                "source_file": record["source_file"],
                "chunk_id": record["chunk_id"],
                "rrf_score": round(score, 4),
            })

        return results


if __name__ == "__main__":
    retriever = HybridRetriever()
    sample_query = "Patient presents with severe acute rhinosinusitis, facial pain."
    results = retriever.retrieve(sample_query, domain="ent", top_k=3)
    for r in results:
        print(f"\n--- {r['source_file']} | RRF: {r['rrf_score']} ---")
        print(r["text"][:300])