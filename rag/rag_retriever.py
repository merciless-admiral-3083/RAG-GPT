import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from .reranker import SimpleReranker


class RAGRetriever:
    def __init__(self, index_path, data_path):
        self.index = faiss.read_index(index_path)

        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.reranker = SimpleReranker()

    def retrieve(self, query, top_k=3):
        q_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        D, I = self.index.search(q_emb, max(top_k * 5, top_k))

        candidates = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            candidates.append({
                "text": self.data[idx]["text"],
                "distance": float(dist)   
            })

        if not candidates:
            return []

        reranked = self.reranker.rerank(query, candidates)

        return reranked[:top_k]