import faiss
import numpy as np


class VectorStore:
    """A small in-memory FAISS index using cosine similarity via normalized vectors."""

    def __init__(self):
        self.index = None
        self.chunks = []

    def build(self, vectors, chunks: list[dict]):
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim != 2 or len(vectors) == 0:
            raise ValueError("Expected a non-empty 2D embedding array.")
        if len(vectors) != len(chunks):
            raise ValueError("The number of vectors must match the number of chunks.")

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks = chunks

    def search(self, query_vector, top_k: int = 4) -> list[dict]:
        if self.index is None:
            raise RuntimeError("Vector index has not been built.")
        query_vector = np.asarray(query_vector, dtype="float32")
        scores, indices = self.index.search(query_vector, min(top_k, len(self.chunks)))

        results = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            results.append({
                "text": self.chunks[index]["text"],
                "metadata": self.chunks[index]["metadata"],
                "score": float(score)
            })
        return results
