from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
        
    def rerank(self, query, candidates, top_k=5):
        """
        Reranks a list of candidate documents based on the query.
        candidates: List of dicts, each must have 'text' and 'chunk_id'.
        """
        if not candidates:
            return []
            
        pairs = [[query, doc["text"]] for doc in candidates]
        scores = self.model.predict(pairs)
        
        # Attach scores to candidates
        for i, doc in enumerate(candidates):
            doc["cross_encoder_score"] = float(scores[i])
            
        # Sort by score descending
        ranked = sorted(candidates, key=lambda x: x["cross_encoder_score"], reverse=True)
        return ranked[:top_k]

