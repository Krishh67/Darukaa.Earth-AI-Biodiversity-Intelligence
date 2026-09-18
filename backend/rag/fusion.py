def reciprocal_rank_fusion(dense_results, lexical_results, k=60, top_n=30):
    """
    Fuses two lists of ranked results using Reciprocal Rank Fusion.
    dense_results, lexical_results: Lists of dicts representing documents.
    Each document dict must have a unique 'chunk_id'.
    """
    rrf_scores = {}
    
    # helper to process a result list
    def add_ranks(results):
        for rank, doc in enumerate(results):
            chunk_id = doc["chunk_id"]
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {"score": 0.0, "doc": doc}
            # RRF formula: 1 / (k + rank)
            # rank is 0-indexed, so we add 1
            rrf_scores[chunk_id]["score"] += 1.0 / (k + rank + 1)
            
    add_ranks(dense_results)
    add_ranks(lexical_results)
    
    # Sort by RRF score descending
    fused = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    
    # Return just the doc dictionaries, possibly adding the rrf_score to them
    final_results = []
    for item in fused[:top_n]:
        doc = item["doc"].copy()
        doc["rrf_score"] = item["score"]
        final_results.append(doc)
        
    return final_results

