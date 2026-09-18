import os
import json
import numpy as np
import faiss
import pickle
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Configuration
DATA_DIR = "data"
RAG_DIR = os.path.join(DATA_DIR, "rag")
CHUNKS_DIR = os.path.join(RAG_DIR, "chunks")
CHUNKS_FILE = os.path.join(CHUNKS_DIR, "chunks.jsonl")

EMBED_DIR = os.path.join(RAG_DIR, "embeddings")
INDEX_DIR = os.path.join(RAG_DIR, "index")

CHUNK_IDS_FILE = os.path.join(EMBED_DIR, "chunk_ids.json")
FAISS_INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")
BM25_INDEX_FILE = os.path.join(INDEX_DIR, "bm25.pkl")

EMBEDDING_MODEL = "gemini-embedding-2"
DIMENSIONS = 3072

client = genai.Client()

def get_query_embedding(query):
    config = types.EmbedContentConfig(
        output_dimensionality=DIMENSIONS
    )
    formatted_query = f"task: search result | query: {query}"
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=formatted_query,
        config=config
    )
    # the embeddings are in a list
    emb = np.array(result.embeddings[0].values, dtype=np.float32)
    # normalize for FAISS IP
    emb = emb / np.linalg.norm(emb)
    return emb

class RAGSystem:
    def __init__(self):
        print("Loading chunks...")
        self.chunks = {}
        with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                c = json.loads(line)
                self.chunks[c["chunk_id"]] = c
                
        print("Loading chunk IDs...")
        with open(CHUNK_IDS_FILE, 'r', encoding='utf-8') as f:
            self.chunk_ids = json.load(f)
            
        print("Loading FAISS index...")
        self.faiss_index = faiss.read_index(FAISS_INDEX_FILE)
        
        print("Loading BM25 index...")
        with open(BM25_INDEX_FILE, 'rb') as f:
            self.bm25 = pickle.load(f)
            
        print("Loading Cross-Encoder...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def retrieve(self, query, top_k=5, candidate_k=30):
        # 1. FAISS retrieval
        q_emb = get_query_embedding(query)
        q_emb = np.expand_dims(q_emb, axis=0)
        faiss_scores, faiss_indices = self.faiss_index.search(q_emb, candidate_k)
        
        faiss_results = []
        for rank, idx in enumerate(faiss_indices[0]):
            chunk_id = self.chunk_ids[idx]
            faiss_results.append(chunk_id)
            
        # 2. BM25 retrieval
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # get top candidate_k indices
        bm25_indices = np.argsort(bm25_scores)[::-1][:candidate_k]
        
        bm25_results = []
        for idx in bm25_indices:
            chunk_id = self.chunk_ids[idx]
            bm25_results.append(chunk_id)
            
        # 3. RRF (Reciprocal Rank Fusion)
        rrf_scores = {}
        k_rrf = 60
        for rank, chunk_id in enumerate(faiss_results):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k_rrf + rank + 1)
            
        for rank, chunk_id in enumerate(bm25_results):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k_rrf + rank + 1)
            
        # Sort RRF results
        rrf_sorted = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:candidate_k]
        rrf_candidates = [cid for cid, _ in rrf_sorted]
        
        # 4. Cross-Encoder Re-ranking
        cross_encoder_pairs = []
        for cid in rrf_candidates:
            text = self.chunks[cid]["text"]
            cross_encoder_pairs.append([query, text])
            
        ce_scores = self.cross_encoder.predict(cross_encoder_pairs)
        
        # Combine chunks and scores
        final_results = []
        for idx, cid in enumerate(rrf_candidates):
            final_results.append({
                "chunk_id": cid,
                "score": float(ce_scores[idx])
            })
            
        # Sort by Cross-Encoder score
        final_results.sort(key=lambda x: x["score"], reverse=True)
        top_results = final_results[:top_k]
        
        # Format the output JSON
        retrieved_evidence = []
        for rank, res in enumerate(top_results):
            cid = res["chunk_id"]
            chunk_data = self.chunks[cid]
            evidence = {
                "rank": rank + 1,
                "score": res["score"],
                "text": chunk_data["text"],
                "document_title": chunk_data.get("document_title", ""),
                "source_file": chunk_data.get("source_file", ""),
                "page": chunk_data.get("page_start", 0),
                "section": chunk_data.get("section", ""),
                "environmental_variables": chunk_data.get("environmental_variables", []),
                "chunk_id": cid
            }
            retrieved_evidence.append(evidence)
            
        return {
            "query": query,
            "faiss_results": faiss_results[:5],  # Just top 5 for printing
            "bm25_results": bm25_results[:5],
            "rrf_results": rrf_candidates[:5],
            "retrieved_evidence": retrieved_evidence
        }

def main():
    rag = RAGSystem()
    
    test_queries = [
        "How does soil organic carbon affect biodiversity?",
        "How does habitat fragmentation affect biodiversity and connectivity?",
        "How do rainfall and water availability affect species survival?",
        "What are effective approaches for restoring degraded land?",
        "How do soil health, land use and climate interact to affect biodiversity?"
    ]
    
    print("\n" + "="*50)
    print("RUNNING RAG RETRIEVAL TESTS")
    print("="*50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        results = rag.retrieve(query, top_k=3)
        
        print("Dense results (Top 5 chunk_ids):")
        print(results["faiss_results"])
        
        print("\nBM25 results (Top 5 chunk_ids):")
        print(results["bm25_results"])
        
        print("\nRRF results (Top 5 chunk_ids):")
        print(results["rrf_results"])
        
        print("\nFinal reranked evidence:")
        print(json.dumps({
            "query": results["query"],
            "retrieved_evidence": results["retrieved_evidence"]
        }, indent=2))
        print("="*50)

if __name__ == "__main__":
    main()

