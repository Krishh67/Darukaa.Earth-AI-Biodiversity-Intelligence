import os
import json
import numpy as np
import faiss
import pickle
from google import genai
from google.genai import types

from backend.rag.fusion import reciprocal_rank_fusion
from backend.rag.reranker import Reranker

class RAGRetriever:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.rag_dir = os.path.join(self.data_dir, "rag")
        
        self.chunks_file = os.path.join(self.rag_dir, "chunks", "chunks.jsonl")
        self.embed_dir = os.path.join(self.rag_dir, "embeddings")
        self.index_dir = os.path.join(self.rag_dir, "index")
        
        self.chunk_ids_file = os.path.join(self.embed_dir, "chunk_ids.json")
        self.faiss_index_file = os.path.join(self.index_dir, "faiss.index")
        self.bm25_index_file = os.path.join(self.index_dir, "bm25.pkl")
        
        self.dimensions = 3072
        self.embedding_model = "gemini-embedding-2"
        self.client = genai.Client()
        
        self.reranker = Reranker()
        
        self._load_indices()
        
    def _load_indices(self):
        print("Loading RAG Indices...")
        
        # Load FAISS
        if os.path.exists(self.faiss_index_file):
            self.faiss_index = faiss.read_index(self.faiss_index_file)
        else:
            self.faiss_index = None
            print("Warning: FAISS index not found.")
            
        # Load BM25
        if os.path.exists(self.bm25_index_file):
            with open(self.bm25_index_file, 'rb') as f:
                self.bm25 = pickle.load(f)
        else:
            self.bm25 = None
            print("Warning: BM25 index not found.")
            
        # Load Chunk IDs mapped to the index
        if os.path.exists(self.chunk_ids_file):
            with open(self.chunk_ids_file, 'r', encoding='utf-8') as f:
                self.chunk_ids = json.load(f)
        else:
            self.chunk_ids = []
            
        # Load full chunks metadata
        self.chunks_db = {}
        if os.path.exists(self.chunks_file):
            with open(self.chunks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    chunk = json.loads(line)
                    self.chunks_db[chunk["chunk_id"]] = chunk
                    
    def get_query_embedding(self, query):
        config = types.EmbedContentConfig(
            output_dimensionality=self.dimensions
        )
        formatted_query = f"task: search result | query: {query}"
        result = self.client.models.embed_content(
            model=self.embedding_model,
            contents=formatted_query,
            config=config
        )
        emb = np.array(result.embeddings[0].values, dtype=np.float32)
        emb = emb / np.linalg.norm(emb)
        return emb
        
    def retrieve(self, query, top_k=5):
        if not self.faiss_index or not self.bm25 or not self.chunk_ids:
            return []
            
        # 1. Semantic Search (FAISS)
        query_emb = self.get_query_embedding(query)
        q_matrix = np.array([query_emb], dtype=np.float32)
        k_retrieve = 30
        distances, indices = self.faiss_index.search(q_matrix, k_retrieve)
        
        dense_results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunk_ids):
                cid = self.chunk_ids[idx]
                if cid in self.chunks_db:
                    doc = self.chunks_db[cid].copy()
                    doc["faiss_score"] = float(distances[0][i])
                    dense_results.append(doc)
                    
        # 2. Keyword Search (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:k_retrieve]
        
        lexical_results = []
        for idx in top_bm25_indices:
            if idx < len(self.chunk_ids):
                cid = self.chunk_ids[idx]
                if cid in self.chunks_db:
                    doc = self.chunks_db[cid].copy()
                    doc["bm25_score"] = float(bm25_scores[idx])
                    lexical_results.append(doc)
                    
        # 3. Reciprocal Rank Fusion
        fused_candidates = reciprocal_rank_fusion(dense_results, lexical_results, top_n=30)
        
        # 4. Cross-Encoder Reranking
        final_results = self.reranker.rerank(query, fused_candidates, top_k=top_k)
        
        return final_results

