import os
import json
import numpy as np
import faiss
import pickle
from rank_bm25 import BM25Okapi
from google import genai
from google.genai import types
import time
import math
from dotenv import load_dotenv

load_dotenv(override=True)

# Configuration
TEST_MODE = False  # Set to False to run all chunks
TEST_LIMIT = 5

DATA_DIR = "data"
RAG_DIR = os.path.join(DATA_DIR, "rag")
CHUNKS_DIR = os.path.join(RAG_DIR, "chunks")
CHUNKS_FILE = os.path.join(CHUNKS_DIR, "chunks.jsonl")

EMBED_DIR = os.path.join(RAG_DIR, "embeddings")
INDEX_DIR = os.path.join(RAG_DIR, "index")

os.makedirs(EMBED_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

EMBEDDINGS_FILE = os.path.join(EMBED_DIR, "embeddings.npy")
CHUNK_IDS_FILE = os.path.join(EMBED_DIR, "chunk_ids.json")
FAISS_INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")
BM25_INDEX_FILE = os.path.join(INDEX_DIR, "bm25.pkl")

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("WARNING: GEMINI_API_KEY environment variable not set.")
client = genai.Client()

EMBEDDING_MODEL = "gemini-embedding-2"
DIMENSIONS = 3072

def exponential_backoff(attempt):
    time.sleep(math.pow(2, attempt) + np.random.uniform(0, 1))

def get_embeddings_batch(batch_chunks):
    config = types.EmbedContentConfig(
        output_dimensionality=DIMENSIONS
    )
    
    contents = []
    for chunk in batch_chunks:
        title_str = chunk.get("document_title", "")
        title_str = title_str if title_str else "none"
        formatted_content = f"title: {title_str} | text: {chunk['text']}"
        # Create a Content object format for independent embeddings
        contents.append({"parts": [{"text": formatted_content}]})
    
    max_retries = 10
    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=contents,
                config=config
            )
            return [emb.values for emb in result.embeddings]
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or attempt < max_retries - 1:
                # Try to extract the requested retry delay
                sleep_time = math.pow(2, attempt) + np.random.uniform(0, 1)
                import re
                match = re.search(r'retry in ([\d\.]+)s', err_str)
                if match:
                    sleep_time = max(sleep_time, float(match.group(1)) + 2.0)
                
                print(f"  Retry {attempt + 1}/{max_retries} due to error (Waiting {sleep_time:.1f}s)...")
                time.sleep(sleep_time)
            else:
                print(f"Failed after {max_retries} attempts.")
                return None

def generate_embeddings():
    print("Loading chunks...")
    chunks = []
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
            
    if TEST_MODE:
        # For testing, grab chunks that are NOT embedded yet, up to TEST_LIMIT
        pass # Handle below to ensure we test NEW chunks
            
    existing_embeddings = []
    existing_ids = []
    
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(CHUNK_IDS_FILE):
        print("Found existing embeddings, loading for resumption...")
        with open(CHUNK_IDS_FILE, 'r', encoding='utf-8') as f:
            existing_ids = json.load(f)
        if len(existing_ids) > 0:
            existing_embeddings = list(np.load(EMBEDDINGS_FILE))
            print(f"Loaded {len(existing_ids)} existing embeddings.")
        
    existing_ids_set = set(existing_ids)
    
    remaining_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids_set]
    
    if TEST_MODE:
        remaining_chunks = remaining_chunks[:TEST_LIMIT]
        print(f"TEST MODE: Limited to {TEST_LIMIT} NEW chunks.")
        
    print(f"{len(remaining_chunks)} chunks left to embed.")
    
    if remaining_chunks:
        print("Starting embedding generation...")
        
        batch_size = 50
        
        for i in range(0, len(remaining_chunks), batch_size):
            batch = remaining_chunks[i:i+batch_size]
            print(f"Embedding batch {i//batch_size + 1}/{(len(remaining_chunks) + batch_size - 1)//batch_size} (size: {len(batch)})...")
            
            batch_embeddings = get_embeddings_batch(batch)
            
            if batch_embeddings is None:
                print("Batch embedding failed after retries. Stopping gracefully to allow resumption later.")
                break
                
            if len(batch_embeddings) != len(batch):
                raise ValueError(f"Mismatch: sent {len(batch)} inputs, got {len(batch_embeddings)} embeddings")
                
            batch_ids = [c["chunk_id"] for c in batch]
            
            existing_embeddings.extend(batch_embeddings)
            existing_ids.extend(batch_ids)
            
            # Save progress after every successful batch
            np.save(EMBEDDINGS_FILE, np.array(existing_embeddings, dtype=np.float32))
            with open(CHUNK_IDS_FILE, 'w', encoding='utf-8') as f:
                json.dump(existing_ids, f)
                
            print(f"Saved progress. Total embeddings: {len(existing_ids)}")
                
    emb_array = np.load(EMBEDDINGS_FILE)
    if len(emb_array) > 0 and emb_array.shape[1] != DIMENSIONS:
        raise ValueError(f"Expected {DIMENSIONS} dims, got {emb_array.shape[1]}")
    if np.isnan(emb_array).any() or np.isinf(emb_array).any():
        raise ValueError("Embeddings contain NaN or Inf values")
        
    print(f"Successfully embedded and saved {len(existing_ids)} chunks.")
    return emb_array, existing_ids, chunks

def build_faiss(emb_array):
    print("Building FAISS index...")
    if len(emb_array) == 0:
        print("No embeddings to build FAISS index.")
        return
        
    faiss.normalize_L2(emb_array)
    index = faiss.IndexFlatIP(DIMENSIONS)
    index.add(emb_array)
    
    faiss.write_index(index, FAISS_INDEX_FILE)
    print(f"FAISS index built with {index.ntotal} vectors.")

def build_bm25(chunks, chunk_ids):
    print("Building BM25 index...")
    if not chunk_ids:
        print("No chunks to build BM25 index.")
        return
        
    chunk_dict = {c["chunk_id"]: c["text"] for c in chunks}
    texts_in_order = [chunk_dict[cid] for cid in chunk_ids]
    
    tokenized_corpus = [doc.lower().split() for doc in texts_in_order]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(BM25_INDEX_FILE, 'wb') as f:
        pickle.dump(bm25, f)
    print("BM25 index built.")

def main():
    print("Starting Stage 2: RAG Pipeline - Embeddings & Indices")
    emb_array, chunk_ids, chunks = generate_embeddings()
    build_faiss(emb_array)
    build_bm25(chunks, chunk_ids)
    print("Stage 2 building completed successfully.")

if __name__ == "__main__":
    main()

