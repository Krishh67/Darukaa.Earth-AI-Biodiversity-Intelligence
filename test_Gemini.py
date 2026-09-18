import os
import numpy as np
from google import genai
from google.genai import types

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

client = genai.Client(api_key=API_KEY)

MODEL = "gemini-embedding-2"
DIMENSION = 768

documents = [
    "Soil organic carbon influences soil biodiversity and microbial activity.",
    "Habitat fragmentation can reduce habitat connectivity and affect species movement.",
    "Rainfall and water availability influence species survival and ecosystem functioning."
]

print("=" * 60)
print("Testing Gemini Embedding API")
print("=" * 60)

embeddings = []

for i, text in enumerate(documents, 1):
    print(f"\nEmbedding document {i}...")

    result = client.models.embed_content(
        model=MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=DIMENSION
        )
    )

    vector = np.array(result.embeddings[0].values, dtype=np.float32)

    embeddings.append(vector)

    print(f"  Success")
    print(f"  Dimensions: {len(vector)}")
    print(f"  First 5 values: {vector[:5]}")
    print(f"  Norm: {np.linalg.norm(vector):.6f}")

embeddings = np.vstack(embeddings)

print("\n" + "=" * 60)
print("FINAL TEST")
print("=" * 60)

print(f"Model: {MODEL}")
print(f"Number of embeddings: {len(embeddings)}")
print(f"Embedding shape: {embeddings.shape}")
print(f"Data type: {embeddings.dtype}")
print(f"Contains NaN: {np.isnan(embeddings).any()}")
print(f"Contains Inf: {np.isinf(embeddings).any()}")

# Simple semantic similarity test
similarity = np.dot(embeddings[0], embeddings[1]) / (
    np.linalg.norm(embeddings[0]) *
    np.linalg.norm(embeddings[1])
)

print(f"\nSimilarity between document 1 and 2: {similarity:.4f}")

print("\nGemini Embedding test completed successfully.")