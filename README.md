# Darukaa.Earth AI Biodiversity Intelligence

This is a production-quality hackathon prototype for the Darukaa.Earth AI Biodiversity Challenge. It behaves as an AI Environmental Scientist, integrating Conversational AI, Scientific RAG (FAISS + BM25 + RRF), structured environmental data fetching (SoilGrids, Open-Meteo, ESA WorldCover), Multi-Metric Reasoning, and strict Verification into a unified pipeline.

## Architecture
1. **Intake & Context Manager (Groq)**: A fast frontier model acts as the front-line agent. It intercepts user queries, evaluates conversation context, and asks deep, professional clarification questions if operational details are missing. Once enough context is gathered, it generates multiple optimized queries for the RAG database.
2. **Environmental API Layer**: Automatically fetches live metrics based on coordinates:
   - **SoilGrids (ISRIC)**: Soil pH, Organic Carbon, Nitrogen, Bulk Density.
   - **Open-Meteo**: Historical & forecasted climate data.
   - **Planetary Computer (ESA WorldCover)**: STAC API and raster reading for land cover classification.
3. **Hybrid RAG Pipeline**: Combines dense semantic search (FAISS + `gemini-embedding-2`) with sparse lexical search (BM25). Results are fused using Reciprocal Rank Fusion (RRF) and reranked using a HuggingFace Cross-Encoder.
4. **Reasoning Engine (Gemini 3.5 Flash)**: Synthesizes environmental data and RAG evidence, explicitly connecting 3+ environmental variables to generate multi-metric, heavily cited recommendations.
5. **Verification Node**: A secondary LLM pass acts as a silent protector, strictly verifying the primary LLM's claims against the retrieved evidence and halting hallucinations.

## Database & Schema
- **Vector DB**: FAISS (`data/rag/index/faiss.index`) combined with a mapped JSON lookup array.
- **Lexical DB**: Pickled BM25 index (`data/rag/index/bm25.pkl`).
- **Memory Store**: Local JSON-based persistent conversation storage (`data/sessions/<uuid>.json`).
- **Structured Schemas**: Data flows between the Frontend and Backend using strict Pydantic schemas (`ChatRequest` and `ChatResponse`).

## Local Setup & How to Run

### 1. Prerequisites & Environment Variables
Ensure you have Python 3.9+ installed. Create a `.env` file in the root directory and add the following keys:
```env
GEMINI_API_KEY=your_gemini_key_here  # Get from: https://aistudio.google.com/api-keys
GROQ_API_KEY=your_groq_key_here      # Get from: https://console.groq.com/keys
HF_TOKEN=your_huggingface_token_here  # Optional: For higher rate limits when downloading the Cross-Encoder model
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the Backend API Server
```bash
python -m backend.main
```
The FastAPI server will start on `http://0.0.0.0:8000`. It will automatically load the RAG indices and Cross-Encoder model into memory.

### 4. Open the UI
Navigate to the `frontend/` folder in your file explorer and double-click `index.html` to open it in your web browser. No frontend build step is required!

## CI/CD & Deployment Notes
Given the scope of the hackathon, this prototype is designed for robust local execution. The backend can easily be containerized via Docker (using `uvicorn`) and deployed to services like Google Cloud Run or AWS Fargate. The frontend is vanilla HTML/JS/Tailwind and can be hosted statically on Vercel or GitHub Pages. The FAISS indices are small enough to be bundled in the container, but would be migrated to Pinecone or Weaviate for production scaling.

## How to Demo
1. **The Clarification Flex**: Start a New Chat and type: *"Biodiversity is declining on my land. What should I do?"* The Groq Intake manager will immediately intercept this vague query and ask a professional follow-up question.
2. **The Multi-Metric AI Scientist Flex**: In the top navigation bar, select **Mumbai, India**. Open the **Structured JSON Input** box and paste:
   ```json
   {
     "soil": { "soc": 0.3, "ph": 6.5 },
     "climate": { "precipitation_annual_sum": 300 },
     "land_cover": { "class_name": "Monoculture wheat" }
   }
   ```
   Then send: *"I farm monoculture wheat in a semi-arid region. My soil organic carbon has dropped to critically low levels and water availability is poor. How can I restore soil health and improve biodiversity?"*
   The system will extract your metrics, fetch RAG data, and output a highly detailed, referenced action plan spanning multiple variables!

*Note: The raw PDF sources used for RAG are available here: https://drive.google.com/drive/folders/1tEAK9_1C3ixIEyigld873FE6Rga205pL?usp=sharing*
