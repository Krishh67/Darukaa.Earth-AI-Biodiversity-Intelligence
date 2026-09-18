# Darukaa.Earth AI Biodiversity Intelligence

This is a production-quality hackathon prototype for the Darukaa.Earth AI Biodiversity Challenge. It integrates Conversational AI, Scientific RAG (FAISS + BM25 + RRF), structured environmental data fetching (SoilGrids, Open-Meteo, ESA WorldCover), Multi-Metric Reasoning, and Verification into a unified pipeline.

## Features
- **Modern Web UI**: Built with TailwindCSS.
- **RAG Pipeline**: Fuses semantic search and keyword search, reranked by a Cross-Encoder.
- **Environmental Context**: Automatically pulls soil health, climate, and land-cover data based on exact coordinates.
- **Verification Node**: Automatically flags hallucinations and ensures the LLM's recommendation is supported by the retrieved scientific papers.

## Project Structure
- `backend/main.py` - FastAPI entry point.
- `backend/rag/` - FAISS, BM25, Fusion, and Cross-Encoder reranking.
- `backend/environmental/` - Integrations with SoilGrids, OpenMeteo, and Planetary Computer.
- `backend/reasoning/` - Gemini LLM structured outputs and verification.
- `frontend/` - HTML, JS, CSS for the frontend.

## How to Run
1. Ensure your `.env` file contains your active `GEMINI_API_KEY`.
2. Start the Backend API Server:
   ```bash
   python -m backend.main
   ```
   (The server will start on `http://0.0.0.0:8000`)
3. Open the UI:
   Navigate to the `frontend/` folder in your file explorer and double-click `index.html` to open it in your browser.

## Using the Demo
1. **MANDATORY**: From the top navigation bar, select a location from the dropdown (e.g., **Mumbai, India**) or select **Custom Lat/Long** to enter specific coordinates.
2. Ask a question like: *"How can I improve biodiversity in this degraded urban land?"*
3. The system will extract your coordinates, fetch soil properties, retrieve relevant RAG chunks, and stream the fully reasoned recommendation!

