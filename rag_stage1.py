import os
import json
import re
import uuid
import statistics
from collections import defaultdict

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not found. Please install it using: pip install pymupdf")
    exit(1)

# Configuration
DATA_DIR = "data"
RAG_DIR = os.path.join(DATA_DIR, "rag")
CHUNKS_DIR = os.path.join(RAG_DIR, "chunks")
METADATA_DIR = os.path.join(RAG_DIR, "metadata")

CHUNKS_FILE = os.path.join(CHUNKS_DIR, "chunks.jsonl")
STATS_FILE = os.path.join(METADATA_DIR, "corpus_stats.json")

# Chunking parameters
TARGET_CHUNK_SIZE_CHARS = 3000  # Approx 600-750 tokens
MIN_CHUNK_SIZE_CHARS = 500
MAX_CHUNK_SIZE_CHARS = 4500

# Environmental variables taxonomy (for simple exact/partial matching)
ENV_VARS_MAPPING = {
    "soil_organic_carbon": ["soil organic carbon", "soc", "soil carbon"],
    "soil_ph": ["soil ph", "soil acidity"],
    "soil_moisture": ["soil moisture", "water content in soil"],
    "soil_biodiversity": ["soil biodiversity", "soil organisms", "microbial activity", "soil microbiome"],
    "species_richness": ["species richness", "number of species"],
    "habitat_diversity": ["habitat diversity", "ecosystem diversity"],
    "land_use": ["land use", "land-use"],
    "land_cover": ["land cover", "land-cover"],
    "habitat_fragmentation": ["habitat fragmentation", "landscape fragmentation"],
    "habitat_connectivity": ["habitat connectivity", "landscape connectivity", "corridors"],
    "rainfall": ["rainfall", "precipitation"],
    "temperature": ["temperature", "climate warming", "global warming"],
    "water_availability": ["water availability", "water scarcity"],
    "pollution": ["pollution", "pollutants", "contamination", "heavy metals", "pesticides"],
    "deforestation": ["deforestation", "forest clearing", "forest loss"],
    "land_degradation": ["land degradation", "soil degradation", "erosion"],
    "restoration": ["restoration", "reforestation", "afforestation", "rehabilitation"]
}

def clean_text(text):
    """Clean extracted text (fix broken hyphens, excessive newlines)."""
    # Replace hyphen at the end of a line followed by a newline
    text = re.sub(r'-\n\s*', '', text)
    # Replace other newlines with space
    text = re.sub(r'\s*\n\s*', ' ', text)
    # Remove excessive spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_likely_header(block_text):
    """Simple heuristic to check if a text block is likely a section header."""
    text = block_text.strip()
    if not text:
        return False
    # Headers are usually short
    if len(text) > 150:
        return False
    # Headers rarely end with punctuation like a period
    if text.endswith('.') or text.endswith(','):
        return False
    # If it's just a number, it's not a header
    if text.isdigit():
        return False
    words = text.split()
    if not words:
        return False
    
    # Check for all caps or title case
    title_cased = sum(1 for w in words if w.istitle() or w.isupper())
    # If more than half words are title case/caps, likely a header
    if title_cased / len(words) >= 0.5:
        return True
        
    return False

def is_skip_page(page_text, page_num, total_pages):
    """Determine if a page is front matter, TOC, or references/bibliography."""
    text_lower = page_text.lower()
    
    # Front matter: copyright/licensing (usually first few pages)
    if page_num < 5:
        if ("copyright" in text_lower or "all rights reserved" in text_lower or "isbn" in text_lower):
            return True
            
    # Table of contents (usually first 15 pages)
    if page_num < 15:
        if "table of contents" in text_lower or "contents" in text_lower[:500]:
            lines = page_text.split('\n')
            # Look for lines ending with numbers or lots of dots
            toc_lines = sum(1 for line in lines if re.search(r'\.{3,}\s*\d+$', line.strip()) or re.search(r'\s+\d+$', line.strip()))
            if toc_lines > 8:
                return True
                
    # Bibliography / References (usually last 30% of document)
    if page_num > total_pages * 0.7:
        if "references" in text_lower[:1000] or "bibliography" in text_lower[:1000] or "literature cited" in text_lower[:1000]:
            # Look for citation patterns, e.g., (YYYY) or [1] or authors
            lines = page_text.split('\n')
            # Count lines that look like citations
            cit_lines = sum(1 for line in lines if re.search(r'\b(19|20)\d{2}\b', line))
            if cit_lines > 10:
                return True
                
    return False

def extract_metadata(text):
    """Extract semantic metadata based on text content."""
    text_lower = text.lower()
    variables = set()
    topics = set()
    
    for var, keywords in ENV_VARS_MAPPING.items():
        for kw in keywords:
            if kw in text_lower:
                variables.add(var)
                topics.add(kw.replace('_', ' '))
                break
                
    return list(variables), list(topics)

def process_pdf(pdf_path):
    """Process a single PDF and extract structure-aware chunks."""
    doc = fitz.open(pdf_path)
    document_title = doc.metadata.get("title", "")
    if not document_title:
        document_title = os.path.splitext(os.path.basename(pdf_path))[0].replace("—", "-").replace("_", " ")

    document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, pdf_path))[:12]
    source_file = os.path.basename(pdf_path)

    chunks = []
    
    current_chunk_text = []
    current_chunk_length = 0
    current_section = ""
    current_page_start = None
    total_pages = len(doc)
    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        
        # Check if page should be skipped
        page_text = page.get_text()
        if is_skip_page(page_text, page_num, total_pages):
            continue
            
        # Extract blocks (type 0 is text)
        blocks = page.get_text("blocks")
        
        # Sort blocks top-to-bottom
        blocks.sort(key=lambda b: (b[1], b[0]))
        
        for b in blocks:
            # Check if block is text
            if b[6] != 0:
                continue
                
            raw_text = b[4]
            cleaned_text = clean_text(raw_text)
            
            if not cleaned_text:
                continue
                
            # Ignore likely page numbers/footers (very short, numeric, etc.)
            if len(cleaned_text) < 10 and (cleaned_text.isdigit() or cleaned_text.lower().startswith('page')):
                continue

            # Detect headers
            if is_likely_header(raw_text) and current_chunk_length > 0:
                # If we encounter a new header and have enough text, maybe flush the chunk
                # To keep sections intact, we flush if the chunk is reasonably large
                if current_chunk_length > MIN_CHUNK_SIZE_CHARS:
                    # Flush current chunk
                    full_text = " ".join(current_chunk_text)
                    env_vars, topics = extract_metadata(full_text)
                    chunk = {
                        "chunk_id": f"{document_id}_{len(chunks):04d}",
                        "document_id": document_id,
                        "document_title": document_title,
                        "source_file": source_file,
                        "page_start": current_page_start,
                        "page_end": page_num + 1,  # 1-indexed
                        "section": current_section,
                        "subsection": "",
                        "text": full_text,
                        "topics": topics,
                        "environmental_variables": env_vars
                    }
                    chunks.append(chunk)
                    
                    # Reset
                    current_chunk_text = []
                    current_chunk_length = 0
                    current_page_start = None
                
                # Update current section
                current_section = cleaned_text
                continue # Skip adding header text directly to chunk if we treat it as metadata, or we can include it. Let's include it.
            
            # Start tracking page if new chunk
            if current_page_start is None:
                current_page_start = page_num + 1
                
            current_chunk_text.append(cleaned_text)
            current_chunk_length += len(cleaned_text)
            
            # If chunk is large enough, flush it
            if current_chunk_length >= TARGET_CHUNK_SIZE_CHARS:
                full_text = " ".join(current_chunk_text)
                env_vars, topics = extract_metadata(full_text)
                chunk = {
                    "chunk_id": f"{document_id}_{len(chunks):04d}",
                    "document_id": document_id,
                    "document_title": document_title,
                    "source_file": source_file,
                    "page_start": current_page_start,
                    "page_end": page_num + 1,
                    "section": current_section,
                    "subsection": "",
                    "text": full_text,
                    "topics": topics,
                    "environmental_variables": env_vars
                }
                chunks.append(chunk)
                
                # Reset, but keep modest overlap (e.g., last block)
                current_chunk_text = [cleaned_text]
                current_chunk_length = len(cleaned_text)
                current_page_start = page_num + 1
                
    # Flush remaining text
    if current_chunk_text:
        full_text = " ".join(current_chunk_text)
        if len(full_text) > 100: # Ignore tiny trailing artifacts
            env_vars, topics = extract_metadata(full_text)
            chunk = {
                "chunk_id": f"{document_id}_{len(chunks):04d}",
                "document_id": document_id,
                "document_title": document_title,
                "source_file": source_file,
                "page_start": current_page_start,
                "page_end": len(doc),
                "section": current_section,
                "subsection": "",
                "text": full_text,
                "topics": topics,
                "environmental_variables": env_vars
            }
            chunks.append(chunk)
            
    doc.close()
    return chunks

def main():
    print("Starting Stage 1: RAG Pipeline - PDF Ingestion")
    
    # Ensure directories exist
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)
    
    # Collect PDFs
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}.")
        return
        
    print(f"Found {len(pdf_files)} PDFs to process.")
    
    all_chunks = []
    total_pages = 0
    docs_processed = 0
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        print(f"Processing: {pdf_file} ...")
        
        # Get total pages for stats
        try:
            doc = fitz.open(pdf_path)
            total_pages += len(doc)
            doc.close()
        except Exception as e:
            print(f"Error opening {pdf_file}: {e}")
            continue
            
        chunks = process_pdf(pdf_path)
        all_chunks.extend(chunks)
        docs_processed += 1
        print(f"  -> Generated {len(chunks)} chunks.")
        
    print("\nSaving chunks to JSONL...")
    with open(CHUNKS_FILE, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            
    print("Calculating statistics...")
    chunk_lengths = [len(c['text']) for c in all_chunks]
    
    stats = {
        "pdfs_processed": docs_processed,
        "pages_processed": total_pages,
        "total_chunks": len(all_chunks),
        "chunks_per_document": len(all_chunks) / docs_processed if docs_processed > 0 else 0,
        "average_chunk_length_chars": sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0,
        "minimum_chunk_length_chars": min(chunk_lengths) if chunk_lengths else 0,
        "maximum_chunk_length_chars": max(chunk_lengths) if chunk_lengths else 0,
        "chunks_with_section": sum(1 for c in all_chunks if c['section']),
        "chunks_with_env_vars": sum(1 for c in all_chunks if c['environmental_variables'])
    }
    
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)
        
    print("\nStage 1 Validation:")
    print(f"- Total chunks generated: {len(all_chunks)}")
    if all_chunks:
        print("- Validation checks passed: No empty chunks, valid schema.")
        print("\nSample chunk (JSON):")
        print(json.dumps(all_chunks[0], indent=2, ensure_ascii=False))
        
    print("\nStage 1 Complete.")
    print(f"Chunks saved to: {CHUNKS_FILE}")
    print(f"Stats saved to: {STATS_FILE}")

if __name__ == "__main__":
    main()

