from flask import Flask, render_template, request
import openai
from openai import OpenAI
import os
import time
import logging
import tiktoken
import asyncio
import aiohttp
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import json
from cachetools import TTLCache
import tenacity

# Vector Database and RAG imports
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Vector Database and RAG components
logger.info("Initializing Vector Database and RAG components...")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./vector_db")

# Initialize embedding model for RAG
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Embedding model loaded successfully")
except Exception as e:
    logger.warning(f"Failed to load embedding model: {e}. RAG features may be limited.")
    embedding_model = None

# Create or get collection for PRD documents
try:
    prd_collection = chroma_client.get_or_create_collection(
        name="prd_documents",
        metadata={"description": "PRD and documentation storage for RAG"}
    )
    logger.info("ChromaDB collection initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB collection: {e}")
    prd_collection = None

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Performance optimizations
THREAD_POOL_SIZE = 4
MAX_CACHE_SIZE = 1000
CACHE_TTL = 3600  # 1 hour

# Initialize thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)

# Cache for responses (in-memory with TTL)
response_cache = TTLCache(maxsize=MAX_CACHE_SIZE, ttl=CACHE_TTL)

logger.info("Flask application starting up with performance optimizations...")
logger.info(f"OpenAI API key configured: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
logger.info(f"Thread pool size: {THREAD_POOL_SIZE}")
logger.info(f"Cache size: {MAX_CACHE_SIZE} items, TTL: {CACHE_TTL}s")

@app.route("/prd_parser", methods=["GET"])
def user_story_input():
    logger.info("GET request received for user story input page")
    try:
        logger.info("Rendering poc2_user_story_input.html template")
        return render_template("poc2_user_story_input.html")
    except Exception as e:
        logger.error(f"Error rendering user story input template: {str(e)}")
        raise

@app.route("/user-story-upload", methods=["POST"])
def process_user_story():
    start_time = time.time()
    logger.info("POST request received for user story processing")
    
    # Collect input
    logger.info("Collecting form data and file uploads")
    context = request.form.get("context", "")
    prd_file = request.files.get("prd_file")
    additional_docs = request.files.get("additional_docs")
    
    logger.info(f"Context provided: {'Yes' if context else 'No'}")
    logger.info(f"PRD file uploaded: {'Yes' if prd_file else 'No'}")
    logger.info(f"Additional docs uploaded: {'Yes' if additional_docs else 'No'}")
    
    if prd_file:
        logger.info(f"PRD file name: {prd_file.filename}")
    if additional_docs:
        logger.info(f"Additional docs file name: {additional_docs.filename}")

    # Parallel file reading for better performance
    logger.info("Reading uploaded files in parallel")
    
    def read_files_parallel():
        with ThreadPoolExecutor(max_workers=2) as file_executor:
            prd_future = file_executor.submit(safe_read, prd_file) if prd_file else None
            docs_future = file_executor.submit(safe_read, additional_docs) if additional_docs else None
            
            prd_content = prd_future.result() if prd_future else ""
            docs_content = docs_future.result() if docs_future else ""
            
            return prd_content, docs_content    prd_content, docs_content = read_files_parallel()
    
    # Store documents in vector database and create RAG summaries
    logger.info("Processing documents with RAG enhancement")
    
    # Process PRD with RAG if content is substantial
    if prd_content and len(prd_content) > 5000:
        logger.info("Creating RAG-enhanced PRD summary")
        prd_content = create_rag_summary(prd_content, prd_file.filename if prd_file else "prd_content", max_summary_length=25000)
    elif prd_content and len(prd_content) > 30000:
        logger.info("Large PRD detected - using traditional optimization")
        prd_content = optimize_prd_content(prd_content, max_length=40000)
    
    # Process additional docs with RAG
    if docs_content and len(docs_content) > 3000:
        logger.info("Creating RAG-enhanced docs summary")
        docs_content = create_rag_summary(docs_content, additional_docs.filename if additional_docs else "additional_docs", max_summary_length=10000)
    elif docs_content and len(docs_content) > 15000:
        logger.info("Large additional docs detected - truncating for faster processing")
        docs_content = docs_content[:15000] + "...\n[Content truncated for performance]"
    
    logger.info(f"PRD content length (post-RAG): {len(prd_content)} characters")
    logger.info(f"Additional docs content length (post-RAG): {len(docs_content)} characters")

    # Combine prompt
    logger.info("Combining context, PRD, and additional docs into prompt")
    prompt = f"{context}\n\nPRD:\n{prd_content}\n\nAdditional Docs:\n{docs_content}"
    logger.info(f"Final prompt length: {len(prompt)} characters")
      # Check cache first with more granular caching
    prompt_hash = get_cache_key(prompt)
    cached_result = response_cache.get(prompt_hash)
    if cached_result:
        logger.info("Cache hit! Returning cached response")
        processing_time = time.time() - start_time
        logger.info(f"Total processing time (cached): {processing_time:.2f} seconds")
        return render_template("poc2_epic_story_screen.html", epics=cached_result)
    
    # Check if we can use chunked processing for large prompts
    should_chunk = prompt_tokens > 50000  # Chunk if over 50k tokens
    if should_chunk:
        logger.info(f"Large prompt detected ({prompt_tokens:,} tokens). Using chunked processing for better performance.")
        return process_large_prompt_chunked(prompt, start_time)
    
    # Log token count for the initial prompt
    prompt_tokens = count_tokens(prompt, "gpt-4o")
    logger.info(f"Initial prompt token count: {prompt_tokens:,} tokens")
    
    # Check if prompt is within token limits
    if prompt_tokens > 120000:
        logger.warning(f"Prompt token count ({prompt_tokens:,}) is approaching the 128k limit!")
    else:
        logger.info(f"Prompt is within safe token limits ({prompt_tokens:,}/128,000 tokens)")

    # Parallel assistant processing for maximum performance
    logger.info("****************Starting parallel assistant interactions")
    
    try:
        # Process both assistants in parallel using thread pool
        with ThreadPoolExecutor(max_workers=2) as assistant_executor:
            # Start first assistant
            future_1 = assistant_executor.submit(ask_assistant_from_file_optimized, "poc2_agent1_prd_parser", prompt)
            
            # Wait for first assistant to complete
            response_1 = future_1.result()
            logger.info("################First assistant response received")
            
            # Log token usage for first interaction
            log_token_usage(prompt, response_1, model="gpt-4o", context="First Assistant (PRD Parser)")
            
            # Start second assistant immediately
            logger.info("Starting second assistant interaction")
            future_2 = assistant_executor.submit(ask_assistant_from_file_optimized, "poc2_agent2_epic_generator", response_1)
            
            response_2 = future_2.result()
            logger.info("Second assistant response received")
            
            # Log token usage for second interaction
            log_token_usage(response_1, response_2, model="gpt-4o", context="Second Assistant (Epic Generator)")

        final_output = response_2
        logger.info(f"Final output length: {len(final_output)} characters")
        
        # Cache the result for future requests
        response_cache[prompt_hash] = final_output
        logger.info("Response cached for future requests")
        
        processing_time = time.time() - start_time
        logger.info(f"Total processing time: {processing_time:.2f} seconds")

        # Render page 2 with response
        logger.info("Rendering epic story screen with final output")
        return render_template("poc2_epic_story_screen.html", epics=final_output)
        
    except Exception as e:
        logger.error(f"Error in parallel processing: {str(e)}")
        raise

def process_large_prompt_chunked(prompt, start_time):
    """Process large prompts in chunks for better performance."""
    logger.info("Processing large prompt using chunked approach")
    
    # Split prompt into sections
    sections = prompt.split('\n\n')
    context = sections[0] if sections else ""
    prd_section = ""
    docs_section = ""
    
    # Extract PRD and Docs sections
    for i, section in enumerate(sections):
        if 'PRD:' in section:
            prd_section = '\n\n'.join(sections[i:i+10])  # Take next 10 sections for PRD
            break
    
    for i, section in enumerate(sections):
        if 'Additional Docs:' in section:
            docs_section = '\n\n'.join(sections[i:])  # Take remaining for docs
            break
    
    # Create optimized prompt with most important content first
    optimized_prompt = f"{context}\n\nPRD (Key Sections):\n{prd_section[:30000]}\n\nAdditional Context:\n{docs_section[:10000]}"
    
    logger.info(f"Optimized prompt length: {len(optimized_prompt)} characters (reduced from {len(prompt)})")
    
    try:
        # Use faster processing for optimized prompt
        response_1 = ask_assistant_from_file_fast("poc2_agent1_prd_parser", optimized_prompt)
        logger.info("First assistant (chunked) response received")
        
        response_2 = ask_assistant_from_file_fast("poc2_agent2_epic_generator", response_1)
        logger.info("Second assistant (chunked) response received")
        
        final_output = response_2
        
        # Cache the result
        prompt_hash = get_cache_key(prompt)
        response_cache[prompt_hash] = final_output
        
        processing_time = time.time() - start_time
        logger.info(f"Total chunked processing time: {processing_time:.2f} seconds")
        
        return render_template("poc2_epic_story_screen.html", epics=final_output)
        
    except Exception as e:
        logger.error(f"Error in chunked processing: {str(e)}")
        raise

def get_cache_key(prompt):
    """Generate a cache key from the prompt."""
    return hashlib.md5(prompt.encode('utf-8')).hexdigest()

def store_document_in_vector_db(content, filename, doc_type="prd"):
    """Store document content in vector database for RAG retrieval."""
    if not prd_collection or not embedding_model:
        logger.warning("Vector DB or embedding model not available. Skipping storage.")
        return None
    
    try:
        logger.info(f"Storing {doc_type} document in vector database: {filename}")
        
        # Split content into chunks for better retrieval
        chunks = split_document_into_chunks(content, chunk_size=1000, overlap=200)
        logger.info(f"Document split into {len(chunks)} chunks")
        
        # Generate embeddings for each chunk
        chunk_embeddings = embedding_model.encode(chunks)
        
        # Create metadata for each chunk
        documents = []
        metadatas = []
        ids = []
        
        doc_id = hashlib.md5(f"{filename}_{doc_type}".encode()).hexdigest()
        
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            chunk_id = f"{doc_id}_chunk_{i}"
            
            documents.append(chunk)
            metadatas.append({
                "filename": filename,
                "doc_type": doc_type,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "timestamp": datetime.now().isoformat()
            })
            ids.append(chunk_id)
        
        # Store in ChromaDB
        prd_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Successfully stored {len(chunks)} chunks in vector database")
        return doc_id
        
    except Exception as e:
        logger.error(f"Error storing document in vector DB: {str(e)}")
        return None

def split_document_into_chunks(content, chunk_size=1000, overlap=200):
    """Split document content into overlapping chunks for better retrieval."""
    if len(content) <= chunk_size:
        return [content]
    
    chunks = []
    start = 0
    
    while start < len(content):
        end = start + chunk_size
        
        # Try to break at sentence boundaries
        if end < len(content):
            # Look for sentence endings within overlap range
            for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                last_punct = content.rfind(punct, start, end)
                if last_punct > start:
                    end = last_punct + len(punct)
                    break
        
        chunk = content[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        
        # Avoid infinite loop
        if start >= end:
            start = end
    
    return chunks

def retrieve_relevant_content(query, top_k=5, doc_type=None):
    """Retrieve relevant content from vector database using RAG."""
    if not prd_collection or not embedding_model:
        logger.warning("Vector DB or embedding model not available. Skipping RAG retrieval.")
        return []
    
    try:
        logger.info(f"Retrieving relevant content for query (top {top_k} results)")
        
        # Create query filter if doc_type is specified
        where_filter = {"doc_type": doc_type} if doc_type else None
        
        # Query the collection
        results = prd_collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter
        )
        
        relevant_chunks = []
        if results['documents'] and results['documents'][0]:
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                relevant_chunks.append({
                    'content': doc,
                    'metadata': metadata,
                    'relevance_rank': i + 1
                })
                
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")
        return relevant_chunks
        
    except Exception as e:
        logger.error(f"Error retrieving from vector DB: {str(e)}")
        return []

def create_rag_summary(content, filename, max_summary_length=5000):
    """Create a RAG-enhanced summary of the document content."""
    logger.info(f"Creating RAG-enhanced summary for {filename}")
    
    # Store the document in vector DB
    doc_id = store_document_in_vector_db(content, filename)
    
    if not doc_id:
        # Fallback to simple truncation if vector DB fails
        logger.warning("Vector DB storage failed, using simple truncation")
        return content[:max_summary_length] + "..." if len(content) > max_summary_length else content
    
    # Create summary queries to extract key information
    summary_queries = [
        "requirements and functional specifications",
        "user stories and acceptance criteria", 
        "business objectives and goals",
        "system constraints and dependencies",
        "key features and capabilities"
    ]
    
    # Retrieve relevant content for each query
    summary_parts = []
    for query in summary_queries:
        relevant_chunks = retrieve_relevant_content(query, top_k=2, doc_type="prd")
        
        if relevant_chunks:
            # Take the most relevant chunk for each query
            best_chunk = relevant_chunks[0]['content']
            summary_parts.append(f"[{query.title()}]\n{best_chunk}\n")
   