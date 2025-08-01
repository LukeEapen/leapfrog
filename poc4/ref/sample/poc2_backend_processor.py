from flask import Flask, render_template, request
import openai
from openai import OpenAI
import os
import time
import logging
import tiktoken
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
from cachetools import TTLCache
import tenacity

# Vector Database and RAG imports (with error handling)
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available. Install with: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("SentenceTransformers not available. Install with: pip install sentence-transformers")

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

# Performance optimizations
THREAD_POOL_SIZE = 4
MAX_CACHE_SIZE = 1000
CACHE_TTL = 3600  # 1 hour

# Initialize thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)

# Cache for responses (in-memory with TTL)
response_cache = TTLCache(maxsize=MAX_CACHE_SIZE, ttl=CACHE_TTL)

# Initialize Vector Database and RAG components
logger.info("Initializing Vector Database and RAG components...")

# Initialize ChromaDB client
if CHROMADB_AVAILABLE:
    try:
        chroma_client = chromadb.PersistentClient(path="./vector_db")
        prd_collection = chroma_client.get_or_create_collection(
            name="prd_documents",
            metadata={"description": "PRD and documentation storage for RAG"}
        )
        logger.info("ChromaDB collection initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        chroma_client = None
        prd_collection = None
else:
    chroma_client = None
    prd_collection = None

# Initialize embedding model for RAG
if SENTENCE_TRANSFORMERS_AVAILABLE:
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Embedding model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load embedding model: {e}")
        embedding_model = None
else:
    embedding_model = None

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

logger.info("Flask application starting up with RAG optimization...")
logger.info(f"OpenAI API key configured: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
logger.info(f"Thread pool size: {THREAD_POOL_SIZE}")
logger.info(f"ChromaDB available: {CHROMADB_AVAILABLE}")
logger.info(f"Embedding model available: {SENTENCE_TRANSFORMERS_AVAILABLE}")

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
            
            return prd_content, docs_content
    
    prd_content, docs_content = read_files_parallel()
    
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
    prompt = f"{context}\n\nPRD (RAG-Enhanced Summary):\n{prd_content}\n\nAdditional Docs (RAG-Enhanced):\n{docs_content}"
    logger.info(f"Final prompt length: {len(prompt)} characters")
    
    # Check cache first with more granular caching
    prompt_hash = get_cache_key(prompt)
    cached_result = response_cache.get(prompt_hash)
    if cached_result:
        logger.info("Cache hit! Returning cached response")
        processing_time = time.time() - start_time
        logger.info(f"Total processing time (cached): {processing_time:.2f} seconds")
        return render_template("poc2_epic_story_screen.html", epics=cached_result)
    
    # Log token count for the initial prompt
    prompt_tokens = count_tokens(prompt, "gpt-4o")
    logger.info(f"Initial prompt token count: {prompt_tokens:,} tokens")
    
    # Check if prompt is within token limits
    if prompt_tokens > 120000:
        logger.warning(f"Prompt token count ({prompt_tokens:,}) is approaching the 128k limit!")
    else:
        logger.info(f"Prompt is within safe token limits ({prompt_tokens:,}/128,000 tokens)")

    # Process with optimized assistants
    logger.info("****************Starting RAG-enhanced assistant interactions")
    
    try:
        # Process both assistants sequentially with RAG-enhanced content
        logger.info("Starting first assistant with RAG-enhanced content")
        response_1 = ask_assistant_from_file_optimized("poc2_agent1_prd_parser", prompt)
        logger.info("################First assistant response received")
        
        # Log token usage for first interaction
        log_token_usage(prompt, response_1, model="gpt-4o", context="First Assistant (RAG-Enhanced PRD Parser)")
        
        # Start second assistant immediately
        logger.info("Starting second assistant interaction")
        response_2 = ask_assistant_from_file_optimized("poc2_agent2_epic_generator", response_1)
        logger.info("Second assistant response received")
        
        # Log token usage for second interaction
        log_token_usage(response_1, response_2, model="gpt-4o", context="Second Assistant (Epic Generator)")

        final_output = response_2
        logger.info(f"Final output length: {len(final_output)} characters")
        
        # Cache the result for future requests
        response_cache[prompt_hash] = final_output
        logger.info("Response cached for future requests")
        
        processing_time = time.time() - start_time
        logger.info(f"Total RAG-enhanced processing time: {processing_time:.2f} seconds")

        # Render page 2 with response
        logger.info("Rendering epic story screen with final output")
        return render_template("poc2_epic_story_screen.html", epics=final_output)
        
    except Exception as e:
        logger.error(f"Error in RAG-enhanced processing: {str(e)}")
        raise

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
    
    # Combine all parts into a comprehensive summary
    rag_summary = "\n".join(summary_parts)
    
    # Ensure summary is within length limits
    if len(rag_summary) > max_summary_length:
        rag_summary = rag_summary[:max_summary_length] + "\n[Summary truncated for performance]"
    
    logger.info(f"RAG summary created: {len(rag_summary)} characters")
    return rag_summary

def get_cache_key(prompt):
    """Generate a cache key from the prompt."""
    return hashlib.md5(prompt.encode('utf-8')).hexdigest()

def safe_read(file):
    try:
        logger.debug(f"Attempting to read file: {file.filename if file else 'None'}")
        content = file.read().decode("utf-8")
        logger.debug(f"Successfully read file: {file.filename}, length: {len(content)} characters")
        return content
    except UnicodeDecodeError as e:
        logger.warning(f"Failed to decode file {file.filename} as UTF-8: {str(e)}")
        return "[Unable to decode file as UTF-8 text]"
    except Exception as e:
        logger.error(f"Error reading file {file.filename}: {str(e)}")
        return "[Unable to read file]"

def optimize_prd_content(prd_content, max_length=40000):
    """Optimize PRD content by extracting key sections and reducing verbosity."""
    if len(prd_content) <= max_length:
        return prd_content
    
    logger.info(f"Optimizing PRD content from {len(prd_content)} to ~{max_length} characters")
    
    # Split into sections and prioritize important ones
    sections = prd_content.split('\n\n')
    important_keywords = [
        'requirement', 'feature', 'user story', 'acceptance criteria', 
        'functional', 'non-functional', 'business rule', 'constraint',
        'objective', 'goal', 'scope', 'assumption', 'dependency'
    ]
    
    # Score sections based on importance
    scored_sections = []
    for section in sections:
        score = 0
        section_lower = section.lower()
        
        # Score based on keywords
        for keyword in important_keywords:
            score += section_lower.count(keyword) * 10
        
        # Prefer sections with structured content
        if any(marker in section for marker in ['•', '-', '1.', '2.', '*']):
            score += 5
        
        # Prefer shorter, more concise sections
        if len(section) < 500:
            score += 3
        elif len(section) > 2000:
            score -= 2
        
        scored_sections.append((score, section))
    
    # Sort by score and take top sections
    scored_sections.sort(key=lambda x: x[0], reverse=True)
    
    optimized_content = ""
    for score, section in scored_sections:
        if len(optimized_content) + len(section) <= max_length:
            optimized_content += section + "\n\n"
        else:
            # Add partial section if there's room
            remaining_space = max_length - len(optimized_content)
            if remaining_space > 200:  # Only add if meaningful space left
                optimized_content += section[:remaining_space-10] + "..."
            break
    
    logger.info(f"PRD content optimized to {len(optimized_content)} characters")
    return optimized_content.strip()

client = OpenAI()

# Cached token counting for better performance
@lru_cache(maxsize=1000)
def count_tokens(text, model="gpt-4o"):
    """Count the number of tokens in a text string for a given model (cached)."""
    try:
        if "gpt-4" in model:
            encoding = tiktoken.encoding_for_model("gpt-4")
        elif "gpt-3.5" in model:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        logger.warning(f"Error counting tokens: {str(e)}")
        return len(text) // 4

def log_token_usage(prompt_text, response_text, model="gpt-4o", context=""):
    """Log detailed token usage information."""
    prompt_tokens = count_tokens(prompt_text, model)
    response_tokens = count_tokens(response_text, model)
    total_tokens = prompt_tokens + response_tokens
    
    logger.info(f"TOKEN USAGE {context}:")
    logger.info(f"  Model: {model}")
    logger.info(f"  Prompt tokens: {prompt_tokens:,}")
    logger.info(f"  Response tokens: {response_tokens:,}")
    logger.info(f"  Total tokens: {total_tokens:,}")
    
    if "gpt-4o" in model:
        input_cost = prompt_tokens * 0.000005
        output_cost = response_tokens * 0.000015
        total_cost = input_cost + output_cost
    else:
        total_cost = total_tokens * 0.00003
    
    logger.info(f"  Estimated cost: ${total_cost:.4f}")
    
    return {
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": total_tokens,
        "model": model,
        "estimated_cost": total_cost
    }

# Retry decorator for API calls
@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    stop=tenacity.stop_after_attempt(3),
    retry=tenacity.retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError))
)
def ask_assistant_from_file_optimized(code_filepath, user_prompt):
    start_time = time.time()
    logger.info(f"Creating RAG-enhanced assistant from file: {code_filepath}")
    
    # Log input token count
    input_tokens = count_tokens(user_prompt, "gpt-4o")
    logger.info(f"Input prompt token count: {input_tokens:,} tokens")
    
    # Construct the full path to the file in the agents folder
    full_filepath = os.path.join("agents", code_filepath)
    logger.info(f"Full file path: {full_filepath}")
    
    # Check if file exists
    if not os.path.exists(full_filepath):
        logger.error(f"File not found: {full_filepath}")
        raise FileNotFoundError(f"Assistant file not found: {full_filepath}")
    
    logger.info(f"File exists, size: {os.path.getsize(full_filepath)} bytes")
    
    try:
        # Upload the file
        logger.info(f"Uploading file to OpenAI: {full_filepath}")
        with open(full_filepath, "rb") as f:
            uploaded_file = client.files.create(file=f, purpose="assistants")
        logger.info(f"File uploaded successfully: {uploaded_file.id}")
        
        # Create assistant with RAG-optimized settings
        logger.info("Creating RAG-enhanced OpenAI assistant")
        assistant = client.beta.assistants.create(
            name="RAG Enhanced Assistant",
            model="gpt-4o",
            temperature=0.0,  # Set to 0 for fastest response
            instructions="You are processing RAG-enhanced content that has been intelligently summarized. The input contains the most relevant sections extracted using semantic search. Focus on the key requirements, features, and user stories. Provide concise, well-structured responses.",
            tools=[{"type": "code_interpreter"}],
            tool_resources={
                "code_interpreter": {
                    "file_ids": [uploaded_file.id]
                }
            }
        )
        logger.info(f"RAG-enhanced assistant created: {assistant.id}")

        # Create thread
        logger.info("Creating thread for RAG-enhanced interaction")
        thread = client.beta.threads.create()
        logger.info(f"Thread created: {thread.id}")
        
        logger.info("Adding RAG-enhanced message to thread")
        client.beta.threads.messages.create(thread_id=thread.id, role="user", content=user_prompt)
        logger.info(f"User message added (length: {len(user_prompt)} characters)")

        # Run assistant with optimized polling
        logger.info("Starting RAG-enhanced assistant run")
        run = client.beta.threads.runs.create(assistant_id=assistant.id, thread_id=thread.id)
        logger.info(f"Run created: {run.id}")

        logger.info("Polling for completion with RAG optimization")
        poll_count = 0
        poll_interval = 0.2  # Start with very fast polling
        timeout_seconds = 90  # Shorter timeout for RAG-enhanced content
        start_poll_time = time.time()
        
        while True:
            poll_count += 1
            
            # Check for timeout
            elapsed_time = time.time() - start_poll_time
            if elapsed_time > timeout_seconds:
                logger.error(f"RAG-enhanced assistant timed out after {timeout_seconds} seconds")
                raise RuntimeError(f"Assistant run timed out after {timeout_seconds} seconds")
            
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            logger.debug(f"Poll #{poll_count}: Status = {run.status} (elapsed: {elapsed_time:.1f}s)")
            
            if run.status == "succeeded":
                logger.info(f"RAG-enhanced run completed after {poll_count} polls in {elapsed_time:.2f} seconds")
                break
            elif run.status in ("failed", "cancelled"):
                logger.error(f"RAG-enhanced run failed: {run.status}")
                raise RuntimeError(f"Run failed with status: {run.status}")
            
            time.sleep(poll_interval)
            # Adaptive polling for RAG content
            if poll_count > 8:
                poll_interval = min(poll_interval * 1.2, 1.0)

        # Get the response
        logger.info("Retrieving RAG-enhanced response")
        messages = client.beta.threads.messages.list(thread_id=thread.id, limit=1).data
        response_content = messages[0].content[0].text.value
        logger.info(f"RAG-enhanced response received, length: {len(response_content)} characters")
        
        # Log output token count
        output_tokens = count_tokens(response_content, "gpt-4o")
        logger.info(f"Output response token count: {output_tokens:,} tokens")
        total_tokens = input_tokens + output_tokens
        logger.info(f"Total tokens used in RAG interaction: {total_tokens:,} tokens")
        
        processing_time = time.time() - start_time
        logger.info(f"RAG-enhanced assistant processing time: {processing_time:.2f} seconds")
        
        return response_content
        
    except Exception as e:
        logger.error(f"RAG-enhanced assistant error: {str(e)}")
        raise


if __name__ == "__main__":
    # Enable threaded mode for better performance
    app.run(debug=True, port=5000, threaded=True)
