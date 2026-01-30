from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
try:
    import faiss
except ImportError:
    faiss = None
    print("Warning: faiss module not found. Index search will be disabled.")
import pickle
import os
import sys
import numpy as np
# Use local embeddings implementation to avoid crashing sentence-transformers
from local_embeddings import LocalSentenceTransformer as SentenceTransformer
from llama_cpp import Llama
from contextlib import asynccontextmanager
import json
import requests

# Import Configuration
try:
    from config import (
        DATA_FILES,
        INDEX_FOLDER,
        INDEX_FILE,
        METADATA_FILE,
        EMBEDDING_MODEL_NAME,
        LLM_MODEL_PATH,
        CTX_MAX_TOKENS,
        N_GPU_LAYERS,
        TOP_K_RETRIEVAL,
        SIMILARITY_THRESHOLD,
        GENERATION_MAX_TOKENS,
        TEMPERATURE,
        PROMPT_STYLE,
        get_system_prompt
    )
except ImportError:
    # Fallback if config.py is not found
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import (
        DATA_FILES,
        INDEX_FOLDER,
        INDEX_FILE,
        METADATA_FILE,
        EMBEDDING_MODEL_NAME,
        LLM_MODEL_PATH,
        CTX_MAX_TOKENS,
        N_GPU_LAYERS,
        TOP_K_RETRIEVAL,
        SIMILARITY_THRESHOLD,
        GENERATION_MAX_TOKENS,
        TEMPERATURE,
        PROMPT_STYLE,
        get_system_prompt
    )

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RASA_URL = os.environ.get("RASA_SERVER_URL", "http://localhost:5005/model/parse")

# Global variables
models = {}
MOCK_MODE = False  # Enabled real models with fallback

def construct_prompt(query, context_str, style=PROMPT_STYLE):
    system_prompt = get_system_prompt()
    
    if style == "chatml": # Qwen, OpenHermes
        return f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
Context:
{context_str}

Question: {query}
<|im_end|>
<|im_start|>assistant
"""
    elif style == "llama2": # Llama 2 Chat
        return f"""[INST] <<SYS>>
{system_prompt}
<</SYS>>

Context:
{context_str}

Question: {query} [/INST]"""
    elif style == "alpaca": # Alpaca / Generic
        return f"""### Instruction:
{system_prompt}

Context:
{context_str}

### Input:
{query}

### Response:
"""
    else: # Plain text or default
        return f"""System: {system_prompt}

Context:
{context_str}

User: {query}

Assistant:"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MOCK_MODE:
        # Load Embedding Model
        print(f"Loading embedding model ({EMBEDDING_MODEL_NAME})...")
        # Use full model name for transformers compatibility
        models['embedding'] = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # Load topics
        allowed_topics = []
        restricted_topics = []
        # Use configured data files
        for path in DATA_FILES:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    scope = data.get("chatbot_scope", {})
                    allowed_topics.extend(scope.get("allowed_topics", []))
                    restricted_topics.extend(scope.get("restricted_topics", []))
            except Exception:
                pass
        allowed_topics = list({t.strip(): None for t in allowed_topics}.keys())
        # Manual additions for better semantic matching
        allowed_topics.extend(["Greetings", "Hello", "Hi", "PSG", "Prime Source Global", "Who are you", "Bot identity", "General chat"])
        restricted_topics = list({t.strip(): None for t in restricted_topics}.keys())
        models["allowed_topics"] = allowed_topics
        models["restricted_topics"] = restricted_topics
        
        if allowed_topics:
            a_emb = models["embedding"].encode(allowed_topics)
            a_emb = a_emb / np.linalg.norm(a_emb, axis=1, keepdims=True)
            models["allowed_topic_embeddings"] = a_emb.astype("float32")
        else:
            models["allowed_topic_embeddings"] = None
            
        if restricted_topics:
            r_emb = models["embedding"].encode(restricted_topics)
            r_emb = r_emb / np.linalg.norm(r_emb, axis=1, keepdims=True)
            models["restricted_topic_embeddings"] = r_emb.astype("float32")
        else:
            models["restricted_topic_embeddings"] = None
        
        # Load Metadata
        print("Loading metadata...")
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'rb') as f:
                metadata = pickle.load(f)
                models['metadata'] = metadata
                # Check for model mismatch
                if isinstance(metadata, dict) and 'model_name' in metadata:
                    if metadata['model_name'] != EMBEDDING_MODEL_NAME:
                        print(f"WARNING: Metadata was created with {metadata['model_name']} but current config is {EMBEDDING_MODEL_NAME}. Please re-run ingest.py!")
        else:
            print(f"Warning: Metadata file not found at {METADATA_FILE}")
            models['metadata'] = None

        # Load FAISS Index or Fallback
        print("Loading Index...")
        models['index'] = None
        models['doc_embeddings'] = None
        
        if faiss and os.path.exists(INDEX_FILE):
            try:
                models['index'] = faiss.read_index(INDEX_FILE)
                print("FAISS index loaded successfully.")
            except Exception as e:
                print(f"Warning: Failed to load FAISS index: {e}")
        
        if models['index'] is None and models['metadata']:
            print("Fallback: Computing in-memory embeddings for retrieval...")
            try:
                texts = models['metadata']['texts']
                embeddings = models['embedding'].encode(texts)
                # Normalize
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                models['doc_embeddings'] = embeddings.astype('float32')
                print(f"Computed embeddings for {len(texts)} documents.")
            except Exception as e:
                print(f"Error computing fallback embeddings: {e}")

        # Load LLM
        print(f"Loading LLM from {LLM_MODEL_PATH}...")
        if os.path.exists(LLM_MODEL_PATH):
            models['llm'] = Llama(
                model_path=LLM_MODEL_PATH, 
                n_ctx=CTX_MAX_TOKENS, 
                n_gpu_layers=N_GPU_LAYERS, 
                verbose=False
            ) 
        else:
            print(f"Warning: Model file not found at {LLM_MODEL_PATH}")
            models['llm'] = None
    else:
        print("Running in MOCK MODE. Skipping heavy model loading.")
        models['embedding'] = None
        models['index'] = None
        models['llm'] = None
        models["allowed_topics"] = ["Greetings", "Hello"]
        models["allowed_topic_embeddings"] = None
        models["restricted_topic_embeddings"] = None

    print("Startup complete.")
    yield
    # Clean up if needed
    models.clear()

app = FastAPI(title="PSGBiz RAG Bot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list

def _rasa_intent(query: str):
    try:
        resp = requests.post(RASA_URL, json={"text": query}, timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            intent = (data.get("intent") or {}).get("name")
            conf = (data.get("intent") or {}).get("confidence", 0.0)
            return intent, conf
    except Exception:
        return None, 0.0
    return None, 0.0

def _semantic_guard(query: str):
    if MOCK_MODE:
        return True, 1.0, 0.0
        
    qe = models["embedding"].encode([query])
    qe = qe / np.linalg.norm(qe, axis=1, keepdims=True)
    qe = qe.astype("float32")
    allowed_emb = models.get("allowed_topic_embeddings")
    restricted_emb = models.get("restricted_topic_embeddings")
    allowed_score = -1.0
    restricted_score = -1.0
    if allowed_emb is not None and allowed_emb.shape[0] > 0:
        allowed_score = float(np.max(qe @ allowed_emb.T))
    if restricted_emb is not None and restricted_emb.shape[0] > 0:
        restricted_score = float(np.max(qe @ restricted_emb.T))
    # Lowered threshold to 0.30 to allow broader matching for valid queries
    is_allowed = allowed_score >= 0.30 and allowed_score >= restricted_score
    return is_allowed, allowed_score, restricted_score

@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    query = request.query
    
    intent, confidence = _rasa_intent(query)
    
    # Defined explicitly 'by user requirement
    OUT_OF_DOMAIN_MSG = "This question is outside the assistant’s scope. Please ask about Company Overview, Services and Offerings, Digital Health, Construction and Interior Supplies, Environmental Consultancy, Software and AI Solutions, Travel and Tourism, Careers, Contact Information, Partners, and Business Model, as well as technology platforms such as e-Trainia, LMS, ERP, EdTech solutions, mobile applications, features, pricing, support, and integrations. You may also ask about training and workforce development including training institutes, skill development agencies, corporate learning and development, HR and payroll, along with business operations, finance, and marketing. Additionally, queries related to construction, timber, and interior solutions such as timber products, sawn timber, round logs, plywood, MDF, chip board, hardware, doors, composite wood, furniture, carpentry, interior design, and construction materials are supported."

    if intent is not None:
        if intent in set(models.get("restricted_topics", [])) or confidence < 0.35:
            return QueryResponse(answer=OUT_OF_DOMAIN_MSG, sources=[])
    else:
        # Simple greeting check
        if query.lower().strip() in ["hi", "hello", "hey", "greetings", "good morning", "good evening"]:
            return QueryResponse(answer="Hello! How can I help you with Prime Source Global today?", sources=[])

        # Capabilities check
        if query.lower().strip().strip("?") in ["what can you do", "help", "what do you do", "capabilities"]:
             topics = ", ".join(models.get("allowed_topics", [])[:10]) # Limit to first 10 to avoid too long list
             return QueryResponse(answer=f"I can help you with information about Prime Source Global, including: {topics}. You can ask me about our services, partners, careers, or contact details.", sources=[])


        ok, a_s, r_s = _semantic_guard(query)
        if not ok:
            return QueryResponse(answer=OUT_OF_DOMAIN_MSG, sources=[])
    
    if MOCK_MODE:
        return QueryResponse(answer=f"I am running in MOCK MODE. You asked: {query}. The backend is connected but LLM is disabled for stability.", sources=[])

    if not models.get('llm'):
         raise HTTPException(status_code=503, detail="System not fully initialized (Model missing)")
    
    # Embed query
    query_embedding = models['embedding'].encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    
    # Search
    k = TOP_K_RETRIEVAL
    retrieved_contexts = []
    sources = []

    if models.get('index') is not None:
        distances, indices = models['index'].search(query_embedding, k)
        # indices is a list of lists (for each query)
        if indices.shape[1] > 0:
            for i, idx in enumerate(indices[0]):
                if idx != -1 and models.get('metadata') and idx < len(models['metadata']['texts']):
                    text = models['metadata']['texts'][idx]
                    meta = models['metadata']['metadatas'][idx]
                    retrieved_contexts.append(text)
                    sources.append(meta)
    elif models.get('doc_embeddings') is not None:
        # Cosine similarity fallback
        # doc_embeddings: (N, D), query_embedding: (1, D)
        # scores: (1, N) -> flatten to (N,)
        scores = np.dot(models['doc_embeddings'], query_embedding.T).flatten()
        # Get top k indices
        top_k_indices = np.argsort(scores)[::-1][:k]
        for idx in top_k_indices:
            if models.get('metadata') and idx < len(models['metadata']['texts']):
                text = models['metadata']['texts'][idx]
                meta = models['metadata']['metadatas'][idx]
                retrieved_contexts.append(text)
                sources.append(meta)
    else:
        print("Warning: Index not loaded, skipping retrieval.")
    
    context_str = "\n\n".join(retrieved_contexts)
    
    # Prompt Construction
    prompt = construct_prompt(query, context_str)
    
    # Generate
    output = models['llm'](
        prompt, 
        max_tokens=GENERATION_MAX_TOKENS, 
        stop=["<|im_end|>", "[/INST]", "###"], 
        echo=False,
        temperature=TEMPERATURE
    )
    
    answer = output['choices'][0]['text'].strip()

    # Post-processing: Correct model hallucination of BRS -> PSG
    answer = answer.replace("BRS Global", "Prime Source Global")
    answer = answer.replace("BRS", "PSG")
    
    return QueryResponse(answer=answer, sources=sources)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
