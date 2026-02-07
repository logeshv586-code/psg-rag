import os

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# DATA CONFIGURATION
# ==============================================================================
DATA_DIR = os.path.join(BASE_DIR, "psg_data")
# List of JSON files to ingest
DATA_FILES = [
    os.path.join(DATA_DIR, "psgbiz_rag_ready.json"),
    os.path.join(DATA_DIR, "psgbiz_rag_arabic.json"),
    os.path.join(DATA_DIR, "psgbiz_rag_project_division.json"),
    os.path.join(DATA_DIR, "etrainia.json"),
    os.path.join(DATA_DIR, "psg_timber.json"),
    os.path.join(DATA_DIR, "psg_corporate_comprehensive.json"),
    os.path.join(DATA_DIR, "rag_ui_catalog.json")
]

# ==============================================================================
# VECTOR STORE CONFIGURATION
# ==============================================================================
INDEX_FOLDER = os.path.join(BASE_DIR, "faiss_index")
INDEX_FILE = os.path.join(INDEX_FOLDER, "index.faiss")
METADATA_FILE = os.path.join(INDEX_FOLDER, "metadata.pkl")

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================
# Embedding Model: 
# Smaller/Faster: "all-MiniLM-L6-v2"
# Multilingual (Recommended): "paraphrase-multilingual-MiniLM-L12-v2"
# High Quality: "intfloat/multilingual-e5-large"
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# LLM Path
# You can point this to any GGUF file (Llama, Mistral, Qwen, Gemma, etc.)
# Example: os.path.join(BASE_DIR, "..", "models", "llama-2-7b-chat.Q4_K_M.gguf")
LLM_MODEL_PATH = os.path.join(BASE_DIR, "..", "qwen_model", "Qwen2.5-1.5B-Instruct-Q8_0.gguf")

# LLM Parameters
CTX_MAX_TOKENS = 2048      # Model context window size
N_GPU_LAYERS = -1          # -1 = all layers to GPU (if available)

# ==============================================================================
# RAG GENERATION CONFIGURATION
# ==============================================================================
TOP_K_RETRIEVAL = 3
SIMILARITY_THRESHOLD = 0.3
GENERATION_MAX_TOKENS = 512
TEMPERATURE = 0.7

# ==============================================================================
# PROMPT TEMPLATE CONFIGURATION
# ==============================================================================
# Options: "chatml" (Qwen, OpenHermes), "llama2" ([INST]), "alpaca" (### Instruction), "plain"
# Auto-detect or set manually
PROMPT_STYLE = "chatml" 

def get_system_prompt():
    return (
        "You are a helpful and professional assistant for Prime Source Global (PSG). "
        "Always refer to the company as 'PSG' or 'Prime Source Global'. "
        "Do NOT use 'BSR' or any other acronym. "
        "Use the provided context to answer the user's question. "
        "If the answer is not in the context, politely say you don't know based on the provided information. "
        "IMPORTANT: Always answer in the SAME language as the user's question. "
        "If the user asks in English, answer in English. If the user asks in Arabic, answer in Arabic. "
        "Keep the answer concise and relevant to the domain."
    )
