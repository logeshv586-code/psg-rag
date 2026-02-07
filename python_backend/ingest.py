import json
import os
import sys
try:
    import faiss
except ImportError:
    faiss = None
    print("Warning: faiss module not found. Index creation will be skipped.")
import numpy as np
from local_embeddings import LocalSentenceTransformer as SentenceTransformer
import pickle

# Import Configuration
try:
    from config import (
        DATA_FILES,
        INDEX_FOLDER,
        INDEX_FILE,
        METADATA_FILE,
        EMBEDDING_MODEL_NAME
    )
except ImportError:
    # Fallback if config.py is not found or path issues
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import (
        DATA_FILES,
        INDEX_FOLDER,
        INDEX_FILE,
        METADATA_FILE,
        EMBEDDING_MODEL_NAME
    )

def load_documents(path):
    print(f"Loading data from {path}...")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("documents", [])
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return []
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []

def ingest_data():
    # Create folder if not exists
    if not os.path.exists(INDEX_FOLDER):
        os.makedirs(INDEX_FOLDER)

    documents = []
    for data_file in DATA_FILES:
        documents.extend(load_documents(data_file))

    texts = []
    metadatas = []

    for doc in documents:
        # We can combine title and content for better context
        # Check if keys exist
        title = doc.get('title', '')
        content = doc.get('content', '')
        text_content = f"Title: {title}\nContent: {content}"
        texts.append(text_content)
        metadatas.append(doc)

    print(f"Loaded {len(documents)} documents from configured sources.")

    if not texts:
        print("No documents to process.")
        return

    # Load Model
    print(f"Loading embedding model ({EMBEDDING_MODEL_NAME})...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"Error loading model {EMBEDDING_MODEL_NAME}: {e}")
        model = None

    # Encode
    embeddings = None
    if model:
        print("Encoding texts...")
        embeddings = model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

    # Create FAISS index
    if faiss and embeddings is not None:
        print("Creating FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # Save Index
        print(f"Saving index to {INDEX_FILE}...")
        faiss.write_index(index, INDEX_FILE)
    else:
        print("Skipping FAISS index creation (faiss not installed or embeddings unavailable).")

    # Save Metadata
    print(f"Saving metadata to {METADATA_FILE}...")
    with open(METADATA_FILE, 'wb') as f:
        pickle.dump({'texts': texts, 'metadatas': metadatas, 'model_name': EMBEDDING_MODEL_NAME}, f)

    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_data()
