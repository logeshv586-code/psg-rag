import os
import sys
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FOLDER = os.path.join(BASE_DIR, "faiss_index")
INDEX_FILE = os.path.join(INDEX_FOLDER, "index.faiss")
METADATA_FILE = os.path.join(INDEX_FOLDER, "metadata.pkl")
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def load_rag():
    print("Loading RAG system...")
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        print(f"Index or metadata not found at {INDEX_FOLDER}")
        return None, None, None

    try:
        index = faiss.read_index(INDEX_FILE)
        print("Index loaded.")
        with open(METADATA_FILE, 'rb') as f:
            data = pickle.load(f)
            metadata = data.get('metadatas', [])
        print(f"Metadata loaded: {len(metadata)} items.")
        
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Model loaded.")
        return index, metadata, model
    except Exception as e:
        print(f"Error loading RAG: {e}")
        return None, None, None

def query_rag(query, index, metadata, model, k=3):
    print(f"\nQuerying: {query}")
    query_vector = model.encode([query]).astype('float32')
    distances, indices = index.search(query_vector, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            results.append(metadata[idx])
            print(f"Rank {i+1} (Dist: {distances[0][i]:.4f}): {metadata[idx].get('title', 'No Title')}")
            # print(f"Content: {metadata[idx].get('content')[:200]}...")
    return results

if __name__ == "__main__":
    index, metadata, model = load_rag()
    if index:
        # Debug: list all titles
        # for i, m in enumerate(metadata):
        #     print(f"{i}: {m.get('title')}")
            
        query_rag("What travel services do you offer?", index, metadata, model)
        query_rag("Tell me about software solutions", index, metadata, model)
        query_rag("Careers at PSG", index, metadata, model)
