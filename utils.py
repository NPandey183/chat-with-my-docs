import os
import openai
import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-ada-002"

def chunk_text(text, max_chunk_size=1000, overlap=200):
    """
    Break text into chunks of max_chunk_size with some overlap
    """
    start = 0
    chunks = []
    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += max_chunk_size - overlap
    return chunks

def get_embedding(text):
    """
    Get embedding vector for a text chunk from OpenAI API
    """
    response = openai.Embedding.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response['data'][0]['embedding']

def build_faiss_index(chunks):
    """
    Build FAISS index from text chunks embeddings
    """
    embeddings = [get_embedding(chunk) for chunk in chunks]
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index, chunks

def search_faiss(index, query, chunks, top_k=3):
    """
    Search FAISS index with query embedding to get top_k chunks
    """
    query_embedding = np.array([get_embedding(query)]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    results = [chunks[i] for i in indices[0]]
    return results

