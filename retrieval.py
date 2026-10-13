# Retrieval using Hybrid Search
# Hybrid search combines keyword search (sparse/BM25) and vector search (dense/semnatic) to retrieve more accureate context fir llms
# resutls are merged using ranking (RRF) wor weighted alpha scoring
# use pinecone built in hybrid search capability

from pinecone import Pinecone
from setup import get_pinecone_api_key

pc = get_pinecone_api_key()
index = pc.Index("dense_index")

def search_records(index, question):
    """search_records embeds and queries the Pinecone index in one step"""

    return

def augmentation():
    return

def generation():
    return

def main():
    return