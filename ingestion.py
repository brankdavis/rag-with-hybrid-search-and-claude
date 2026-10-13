import anthropic
import backoff
from datasets import load_dataset
from pinecone import Pinecone
from tqdm.auto import tqdm
from tqdm import tqdm

from setup import get_pinecone_api_key



def load_Thedataset():
    dataset = load_dataset("jamescalam/ai-arxiv2", split="train")
    return dataset

def preprocess(dataset):
    dataset = dataset.remove_columns(['content', 'comment', 'journal_ref', 'references'])
    dataset = dataset.rename_column("summary", "chunk_text")

    print(dataset[0])
    return dataset

def setup_dense_index():
    pc_api_key = get_pinecone_api_key()
    pc = Pinecone(api_key=pc_api_key)

    dense_index_name = "traditional-rag-with-claude-dense"

    if not pc.has_index(dense_index_name):
        pc.create_index_for_model(
            name=dense_index_name,
            cloud="aws",
            region="us-east-1",
            # Chunk text will be the field we embed from our documents
            embed={"model": "llama-text-embed-v2", "field_map": {"text": "chunk_text"}},
        )

    dense_index = pc.Index(dense_index_name)
    dense_index.describe_index_stats()
    return dense_index


def setup_sparse_index():
    pc_api_key = get_pinecone_api_key()
    pc = Pinecone(api_key=pc_api_key)
    
    sparse_index_name = "traditional-rag-with-claude-sparse"
    if not pc.has_index(sparse_index_name):
        pc.create_index_for_model(
            name=sparse_index_name,
            cloud="aws",
            region="us-east-1",
            # Chunk text will be the field we embed from our documents
            embed={
                "model": "pinecone-sparse-english-v0",
                "field_map": {"text": "chunk_text"},
            },
        )

    sparse_index = pc.Index(sparse_index_name)
    sparse_index.describe_index_stats()
    return sparse_index

@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=8,
    max_time=80,
    on_backoff=lambda details: print(f"Backoff: {details['tries']} of 8"),
)
def upsert_in_batches(dataset, index, batchsize=96):
    """Upsert batches with a limit of 96 (set by Pinecone)"""
    for start in tqdm(range(0, len(dataset), batchsize), desc="upserting records in batches"):
        cols = dataset[start : start + batchsize]                      # dict of columns
        records = [dict(zip(cols, row)) for row in zip(*cols.values())]  # list of row dicts
        for r in records:
            r["_id"] = r.pop("id")                                     # integrated indexes use "_id"
        index.upsert_records(records=records, namespace="arxiv")



def search_index(index, question):
    results = index.search(
        namespace="arxiv",
        query={
            # specifies number of results to return
            "top_k": 5,
            # specifies the query to embed and search for
            "inputs": {"text": question}

        }
    )

    return results["results"]["hits"]


if __name__ == "__main__":

    # print("Loading dataset...")
    # dataset = load_Thedataset()
    # dataset_processed = preprocess(dataset)

    # print("Setting up indexes...")
    # dense_index = setup_dense_index()
    # sparse_index =  setup_sparse_index()

    # print("upserting batches")
    # # upsert_in_batches(dataset_processed, dense_index)
    # upsert_in_batches(dataset_processed, sparse_index)

    # print("dense index stats:")
    # dense_index.describe_index_stats()

    # print("sparse index stats:")
    # sparse_index.describe_index_stats()

    # print("Done!")

    USER_QUESTION = (
    "What are Mixture of Expert models, compared to regular transformer models?"
    )
    dense_results = search_index(dense_index, USER_QUESTION)