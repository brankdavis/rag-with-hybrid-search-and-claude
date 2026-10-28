# Retrieval using Hybrid Search
# Hybrid search combines keyword search (sparse/BM25) and vector search (dense/semnatic) to retrieve more accureate context for llms
# resutls are merged using ranking (RRF) wor weighted alpha scoring
# use pinecone built in hybrid search capability

from pinecone import Pinecone
from anthropic import Client
from setup import get_anthropic_api_key, get_pinecone_api_key

pc_api_key = get_pinecone_api_key()
pc = Pinecone(api_key=pc_api_key)
index = pc.Index("traditional-rag-with-claude-dense")  # dense index
sparse_index = pc.Index("traditional-rag-with-claude-sparse")  # sparse index

def search_index(index, question):
    """search_index embeds and queries the Pinecone index in one step"""
    results = index.search(
        namespace="arxiv",
        # top_k=5,
        query={
            # # number of results to return
            "top_k": 5,
            # the query text to embed and search for
            "inputs": {"text": question}
        }
    )

    return results["result"]["hits"]

def augmentation():
    return

def generation():
    return

def merge_chunks(dense_results, sparse_results):
    """get unique chunks from dense and sparse results, and merge them into a array of {'_id', 'chunk_text'} dicts"""
    deduped_hits = {hit["_id"]: hit for hit in sparse_results + dense_results}.values()
    # Sort by _score descending
    sorted_hits = sorted(deduped_hits, key=lambda x: x["_score"], reverse=True)
    # Transform to format for reranking
    result = [{"_id": hit["_id"], "chunk_text": hit["fields"]["chunk_text"]} for hit in sorted_hits]
    return result


def rerank_results(merged_results, question):
    """rerank the merged results using a llm"""
    result = pc.inference.rerank(
        model="bge-reranker-v2-m3",
        query=question,
        documents=merged_results,
        rank_fields=["chunk_text"],
        top_n=5,
        return_documents=True,
        parameters={"truncate": "END"},
    )
    return result.data

def format_reranked_results(extracted: list[str]) -> str:
    result = "\n".join(
        [
            f'<item index="{i + 1}">\n<page_content>\n{r["document"]["chunk_text"]}\n</page_content>\n</item>'
            for i, r in enumerate(extracted)
        ]
    )

    return f"\n<search_results>\n{result}\n</search_results>"

def augment_prompt_with_context(results_list, question) -> str:
    """Augment the prompt with the context from the reranked results"""
    return f"""\n\nHuman: {format_reranked_results(results_list)} Using the search results provided within the <search_results></search_results> tags, please answer the following question <question>{question}</question>. Do not reference the search results in your answer.\n\nAssistant:"""

def generate_answer_with_claude(prompt: str) -> str:
    """Generate an answer using Claude"""
    api_key = get_anthropic_api_key()
    model = "claude-haiku-5-5"

    client = Client(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=1000,
        temperature=1,
        # system="You are a keyword generating assistant. Given a user message, you'll generate keywords to search for products.",
        messages=[{"role": "user", "content": [{"type": "text", "text": f"{prompt}"}]}],
    )
    return message.content

def main():
    USER_QUESTION = (
    "What are Mixture of Expert models, compared to regular transformer models?"
)
    dense_results = search_index(index, USER_QUESTION)
    sparse_results = search_index(sparse_index, USER_QUESTION)

    # for num, result in enumerate(dense_results, start=1):
    #     print(f"Result {num}:")
    #     print(f"ID: {result['_id']}")
    #     print(f"Score: {result['_score']}")
    #     print(result["fields"]["chunk_text"])
    #     print("\n")
    #     # print(f"Metadata: {result['metadata']}")
    #     # print(f"Values: {result['values']}")
    #     print()

    merged_results = merge_chunks(dense_results, sparse_results)
    # print("[\n   " + ",\n   ".join(str(obj) for obj in merged_results) + "\n]")

    reranked_results = rerank_results(merged_results, USER_QUESTION)

    answer = generate_answer_with_claude(augment_prompt_with_context(reranked_results, USER_QUESTION))
    print("Answer:\n", answer[0].text)

    # print("Query", USER_QUESTION)
    # print("-----")
    # for row in reranked_results:
    #     print(row["document"]["_id"])
    #     print(round(row["score"], 2))
    #     print(row["document"]["chunk_text"])
    #     print("\n")


    return

if __name__ == "__main__":
    main()