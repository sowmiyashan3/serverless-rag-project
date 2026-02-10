import requests
import numpy as np
from fastembed import FastEmbed


FAISS_API_URL = "URL"
OPENSEARCH_API_URL = "URL"
LLAMA_API_URL = "URL"

# Initialize FastEmbed
embed_model = FastEmbed(model_name="all-MiniLM-L6-v2")  # same model as before

def normalize_query(q: str) -> str:
    return " ".join(q.lower().split())


def get_query_embedding(query_text: str):
    # FastEmbed automatically returns normalized embeddings
    embedding = embed_model.embed([query_text])[0]
    return embedding.tolist()


def search_faiss(query_embedding, top_k=5):
    payload = {"query_embedding": query_embedding, "top_k": top_k}
    r = requests.post(FAISS_API_URL, json=payload)
    r.raise_for_status()
    return r.json().get("results", [])


def search_opensearch(query_text, top_k=5):
    params = {"q": query_text, "top_k": top_k}
    r = requests.get(OPENSEARCH_API_URL, params=params)
    r.raise_for_status()
    return r.json().get("results", [])


def refine_query_llama(query_text, retrieved_snippets):
    prompt = f"""
query:
{query_text}

Do NOT add extra words
STICK to the main context
Do NOT include the answer for the query

"""

    payload = {"query": prompt}
    r = requests.post(LLAMA_API_URL, json=payload)
    r.raise_for_status()
    return r.json()["optimized_query"].strip()



def iterative_hybrid_search(
    query_text,
    faiss_top_k=5,
    os_top_k=5,
    iterations=5,
    faiss_threshold=0.45,
    os_threshold=0.40
):
    seen_queries = set()

    for i in range(iterations):
        print(f"\n Iteration {i + 1}")
        print(" Query:", query_text)

        normalized = normalize_query(query_text)
        if normalized in seen_queries:
            print(" Query repeated — stopping loop")
            break
        seen_queries.add(normalized)

        # ---- FAISS ----
        query_embedding = get_query_embedding(query_text)
        faiss_results = search_faiss(query_embedding, faiss_top_k)

        # ---- OpenSearch ----
        os_results = search_opensearch(query_text, os_top_k)

        # ---- Logging ----
        print("\n FAISS Results:")
        for r in faiss_results:
            print(r)

        print("\n OpenSearch Results:")
        for r in os_results:
            doc_id = r.get("id", "N/A")
            score = r.get("metadata", {}).get("normalized_score", 0.0)
            content = r.get("content", "")
            print(f"\n ID: {doc_id}")
            print(f" Score: {score:.3f}")
            print(f" Content: {content[:300]}...")

        # ---- Confidence Check ----
        avg_faiss_score = (
            np.mean([r["score"] for r in faiss_results])
            if faiss_results else 0.0
        )

        avg_os_score = (
            np.mean([r["metadata"]["normalized_score"] for r in os_results])
            if os_results else 0.0
        )

        print(f"\nAvg FAISS score: {avg_faiss_score:.3f}")
        print(f"Avg OpenSearch score: {avg_os_score:.3f}")

        should_refine = (
            avg_faiss_score < faiss_threshold or
            avg_os_score < os_threshold
        )

        if not should_refine:
            print("Retrieval strong — stopping")
            break

        print("Low confidence → refining query with LLaMA")

        retrieved_texts = [
            r.get("content", "")[:300]
            for r in os_results
        ]

        new_query = refine_query_llama(query_text, retrieved_texts)

        if new_query == "NO_CHANGE":
            print("Model says query is optimal")
            break

        print("Refined Query:", new_query)
        query_text = new_query

    return {
        "final_query": query_text,
        "faiss_results": faiss_results,
        "opensearch_results": os_results
    }



if __name__ == "__main__":
    query = "Give Your own Query"
    results = iterative_hybrid_search(query)

    print("\nFinal Results")
    print(results)
