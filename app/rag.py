from app.embedder import get_embedding
from app.vectorstore import search_similar
from app.reranker import rerank
from app.llm import generate_answer

async def run_rag(question: str, top_k: int = 5) -> dict:
    """Semantic Search + Reranking RAG """

    query_vector = await get_embedding(question)

    search_results = search_similar(query_vector, top_k=top_k * 2)

    if not search_results:
        return {
            "answer": "not found doc",
            "sources": [],
            "question": question,
        }

    documents = [r["text"] for r in search_results]
    reranked = await rerank(question, documents, top_k=3)

    contexts = [r["text"] for r in reranked]
    answer = await generate_answer(question, contexts)

    return {
        "answer": answer,
        "question": question,
        "sources": [
            {
                "text": r["text"][:200] + "...",
                "score": round(r["score"], 4),
                "reranked": True,
            }
            for r in reranked
        ],
    }
