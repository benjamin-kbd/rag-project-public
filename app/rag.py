from app.embedder import get_embedding
from app.vectorstore import search_similar
from app.llm import generate_answer

async def run_rag(question: str, top_k: int = 5) -> dict:
    """전체 RAG 파이프라인 실행"""

    # 1. 질문 임베딩
    query_vector = await get_embedding(question)

    # 2. 유사 문서 검색
    search_results = search_similar(query_vector, top_k=top_k)

    if not search_results:
        return {
            "answer": "관련 문서를 찾을 수 없습니다.",
            "sources": [],
            "question": question,
        }

    # 3. 컨텍스트 추출
    contexts = [r["text"] for r in search_results]
    scores = [r["score"] for r in search_results]

    # 4. LLM 답변 생성
    answer = await generate_answer(question, contexts)

    return {
        "answer": answer,
        "question": question,
        "sources": [
            {"text": r["text"][:200] + "...", "score": round(r["score"], 4)}
            for r in search_results
        ],
    }