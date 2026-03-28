import httpx
from app.config import settings

HF_RERANK_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-reranker-base"
HEADERS = {
    "Authorization": f"Bearer {settings.HF_API_KEY}",
    "Content-Type": "application/json",
}

async def rerank(query: str, documents: list[str], top_k: int = 3) -> list[dict]:
    if not documents:
        return []

    pairs = [{"text": query, "text_pair": doc} for doc in documents]

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                HF_RERANK_URL,
                headers=HEADERS,
                json={"inputs": pairs, "options": {"wait_for_model": True}},
            )
            response.raise_for_status()
            result = response.json()

        # 응답 구조: [[{'label': 'LABEL_0', 'score': 0.81}, ...]]
        inner_scores = result[0] if isinstance(result[0], list) else result

        ranked = sorted(
            [
                {"text": doc, "score": float(item["score"]), "index": i}
                for i, (doc, item) in enumerate(zip(documents, inner_scores))
            ],
            key=lambda x: x["score"],
            reverse=True,
        )
        return ranked[:top_k]

    except Exception as e:
        print(f"재랭킹 실패, 원본 순서 유지: {e}")
        return [
            {"text": doc, "score": 0.0, "index": i}
            for i, doc in enumerate(documents[:top_k])
        ]
