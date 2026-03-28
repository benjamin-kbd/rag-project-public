import httpx
from app.config import settings

HF_API_URL = f"https://api-inference.huggingface.co/models/{settings.HF_EMBED_MODEL}"
HEADERS = {"Authorization": f"Bearer {settings.HF_API_KEY}"}

async def get_embedding(text: str) -> list[float]:
    """텍스트를 BGE-M3로 임베딩"""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            HF_API_URL,
            headers=HEADERS,
            json={"inputs": text, "options": {"wait_for_model": True}},
        )
        response.raise_for_status()
        result = response.json()

    # HF Feature Extraction API: [[float, ...]] 형태로 반환
    if isinstance(result, list) and isinstance(result[0], list):
        return result[0]
    return result

async def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """배치 임베딩"""
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            HF_API_URL,
            headers=HEADERS,
            json={"inputs": texts, "options": {"wait_for_model": True}},
        )
        response.raise_for_status()
        return response.json()