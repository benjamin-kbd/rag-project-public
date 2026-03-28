import httpx
from app.config import settings

HF_LLM_URL = f"https://api-inference.huggingface.co/models/{settings.HF_LLM_MODEL}"
HEADERS = {"Authorization": f"Bearer {settings.HF_API_KEY}"}

SYSTEM_PROMPT = """당신은 주어진 문서를 바탕으로 질문에 답변하는 AI 어시스턴트입니다.
반드시 제공된 컨텍스트만을 사용하여 답변하세요.
답변은 한국어로 작성하세요."""

async def generate_answer(question: str, contexts: list[str]) -> str:
    context_text = "\n\n---\n\n".join(
        [f"[문서 {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)]
    )

    prompt = f"""<|system|>
{SYSTEM_PROMPT}
<|user|>
[참고 문서]
{context_text}

[질문]
{question}
<|assistant|>"""

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            HF_LLM_URL,
            headers=HEADERS,
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 1024,
                    "temperature": 0.1,
                    "return_full_text": False,
                },
                "options": {"wait_for_model": True},
            },
        )
        response.raise_for_status()
        result = response.json()

    return result[0]["generated_text"].strip()