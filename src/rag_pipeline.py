import requests


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"


def answer_question(question: str, retrieved_chunks: list[dict], model: str) -> dict:
    """Send the question and retrieved document context to a local Ollama chat model."""
    if not retrieved_chunks:
        return {"answer": "I couldn't find relevant information in the uploaded document."}

    context_parts = []
    for item in retrieved_chunks:
        page = item["metadata"]["page"]
        context_parts.append(f"[Source: page {page}]\n{item['text']}")

    context = "\n\n".join(context_parts)
    prompt = f"""You are an assistant that answers questions using the supplied document context.

Rules:
- Answer using only the context below.
- If the answer is not present in the context, say: "I couldn't find that information in the document."
- Do not invent facts.
- Mention the relevant page number(s) when useful.
- Be clear and concise.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=180
        )
        response.raise_for_status()
        data = response.json()
        return {"answer": data["message"]["content"]}
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Cannot connect to Ollama at http://localhost:11434. Start Ollama and try again."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("Ollama took too long to respond. Try a smaller model or shorter question.") from exc
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = response.json().get("error", "")
        except Exception:
            pass
        raise RuntimeError(f"Ollama returned an error: {detail or response.status_code}") from exc
