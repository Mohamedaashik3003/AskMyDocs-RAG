
import os
import requests


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-20b"


def get_groq_api_key():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = None

    if not api_key:
        raise RuntimeError("Groq API key is missing. Configure GROQ_API_KEY in Streamlit Secrets.")

    return api_key


def answer_question(question: str, retrieved_chunks: list[dict], model: str) -> dict:
    """Generate an answer using retrieved PDF context and Ollama or Groq."""

    if not retrieved_chunks:
        return {
            "answer": "I couldn't find relevant information in the uploaded document."
        }

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
- Mention relevant page numbers when useful.
- Be clear and concise.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    try:
        if provider == "groq":
            api_key = get_groq_api_key()

            response = requests.post(
                GROQ_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1
                },
                timeout=120
            )

            response.raise_for_status()
            data = response.json()

            return {
                "answer": data["choices"][0]["message"]["content"]
            }

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

        return {
            "answer": data["message"]["content"]
        }

    except requests.exceptions.ConnectionError as exc:
        if provider == "groq":
            raise RuntimeError("Cannot connect to Groq API. Check your internet connection.") from exc

        raise RuntimeError(
            "Cannot connect to Ollama. Start Ollama and try again."
        ) from exc

    except requests.exceptions.Timeout as exc:
        raise RuntimeError("The AI model took too long to respond. Try again.") from exc

    except requests.exceptions.HTTPError as exc:
        detail = ""

        try:
            detail = response.json().get("error", "")
        except Exception:
            pass

        raise RuntimeError(
            f"AI service returned an error: {detail or response.status_code}"
        ) from exc
