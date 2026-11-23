
from src.text_chunker import chunk_pages


def test_chunk_pages_returns_chunks():
    pages = [
        {
            "page": 1,
            "text": "Artificial Intelligence is transforming the world. " * 50
        }
    ]

    chunks = chunk_pages(pages, chunk_size=200, overlap=30)

    assert len(chunks) > 0
    assert all("text" in chunk for chunk in chunks)
    assert all("metadata" in chunk for chunk in chunks)


def test_chunk_pages_preserves_page_number():
    pages = [
        {
            "page": 1,
            "text": "This is a sample document for testing text chunking."
        }
    ]

    chunks = chunk_pages(pages)

    assert chunks[0]["metadata"]["page"] == 1

