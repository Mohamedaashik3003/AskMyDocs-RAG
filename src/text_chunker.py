def chunk_pages(pages: list[dict], chunk_size: int = 1000, overlap: int = 150) -> list[dict]:
    """
    Split each page into overlapping character chunks.
    Page-level splitting preserves useful source locations for answers.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    chunks = []
    chunk_id = 1

    for page in pages:
        text = " ".join(page["text"].split())
        if not text:
            continue

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Try to end near a sentence or word boundary, without making tiny chunks.
            if end < len(text):
                boundary = max(text.rfind(". ", start, end),
                               text.rfind("? ", start, end),
                               text.rfind("! ", start, end),
                               text.rfind(" ", start, end - 100))
                if boundary > start:
                    end = boundary + 1

            piece = text[start:end].strip()
            if piece:
                chunks.append({
                    "text": piece,
                    "metadata": {
                        "page": page["page"],
                        "chunk_id": chunk_id
                    }
                })
                chunk_id += 1

            if end >= len(text):
                break
            start = max(end - overlap, start + 1)

    return chunks
