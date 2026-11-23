import pymupdf


def extract_pdf_pages(pdf_bytes: bytes) -> list[dict]:
    pages = []

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        for page_index, page in enumerate(document):
            pages.append({
                "page": page_index + 1,
                "text": page.get_text("text").strip()
            })

    return pages