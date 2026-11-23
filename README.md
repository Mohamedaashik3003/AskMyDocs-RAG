# AskMyDocs – PDF Question Answering with RAG

AskMyDocs is a beginner-friendly Retrieval-Augmented Generation (RAG) application. Upload a PDF, ask questions about its contents, and receive answers grounded in retrieved document passages.

## Features

- Extracts selectable text from PDF files using PyMuPDF.
- Splits text into overlapping chunks while preserving page numbers.
- Creates semantic embeddings with Sentence Transformers.
- Uses FAISS for similarity search.
- Sends retrieved context to a local Ollama language model.
- Displays retrieved source passages and page numbers.
- Provides a simple Streamlit user interface.

## Architecture

1. **Load:** Read each PDF page with PyMuPDF.
2. **Chunk:** Split page text into smaller overlapping passages.
3. **Embed:** Convert each passage into a vector using `all-MiniLM-L6-v2`.
4. **Index:** Store vectors in a FAISS inner-product index. Since vectors are normalized, inner product ranks cosine similarity.
5. **Retrieve:** Embed the user's question and retrieve the top matching passages.
6. **Generate:** Send the question and retrieved passages to Ollama.
7. **Show sources:** Display the passages and their original page numbers.

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or 3.11 recommended
- Internet connection for the first installation and embedding-model download
- Ollama installed locally
- A local Ollama model (the examples use `llama3.2:1b`)

## Installation (Windows / Anaconda Prompt)

Create and activate an environment:

```bash
conda create -n rag_project python=3.11 -y
conda activate rag_project
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Install Ollama from https://ollama.com/download and open the Ollama application. Then download the model:

```bash
ollama pull llama3.2:1b
```

If that model is unavailable for your hardware or Ollama setup, choose another model available in Ollama and enter its exact name in the app sidebar.

## Run the application

From the AskMyDocs-RAG project root:

```bash
streamlit run app.py
```

A browser tab should open. Upload a text-based PDF, click **Process PDF**, and ask a question.

## Project structure

```text
AskMyDocs-RAG/
├── app.py
├── src/
│   ├── __init__.py
│   ├── pdf_loader.py
│   ├── text_chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── rag_pipeline.py
├── tests/
│   └── test_text_chunker.py
├── data/
│   └── .gitkeep
├── notebooks/
├── requirements.txt
├── .gitignore
└── README.md
```

## Limitations

- Scanned/image-only PDFs need OCR; OCR is not included in this version.
- The FAISS index is held in memory and is rebuilt when a PDF is processed.
- The first embedding run downloads the Sentence Transformers model.
- Answer quality depends on PDF text quality, chunking, retrieval, and the selected LLM.
- The model is instructed to use retrieved context, but generated answers can still be incorrect. Verify important information against the displayed source passages.

## Future improvements

- Add OCR for scanned PDFs.
- Support multiple PDFs and persistent vector storage.
- Add conversation memory and document selection.
- Evaluate retrieval with Precision@K / Recall@K and answer faithfulness.
- Add configurable chunk size and overlap.
- Add source highlighting and downloadable chat history.

## Privacy

PDF extraction and embeddings run locally. Questions and retrieved text are sent to the local Ollama service on the same machine. Do not upload confidential documents to a public repository.

## License

This project is available under the MIT License. Add a `LICENSE` file if you choose to publish it under that license.
