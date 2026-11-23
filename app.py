import streamlit as st
from src.pdf_loader import extract_pdf_pages
from src.text_chunker import chunk_pages
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.rag_pipeline import answer_question

st.set_page_config(page_title="AskMyDocs | RAG Assistant", page_icon="📚", layout="wide")
st.title("📚 AskMyDocs")
st.caption("Ask questions about your PDF documents using Retrieval-Augmented Generation.")

with st.sidebar:
    st.header("Settings")
    ollama_model = st.text_input("Ollama model", value="llama3.2:1b")
    top_k = st.slider("Retrieved chunks", min_value=1, max_value=8, value=4)
    st.markdown("---")
    st.markdown("**Local components**")
    st.write("Embeddings: all-MiniLM-L6-v2")
    st.write("Vector search: FAISS")
    st.write("LLM: Ollama")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None
if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file is not None:
    if st.button("Process PDF", type="primary"):
        try:
            with st.spinner("Reading PDF and creating embeddings... This may take a little while the first time."):
                pdf_bytes = uploaded_file.getvalue()
                pages = extract_pdf_pages(pdf_bytes)
                if not any(page["text"].strip() for page in pages):
                    st.error("No selectable text was found. This may be a scanned PDF; OCR support is not included yet.")
                    st.stop()

                chunks = chunk_pages(pages)
                if not chunks:
                    st.error("No text chunks could be created from this PDF.")
                    st.stop()

                embedder = EmbeddingModel()
                vectors = embedder.encode([chunk["text"] for chunk in chunks])
                store = VectorStore()
                store.build(vectors, chunks)

                st.session_state.vector_store = store
                st.session_state.document_name = uploaded_file.name
                st.session_state.messages = []
            st.success(f"Processed {uploaded_file.name}: {len(pages)} pages and {len(chunks)} text chunks.")
        except Exception as exc:
            st.error(f"Could not process this PDF: {exc}")

if st.session_state.vector_store is not None:
    st.success(f"Ready to answer questions about **{st.session_state.document_name}**.")
    st.subheader("Ask a question")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("For example: What is the main topic of this document?")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the document and generating an answer..."):
                    embedder = EmbeddingModel()
                    query_vector = embedder.encode([question])
                    retrieved = st.session_state.vector_store.search(query_vector, top_k=top_k)
                    result = answer_question(question, retrieved, ollama_model)
                st.markdown(result["answer"])
                with st.expander("Sources retrieved from your PDF"):
                    for item in retrieved:
                        meta = item["metadata"]
                        st.markdown(f"**Page {meta['page']} · Chunk {meta['chunk_id']} · Similarity {item['score']:.3f}**")
                        st.write(item["text"])
                st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
            except Exception as exc:
                st.error(
                    f"Answer generation failed: {exc}\n\n"
                    "Check that Ollama is running and that the selected model has been downloaded."
                )
else:
    st.info("Upload a PDF and select **Process PDF** to begin.")
    st.markdown(
        "Your PDF is processed locally by this application. Text is embedded locally, "
        "and questions are sent to your local Ollama model."
    )
