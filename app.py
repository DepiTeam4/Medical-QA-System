# app.py
import streamlit as st
from Rag import extract_text_from_pdf, split_text, create_faiss_index, search_index, generate_answer, model

# Streamlit App
st.title("Interactive PDF Question-Answering App")

# File uploader for PDF selection
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if pdf_file is not None:
    st.write("Extracting and processing text from PDF...")
    book_text = extract_text_from_pdf(pdf_file)
    chunks = split_text(book_text)

    # Generate embeddings
    st.write("Generating embeddings for text chunks...")
    embeddings = model.encode(chunks, batch_size=16)
    index = create_faiss_index(embeddings)

    # Query input
    question = st.text_input("Ask a question about the content:")
    if question:
        st.write("Searching for relevant information...")
        retrieved_chunks = search_index(question, model, index, chunks)

        st.write("Generating answer...")
        answer = generate_answer(question, retrieved_chunks)
        st.write(f"Answer: {answer}")
