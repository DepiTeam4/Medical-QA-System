import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import re
from groq import Groq

# Load the sentence transformer model
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Step 1: Extract Text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page_num in range(8, min(310, doc.page_count)):  # Adjustable range
            page = doc.load_page(page_num)
            text += page.get_text()
        return re.sub('\n', ' ', text)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

# Step 2: Split Text and Create Embeddings
def split_text(text, chunk_size=1000):
    sentences = text.split('.')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + '.'
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + '.'
    if current_chunk:
        chunks.append(current_chunk)
    return chunks 

# Step 3: Set Up the Vector Database
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

def search_index(query, model, index, chunks, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    retrieved_chunks = [chunks[idx] for idx in indices[0]]
    return retrieved_chunks

# Step 4: Integrate with Groq API for Answer Generation
def generate_answer(question, retrieved_chunks):
    context = " ".join(retrieved_chunks)
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer this question based on this context"

    client = Groq(api_key='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')
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
    )
    answer = chat_completion.choices[0].message.content
    return answer
