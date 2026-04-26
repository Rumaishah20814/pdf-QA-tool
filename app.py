import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import numpy as np

st.set_page_config(page_title="PDF Q&A Tool", page_icon="📄")
st.title("📄 PDF Question & Answer Tool")
st.write("Upload a PDF and ask any question about it!")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)

def build_index(chunks):
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def get_answer(question, chunks, index):
    question_embedding = model.encode([question])
    question_embedding = np.array(question_embedding).astype('float32')
    distances, indices = index.search(question_embedding, k=3)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Reading PDF..."):
        text = extract_text(uploaded_file)
        chunks = get_chunks(text)
        index, embeddings = build_index(chunks)
    
    st.success(f"✅ PDF loaded! Found {len(chunks)} text chunks.")
    st.divider()

    question = st.text_input("Ask a question about your PDF:",
                             placeholder="e.g. What is the main topic?")

    if st.button("Get Answer") and question:
        with st.spinner("Searching for answer..."):
            relevant_chunks = get_answer(question, chunks, index)

        st.subheader("📝 Answer:")
        for i, chunk in enumerate(relevant_chunks):
            st.info(f"**Relevant Section {i+1}:**\n{chunk}")