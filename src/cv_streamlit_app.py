
import os
import streamlit as st
from dotenv import load_dotenv
from pinecone_registry import PineconeRegistry
from groq import Groq

# --- PDF & Text Processing ---
def read_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file using PyPDF2.
    """
    from pypdf import PdfReader
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Splits long text into overlapping chunks for better retrieval.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(text)

# --- Load environment variables ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# --- Pinecone Setup ---
INDEX_NAME = "cv-rag-index"
registry = PineconeRegistry.create(INDEX_NAME)

# --- RAG Response Generation ---
def generate_response(question: str, relevant_docs: list[str]) -> str:
    """Sends context + question to Groq and returns the answer."""
    context = "\n\n".join(relevant_docs)
    prompt = f"""
    Eres un asistente experto que responde basándote SOLO en el contexto proporcionado.
    Contexto: {context}
    
    Pregunta: {question}
    
    Respuesta clara y concisa:
    """
    
    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=400,
    )
    return completion.choices[0].message.content

# --- Streamlit UI ---
st.title("Chatbot RAG - Consulta mi CV")
st.write("Sube tu CV (PDF o TXT) y luego haz preguntas sobre tu experiencia, formación, etc.")

# File uploader
uploaded_file = st.file_uploader("Sube tu CV (PDF o .txt)", type=["pdf", "txt"])

# Initialize database button
if st.button("Inicializar base de datos con mi CV"):
    if uploaded_file is None:
        st.warning("Primero sube tu CV")
    else:
        with st.spinner("Procesando CV y subiendo a Pinecone..."):
            # Save uploaded file temporarily
            with open("temp_cv.pdf", "wb") if uploaded_file.type == "application/pdf" else open("temp_cv.txt", "w", encoding="utf-8") as f:
                f.write(uploaded_file.getvalue() if uploaded_file.type == "application/pdf" else uploaded_file.getvalue().decode())

            # Read text
            file_path = "temp_cv.pdf" if uploaded_file.type == "application/pdf" else "temp_cv.txt"
            raw_text = read_pdf(file_path) if uploaded_file.type == "application/pdf" else open(file_path, "r", encoding="utf-8").read()

            # Chunk and upload
            chunks = chunk_text(raw_text, chunk_size=500, chunk_overlap=50)
            registry.populate(chunks)

            st.success(f"CV cargado correctamente! {len(chunks)} fragmentos indexados en Pinecone.")
            st.balloons()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Pregunta sobre el CV (ej: ¿Qué experiencia tiene en Python?)"):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Buscando en tu CV..."):
        # Retrieve relevant chunks
        docs = registry.query(prompt, top_k=4)

        # Generate answer
        answer = generate_response(prompt, docs)

    st.chat_message("assistant").write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
