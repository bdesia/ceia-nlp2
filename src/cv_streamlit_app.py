import os
import streamlit as st
from dotenv import load_dotenv
from pinecone_registry import PineconeRegistry
from groq import Groq
from datetime import datetime

# --- CARPETAS ---
TEMP_DIR = "temp/"
EXPORT_DIR = "exports/"

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# --- PDF & Text Processing ---
def read_pdf(file_path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(text)

# --- Cargar variables de entorno ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY no encontrada. Crea un archivo .env con tu clave.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# --- Pinecone ---
INDEX_NAME = "cv-rag-index"
registry = PineconeRegistry.create(INDEX_NAME)

# --- CONFIGURACIÓN ---
st.sidebar.title("Configuración del modelo")
temperature = st.sidebar.slider("Temperatura", 0.0, 1.0, 0.7, 0.05)
max_tokens = st.sidebar.slider("Máximo de tokens", 50, 1024, 400, 50)
top_k = st.sidebar.slider("Fragmentos recuperados (top-k)", 1, 10, 4)

st.sidebar.info(f"""
**Configuración actual**  
Modelo: `llama-3.3-70b-versatile`  
Temperatura: `{temperature}`  
Max tokens: `{max_tokens}`  
Top-k: `{top_k}`
""")

# --- Generación de respuesta ---
def generate_response(question: str, relevant_docs: list[str]) -> str:
    context = "\n\n".join(relevant_docs)
    prompt = f"""
    Eres un asistente experto en análisis de currículums (CVs).
    Responde SOLO con información que esté en el contexto.
    Sé claro, conciso y profesional.

    Contexto:
    {context}

    Pregunta: {question}

    Respuesta:
    """
    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content.strip()

# --- Interfaz ---
st.title("Chatbot RAG - CV Analyzer")
st.write("Subí un CV y preguntá sobre experiencia, formación, habilidades, etc.")

uploaded_file = st.file_uploader("Sube un CV", type=["pdf", "txt"], help="PDF o TXT")

# --- Cargar CV ---
if st.button("Cargar CV a la base de datos"):
    if not uploaded_file:
        st.warning("Primero subí un CV")
    else:
        with st.spinner("Procesando CV y cargando a Pinecone..."):
            file_extension = ".pdf" if uploaded_file.type == "application/pdf" else ".txt"
            temp_path = os.path.join(TEMP_DIR, f"cv_uploaded{file_extension}")

            try:
                if uploaded_file.type == "application/pdf":
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                else:
                    with open(temp_path, "w", encoding="utf-8") as f:
                        f.write(uploaded_file.getvalue().decode("utf-8"))

                raw_text = read_pdf(temp_path) if uploaded_file.type == "application/pdf" else open(temp_path, "r", encoding="utf-8").read()
                chunks = chunk_text(raw_text, chunk_size=500, chunk_overlap=50)
                registry.populate(chunks)

                st.success(f"CV cargado con éxito! {len(chunks)} fragmentos indexados.")
                st.balloons()

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                if os.path.exists(temp_path):
                    try: os.remove(temp_path)
                    except: pass

# --- Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("¿Qué querés saber del CV?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Buscando en el CV..."):
        docs = registry.query(prompt, top_k=top_k)
        answer = generate_response(prompt, docs)

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

# --- EXPORTAR CONVERSACIÓN ---
if st.session_state.messages:
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Exportar chat", type="primary"):
            # Generar contenido Markdown
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_cv_analyzer_{timestamp}.md"
            filepath = os.path.join(EXPORT_DIR, filename)

            export_content = f"# Análisis de CV - Sesión del {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
            export_content += f"**Modelo:** llama-3.3-70b-versatile | **Temp:** {temperature} | **Top-k:** {top_k}\n\n"
            export_content += "---\n\n"

            for msg in st.session_state.messages:
                role = "Usuario" if msg["role"] == "user" else "Asistente"
                export_content += f"### {role}\n{msg['content']}\n\n"

            # Guardar en disco
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(export_content)

            # Ofrecer descarga
            st.download_button(
                label="Descargar como .md",
                data=export_content,
                file_name=filename,
                mime="text/markdown"
            )

            st.success(f"Conversación exportada a:\n`{filepath}`")

# --- Limpiar chat ---
if st.button("Limpiar conversación"):
    st.session_state.messages = []
    st.rerun()