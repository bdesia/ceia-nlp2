import os
import streamlit as st
from dotenv import load_dotenv
from pinecone_registry import PineconeRegistry4agent
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
registry = PineconeRegistry4agent.create(INDEX_NAME)

# --- CONFIGURACIÓN ---
st.sidebar.title("Configuración del modelo")
temperature = st.sidebar.slider("Temperatura", 0.0, 1.0, 0.7, 0.05)
max_tokens = st.sidebar.slider("Máximo de tokens", 50, 1024, 600, 50)
top_k = st.sidebar.slider("Fragmentos recuperados (top-k)", 1, 15, 6)

st.sidebar.info(f"""
**Configuración actual**  
Modelo: `llama-3.3-70b-versatile`  
Temperatura: `{temperature}`  
Max tokens: `{max_tokens}`  
Top-k: `{top_k}`
""")

# --- Extraer nombres de personas de la pregunta ---
def extract_persons(question: str) -> list[str]:
    prompt = f"""
    De la siguiente pregunta, extrae SOLO los nombres propios de personas sobre las que se pregunta.
    Si no hay nombres claros o no se menciona ninguno, responde exactamente "NINGUNO".
    Si hay varios, sepáralos por coma.
    
    Pregunta: {question}
    
    Nombres:"""
    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        max_tokens=100,
    )
    result = completion.choices[0].message.content.strip()
    if result.upper() == "NINGUNO" or not result:
        return []
    return [name.strip() for name in result.split(",") if name.strip()]

# --- Generar respuesta de un agente (por persona) ---
def generate_single_response(question: str, docs: list[str]) -> str:
    context = "\n\n".join(docs)
    prompt = f"""
    Eres un asistente experto en análisis de CVs.
    Responde ÚNICAMENTE con información del contexto proporcionado.
    Sé claro, conciso y profesional.

    Contexto del CV:
    {context}

    Pregunta: {question}
    Respuesta:"""
    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content.strip()

# --- Interfaz ---
st.title("Chatbot RAG - MultiCV Analyzer")
st.write("Carga CVs de diferentes personas y pregunta sobre uno o varios a la vez.")

# --- Carga de CV ---
person_name = st.text_input("Nombre completo de la persona (ej: Juan Pérez)", help="Requerido para asociar el CV")
uploaded_file = st.file_uploader("Sube el CV (PDF o TXT)", type=["pdf", "txt"])

if st.button("Cargar CV a la base de datos"):
    if not person_name.strip() or not uploaded_file:
        st.warning("Debes ingresar el nombre de la persona y subir un archivo.")
    else:
        with st.spinner(f"Procesando CV de **{person_name}**..."):
            file_ext = ".pdf" if uploaded_file.type == "application/pdf" else ".txt"
            temp_path = os.path.join(TEMP_DIR, f"cv_{person_name.replace(' ', '_')}{file_ext}")

            # Guardar temporalmente
            with open(temp_path, "wb" if "pdf" in file_ext else "w", encoding="utf-8") as f:
                f.write(uploaded_file.getvalue() if "pdf" in file_ext else uploaded_file.getvalue().decode())

            # Extraer texto
            raw_text = read_pdf(temp_path) if "pdf" in file_ext else open(temp_path, "r", encoding="utf-8").read()
            chunks = chunk_text(raw_text)

            # Subir a Pinecone con metadata de persona
            registry.populate(chunks, person_name=person_name.strip())

            os.remove(temp_path)
            st.success(f"CV de **{person_name}** cargado correctamente ({len(chunks)} fragmentos).")
            st.balloons()

# --- Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("¿Qué querés saber del CV o de las personas? (ej: '¿Qué experiencia tiene María en marketing?')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Analizando la pregunta y consultando los CVs..."):
        mentioned_persons = extract_persons(prompt)

        if not mentioned_persons:
            answer = "No detecté ningún nombre de persona en tu pregunta. Por favor, mencioná el nombre de la(s) persona(s) sobre la(s) que querés información."
        else:
            agent_responses = []

            for person in mentioned_persons:
                docs = registry.query(prompt, top_k=top_k, filter={"person": {"$eq": person}})
                if docs:
                    resp = generate_single_response(prompt, docs)
                    agent_responses.append((person, resp))
                else:
                    agent_responses.append((person, f"No se encontró información de **{person}** en la base de datos."))

            # Si es una sola persona → respuesta directa
            if len(agent_responses) == 1:
                answer = agent_responses[0][1]

            # Si son varias → supervisor integra/compara
            else:
                parts = []
                for person, resp in agent_responses:
                    parts.append(f"### Agente de {person}\n{resp}\n")

                supervisor_prompt = f"""
                Eres un supervisor experto en análisis de currículums.
                La pregunta del usuario es: "{prompt}"

                Tienes las respuestas de los agentes especializados por persona:

                {"".join(parts)}

                Proporciona una respuesta final clara, profesional y bien estructurada.
                Si la pregunta es comparativa, hacé una comparación explícita.
                Si no, resumí la información relevante de cada persona.

                Respuesta final:
                """
                completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": supervisor_prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=temperature,
                    max_tokens=max_tokens + 200,
                )
                answer = completion.choices[0].message.content.strip()

    # Mostrar respuesta
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

# --- Exportar chat ---
if st.session_state.messages:
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("Exportar chat", type="primary"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_cv_multi_{timestamp}.md"
            filepath = os.path.join(EXPORT_DIR, filename)

            content = f"# Análisis de CVs - {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
            content += f"**Modelo:** llama-3.3-70b-versatile | **Temp:** {temperature} | **Top-k:** {top_k}\n\n---\n\n"
            for msg in st.session_state.messages:
                role = "Usuario" if msg["role"] == "user" else "Asistente"
                content += f"### {role}\n{msg['content']}\n\n"

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

            st.download_button("Descargar .md", content, file_name=filename, mime="text/markdown")
            st.success("Listo para descargar")

# --- Limpiar ---
if st.button("Limpiar conversación"):
    st.session_state.messages = []
    st.rerun()