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

# --- Lista de personas cargadas (persistida en session_state) ---
if "loaded_people" not in st.session_state:
    st.session_state.loaded_people = []

# --- CONFIGURACIÓN EN SIDEBAR ---
st.sidebar.title("Configuración del modelo")

model_options = {
    "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile",
    "Mixtral 8x22B Instruct": "mixtral-8x22b-instruct"
}
selected_model_name = st.sidebar.selectbox("Seleccionar modelo", options=list(model_options.keys()))
model_id = model_options[selected_model_name]

temperature = st.sidebar.slider("Temperatura", 0.0, 1.0, 0.7, 0.05)
max_tokens = st.sidebar.slider("Máximo de tokens", 50, 2048, 1000, 50)
top_k = st.sidebar.slider("Fragmentos recuperados (top-k)", 1, 15, 8)
show_context = st.sidebar.checkbox("Mostrar contexto recuperado (debug)", value=False)

available_people = sorted(st.session_state.loaded_people)
st.sidebar.success(f"Personas cargadas: {len(available_people)}")
if available_people:
    st.sidebar.write(", ".join(available_people))
else:
    st.sidebar.info("Aún no hay CVs cargados.")

st.sidebar.info(f"""
**Modelo actual:** {selected_model_name}  
**ID:** `{model_id}`  
**Temperatura:** {temperature}  
**Max tokens:** {max_tokens}  
**Top-k:** {top_k}
""")

# --- Funciones LLM ---
def call_model(prompt: str, temperature_override: float = None, max_tokens_override: int = None) -> str:
    temp = temperature_override if temperature_override is not None else temperature
    mt = max_tokens_override if max_tokens_override is not None else max_tokens
    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_id,
        temperature=temp,
        max_tokens=mt,
    )
    return completion.choices[0].message.content.strip()

def extract_name_from_cv(text: str) -> str:
    prompt = f"""
    Extrae SOLO el nombre completo de la persona del siguiente CV.
    Busca en encabezado, datos personales o primeras líneas.
    Responde únicamente con el nombre y apellido.
    Si no lo encuentras claramente, responde "DESCONOCIDO".

    Inicio del CV:
    {text[:1500]}

    Nombre:"""
    name = call_model(prompt, temperature_override=0.0, max_tokens_override=50)
    if name.upper() == "DESCONOCIDO" or not name.strip():
        return None
    return name.strip()

# --- Detección flexible de personas ---
def extract_persons(question: str) -> list[str]:
    if not available_people:
        return []

    lower_question = question.lower()
    detected = set()

    for person in available_people:
        lower_person = person.lower()
        # Partes del nombre
        parts = lower_person.replace('.', ' ').split()
        # Si alguna parte coincide
        if any(part in lower_question for part in parts):
            detected.add(person)

    # Si no detectó nada, pero la pregunta parece comparativa/general
    if not detected:
        comparative_keywords = ["quién", "quien", "cual", "más", "mejor", "mayor", "menor", "comparar", "versus", "vs", "experiencia", "python", "java", "sql"]
        if any(keyword in lower_question for keyword in comparative_keywords):
            return available_people  # Comparar todos

    return list(detected)

def generate_subquery(question: str, person: str) -> str:
    prompt = f"""
    Genera una pregunta corta y precisa para recuperar información relevante del CV de {person}.

    Ejemplo:
    Pregunta: "¿Quién tiene más experiencia en Python?"
    Persona: María Gómez
    Sub-pregunta: "Experiencia de María Gómez en Python"

    Pregunta original: {question}
    Persona: {person}
    Sub-pregunta:"""
    return call_model(prompt, temperature_override=0.0, max_tokens_override=100)

def generate_single_response(question: str, docs: list[str]) -> str:
    if not docs:
        return "No se encontró información relevante en el CV."
    context = "\n\n".join(docs)
    prompt = f"""
    Eres un experto en análisis de CVs.
    Responde ÚNICAMENTE con información del contexto proporcionado.
    Sé claro, conciso y profesional.

    Contexto:
    {context}

    Pregunta: {question}
    Respuesta:"""
    return call_model(prompt)

# --- Interfaz principal ---
st.title("CV Analyzer Agent RAG")
st.write("Cargar  CVs y preguntar.")

# --- Carga múltiple ---
uploaded_files = st.file_uploader(
    "Seleccionar CVs" \
    "",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"Seleccionaste **{len(uploaded_files)}** CV(s). Presioná el botón para procesar.")

if st.button("Procesar y cargar todos los CVs") and uploaded_files:
    progress_bar = st.progress(0)
    status_text = st.empty()
    success_count = 0

    for idx, file in enumerate(uploaded_files):
        status_text.text(f"Procesando {idx + 1}/{len(uploaded_files)}: {file.name}...")

        try:
            ext = ".pdf" if file.type == "application/pdf" else ".txt"
            temp_path = os.path.join(TEMP_DIR, f"temp_{idx}_{file.name}")

            if file.type == "application/pdf":
                with open(temp_path, "wb") as f:
                    f.write(file.getvalue())
                text = read_pdf(temp_path)
            else:
                with open(temp_path, "w", encoding="utf-8") as f:
                    f.write(file.getvalue().decode("utf-8"))
                text = open(temp_path, "r", encoding="utf-8").read()

            name = extract_name_from_cv(text) or os.path.splitext(file.name)[0].replace('_', ' ')
            final_name = name.strip()

            chunks = chunk_text(text)
            registry.populate(chunks, person_name=final_name)

            if final_name not in st.session_state.loaded_people:
                st.session_state.loaded_people.append(final_name)

            if os.path.exists(temp_path):
                os.remove(temp_path)

            success_count += 1

        except Exception as e:
            st.error(f"Error con {file.name}: {e}")

        progress_bar.progress((idx + 1) / len(uploaded_files))

    progress_bar.empty()
    status_text.empty()
    st.success(f"¡{success_count} CVs cargados correctamente! 🎉")
    st.balloons()

# --- Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("¿Qué querés saber sobre los candidatos?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    reasoning = ""
    people_ph = st.empty()
    context_ph = st.empty()
    answer_ph = st.empty()

    with st.spinner("Analizando..."):
        persons = extract_persons(prompt)

        if not persons and available_people:
            persons = available_people
            reasoning += "**Modo comparación automática activado** (pregunta general/comparativa).\n\n"

        reasoning += f"**Candidatos analizados:** {', '.join(persons) if persons else 'Ninguno'}\n\n"

        with people_ph.container():
            st.markdown("**Candidatos analizados:**")
            st.write(", ".join(persons) if persons else "Ninguno (no hay CVs)")

        if not persons:
            answer = "No hay CVs cargados o no se detectaron candidatos válidos."
        else:
            responses = []
            contexts = {}

            for person in persons:
                subq = generate_subquery(prompt, person)
                docs = registry.query(subq, top_k=top_k, filter={"person": {"$eq": person}})
                contexts[person] = docs
                reasoning += f"**{person}**\n- Sub-pregunta: `{subq}`\n- Fragmentos: {len(docs)}\n\n"
                resp = generate_single_response(prompt, docs)
                responses.append((person, resp))

            with context_ph.container():
                st.markdown("**Contexto recuperado:**")
                for p, d in contexts.items():
                    with st.expander(f"{p} ({len(d)} fragmentos)"):
                        if show_context and d:
                            st.text("\n\n".join(d))

            if len(responses) == 1:
                answer = responses[0][1]
            else:
                parts = [f"### {p}\n{r}" for p, r in responses]
                joined = "\n\n".join(parts)
                supervisor_prompt = f"""
                Eres un supervisor experto en selección de personal.
                Pregunta del usuario: "{prompt}"

                Respuestas individuales:
                {joined}

                Da una respuesta final clara, profesional y comparativa.
                Respuesta final:"""
                answer = call_model(supervisor_prompt, max_tokens_override=max_tokens + 400)

        full_response = f"{reasoning}\n**Respuesta final:**\n\n{answer}"
        with answer_ph.container():
            st.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- Exportar y limpiar ---
if st.session_state.messages:
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("Exportar chat", type="primary"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_cv_{ts}.md"
            content = f"# Análisis de CVs\n**Modelo:** {selected_model_name}\n\n"
            for m in st.session_state.messages:
                role = "Usuario" if m["role"] == "user" else "Asistente"
                content += f"### {role}\n{m['content']}\n\n"
            st.download_button("Descargar .md", content, file_name=filename, mime="text/markdown")

if st.button("Limpiar conversación"):
    st.session_state.messages = []
    st.rerun()