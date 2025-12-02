import os
import streamlit as st
import tempfile
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from pinecone_registry import PineconeRegistry
from groq import Groq

# --- PDF Export (opcional) ---
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("Instala `pip install fpdf2` para exportar a PDF")

# ------------------- CONFIG -------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY no encontrada en .env")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
INDEX_NAME = "cv-rag-index"
registry = PineconeRegistry.create(INDEX_NAME)

# ------------------- LECTURA DE ARCHIVOS -------------------
def read_pdf(file_path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text.strip()

def read_txt(file_path: str) -> str:
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read().strip()
        except UnicodeDecodeError:
            continue
    return open(file_path, "r", encoding="utf-8", errors="replace").read().strip()

def read_document(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return read_pdf(file_path)
    elif suffix in {".txt", ".md"}:
        return read_txt(file_path)
    else:
        raise ValueError(f"Formato no soportado: {suffix}")

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    return splitter.split_text(text)

# ------------------- GENERAR RESPUESTA -------------------
def generate_response(question: str, relevant_docs: list[str]) -> str:
    context = "\n\n".join(relevant_docs)
    prompt = f"""
    Eres un asistente experto en análisis de CVs.
    Responde SOLO con información del contexto proporcionado.
    Si no aparece explícitamente, di: "No se menciona en el CV".

    Contexto del CV:
    {context}

    Pregunta: {question}

    Respuesta clara y profesional:"""

    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=st.session_state.temperature,
        max_tokens=st.session_state.max_tokens,
    )
    return completion.choices[0].message.content.strip()

# ------------------- EXPORTAR CHAT -------------------
def export_chat_txt():
    content = f"Consulta de CV - {datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
    content += f"CV analizado: {st.session_state.cv_name}\n"
    content += "="*60 + "\n\n"
    for msg in st.session_state.messages:
        role = "Tú" if msg["role"] == "user" else "Asistente"
        content += f"{role}: {msg['content']}\n\n"
    return content

def export_chat_md():
    content = f"# Consulta de CV\n\n"
    content += f"**Fecha:** {datetime.now().strftime('%d de %B de %Y, %H:%M')}\n"
    content += f"**CV:** {st.session_state.cv_name}\n\n---\n\n"
    for msg in st.session_state.messages:
        role = "Tú" if msg["role"] == "user" else "Asistente"
        content += f"### {role}\n{msg['content']}\n\n---\n"
    return content

def export_chat_pdf():
    if not PDF_AVAILABLE:
        return None
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Consulta de CV", ln=True, align="C")
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 10, f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True)
    pdf.cell(0, 10, f"CV: {st.session_state.cv_name}", ln=True)
    pdf.ln(10)
    pdf.set_font("Helvetica", size=11)
    for msg in st.session_state.messages:
        role = "Tú" if msg["role"] == "user" else "Asistente"
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 8, f"{role}:")
        pdf.set_font("Helvetica", size=11)
        clean_text = msg["content"].encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 8, clean_text)
        pdf.ln(5)
    return pdf.output(dest="S").encode("latin1")

# ------------------- GUARDAR EN CARPETA -------------------
def save_chat_to_folder():
    if not st.session_state.messages or not st.session_state.cv_loaded:
        return

    export_dir = Path("exports")
    export_dir.mkdir(exist_ok=True)

    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in st.session_state.cv_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{timestamp}_{safe_name}"

    # TXT
    with open(export_dir / f"{base}.txt", "w", encoding="utf-8") as f:
        f.write(export_chat_txt())

    # Markdown
    with open(export_dir / f"{base}.md", "w", encoding="utf-8") as f:
        f.write(export_chat_md())

    # PDF
    if PDF_AVAILABLE:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Consulta de CV", ln=True, align="C")
        pdf.set_font("Helvetica", size=10)
        pdf.cell(0, 10, f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True)
        pdf.cell(0, 10, f"CV: {st.session_state.cv_name}", ln=True)
        pdf.ln(10)
        pdf.set_font("Helvetica", size=11)
        for msg in st.session_state.messages:
            role = "Tú" if msg["role"] == "user" else "Asistente"
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(0, 8, f"{role}:")
            pdf.set_font("Helvetica", size=11)
            pdf.multi_cell(0, 8, msg["content"].encode('latin-1', 'replace').decode('latin-1'))
            pdf.ln(5)
        pdf.output(str(export_dir / f"{base}.pdf"))

# ------------------- MAIN -------------------
def main():
    st.set_page_config(page_title="CV Consultant RAG", page_icon="Resume", layout="centered")
    st.title("Chatbot RAG - Consultor de CVs")
    st.markdown("#### Sube tu CV y haz preguntas sobre experiencia, formación, proyectos...")

    # Estado
    for key, default in [("messages", []), ("cv_loaded", False), ("cv_name", None),
                         ("temperature", 0.7), ("max_tokens", 500), ("top_k", 5)]:
        if key not in st.session_state:
            st.session_state[key] = default

    # Configuración avanzada
    with st.expander("Configuración avanzada", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1: st.session_state.temperature = st.slider("Temperatura", 0.0, 1.0, st.session_state.temperature, 0.1)
        with c2: st.session_state.max_tokens = st.slider("Máx. tokens", 100, 2000, st.session_state.max_tokens, 50)
        with c3: st.session_state.top_k = st.slider("Top-k", 1, 10, st.session_state.top_k)

    # Carga CV
    uploaded_file = st.file_uploader("Sube tu CV", type=["pdf", "txt"])
    if st.button("Cargar CV y activar chatbot", type="primary", use_container_width=True):
        if not uploaded_file:
            st.warning("Sube un archivo primero")
            st.stop()
        with st.spinner("Procesando CV..."):
            suffix = Path(uploaded_file.name).suffix.lower()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            try:
                tmp.write(uploaded_file.getvalue())
                tmp.close()
                raw_text = read_document(tmp.name)
                if not raw_text.strip():
                    st.error("No se pudo extraer texto")
                    st.stop()
                chunks = chunk_text(raw_text)
                registry.delete_all()
                registry.populate(chunks)
                st.session_state.cv_loaded = True
                st.session_state.cv_name = uploaded_file.name
                st.success(f"CV cargado: **{uploaded_file.name}**")
                st.balloons()
            finally:
                Path(tmp.name).unlink(missing_ok=True)

    if st.session_state.cv_loaded:
        st.success(f"CV activo: **{st.session_state.cv_name}**")
    else:
        st.info("Sube un CV para comenzar")

    st.markdown("---")

    # Chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.cv_loaded and (prompt := st.chat_input("¿Qué quieres saber del CV?")):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                docs = registry.query(prompt, top_k=st.session_state.top_k)
                answer = generate_response(prompt, docs)
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Guardar en carpeta automáticamente
        save_chat_to_folder()

    # --- EXPORTAR CON BOTONES ---
    if st.session_state.messages:
        st.markdown("---")
        st.subheader("Exportar conversación")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in st.session_state.cv_name)
        filename = f"{timestamp}_{safe_name}"

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.download_button("TXT", data=export_chat_txt(), file_name=f"{filename}.txt", mime="text/plain")
        with col2:
            st.download_button("Markdown", data=export_chat_md(), file_name=f"{filename}.md", mime="text/markdown")
        with col3:
            if PDF_AVAILABLE:
                pdf_data = export_chat_pdf()
                if pdf_data:
                    st.download_button("PDF", data=pdf_data, file_name=f"{filename}.pdf", mime="application/pdf")
            else:
                st.write("PDF no disponible")
        with col4:
            st.caption(f"Guardado en\n`./CVconsultant_exports/`")

    # Limpiar
    if st.session_state.messages:
        if st.sidebar.button("Limpiar conversación"):
            st.session_state.messages = []
            st.rerun()

    st.sidebar.caption("CV Consultant RAG © 2025\nConversaciones guardadas en `./CVconsultant_exports/`")

if __name__ == "__main__":
    main()