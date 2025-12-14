import os
import pinecone
from sentence_transformers import SentenceTransformer
import time
import unicodedata

class PineconeRegistry:
    """Handles creation, population, and querying of a Pinecone vector index."""

    @staticmethod
    def create(index_name: str = "cv-index",
               embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Creates a Pinecone index if it doesn't exist and returns a ready-to-use PineconeRegistry instance.
        Uses serverless index on AWS us-east-1 (free Starter plan).
        """
        # Initialize Pinecone client
        pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        # Load embedding model (384 dimensions by default)
        model = SentenceTransformer(embedding_model_name)

        # Create index if it does not exist
        if index_name not in pc.list_indexes().names():
            print(f"Creating index '{index_name}'...")
            pc.create_index(
                name=index_name,
                dimension=model.get_sentence_embedding_dimension(),
                metric="cosine",
                spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Wait until index is ready
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)
            print("Index created and ready.")

        # Connect to the index
        index = pc.Index(index_name)
        return PineconeRegistry(index, model)

    def __init__(self, pinecone_index, embedding_model):
        """Initialize with Pinecone index object and embedding model."""
        self.index = pinecone_index
        self.model = embedding_model

    def populate(self, documents: list[str]):
        """
        Uploads all CV chunks to Pinecone.
        Each chunk is converted to a vector and stored with its original text in metadata.
        Uses batches of 100 for optimal performance.
        """
        vectors = []
        for i, doc in enumerate(documents):
            # Convert text to embedding vector
            embedding = self.model.encode(doc).tolist()

            # Unique ID for each chunk
            doc_id = f"cv-{i:06d}"

            # Store original text in metadata for later retrieval
            metadata = {"text": doc.strip()}

            vectors.append((doc_id, embedding, metadata))

        # Upsert in batches of 100 (Pinecone recommendation)
        print(f"Uploading {len(vectors)} vectors to Pinecone...")
        for i in range(0, len(vectors), 100):
            batch = vectors[i:i + 100]
            self.index.upsert(vectors=batch)
        print("All vectors successfully uploaded to Pinecone.")

    def query(self, query_text: str, top_k: int = 3) -> list[str]:
        """
        Finds the top_k most similar CV chunks to the query.
        Returns only the original text from metadata.
        Similarity is calculated using cosine similarity.
        """
        # Encode query to vector
        query_embedding = self.model.encode(query_text).tolist()

        # Query Pinecone index
        result = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        # Extract original text from results
        retrieved_texts = []
        for match in result['matches']:
            if 'text' in match['metadata']:
                retrieved_texts.append(match['metadata']['text'])

        return retrieved_texts


def ascii_safe_string(s: str) -> str:
    """Convierte una cadena a ASCII puro, eliminando acentos y caracteres especiales."""
    # Normaliza y elimina combining marks (acentos)
    normalized = unicodedata.normalize('NFKD', s)
    ascii_str = normalized.encode('ascii', 'ignore').decode('ascii')
    # Reemplaza espacios y caracteres no válidos por underscore
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in ascii_str).lower()

class PineconeRegistry4agent:
    @staticmethod
    def create(index_name: str = "cv-rag-index-by-user",
               embedding_model_name: str = "all-MiniLM-L6-v2"):
        if not os.getenv("PINECONE_API_KEY"):
            raise ValueError("PINECONE_API_KEY no está configurada en las variables de entorno.")

        pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        model = SentenceTransformer(embedding_model_name, device='cpu')

        existing_indexes = [idx.name for idx in pc.list_indexes()]

        if index_name not in existing_indexes:
            print(f"Creando índice serverless '{index_name}'...")
            pc.create_index(
                name=index_name,
                dimension=model.get_sentence_embedding_dimension(),
                metric="cosine",
                spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
            )

            while index_name not in [idx.name for idx in pc.list_indexes()]:
                time.sleep(1)

            print("Índice creado.")

            while not pc.describe_index(index_name).status.get('ready', False):
                time.sleep(1)

            print("Índice listo para usar.")

        index = pc.Index(index_name)
        return PineconeRegistry4agent(index, model)

    def __init__(self, pinecone_index, embedding_model):
        self.index = pinecone_index
        self.model = embedding_model

    def populate(self, documents: list[str], person_name: str):
        if not documents:
            print(f"No hay documentos para cargar de {person_name}.")
            return

        vectors = []
        original_person_name = person_name.strip()  # Nombre con acentos para metadata y display

        # Sanitizar para ID ASCII-safe
        safe_person_name = ascii_safe_string(original_person_name)

        for i, doc in enumerate(documents):
            embedding = self.model.encode(doc).tolist()
            doc_id = f"{safe_person_name}-chunk-{i:06d}"
            metadata = {
                "person": original_person_name,  # Nombre original para filtros {"person": {"$eq": "Nicolás Craig"}}
                "text": doc.strip()
            }
            vectors.append((doc_id, embedding, metadata))

        print(f"Subiendo {len(vectors)} fragmentos de '{original_person_name}' (ID base: {safe_person_name})...")
        for i in range(0, len(vectors), 100):
            batch = vectors[i:i + 100]
            self.index.upsert(vectors=batch)

        print(f"CV de '{original_person_name}' cargado exitosamente.")

    def query(self, query_text: str, top_k: int = 6, filter: dict = None) -> list[str]:
        query_vec = self.model.encode(query_text).tolist()
        result = self.index.query(
            vector=query_vec,
            top_k=top_k,
            include_metadata=True,
            filter=filter
        )
        texts = []
        for match in result.matches:
            if match.metadata and 'text' in match.metadata:
                texts.append(match.metadata['text'])
        return texts