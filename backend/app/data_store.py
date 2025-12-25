
# data_store.py
from pypdf import PdfReader
from openai import OpenAI


def load_document(pdf_path: str):
    global chunks

    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    # simple chunking (same logic as notebook)
    chunk_size = 800
    overlap = 150

    start = 0
    chunks = []

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap


def embed_chunks():
    global embeddings

    client = OpenAI()

    embeddings = []

    for chunk in chunks:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        vector = response.data[0].embedding
        embeddings.append(vector)
