import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import ollama
from sentence_transformers import SentenceTransformer
import PyPDF2
from openai import OpenAI
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
from semantic_chunker.core import SemanticChunker
from semantic_text_splitter import TextSplitter

# Initialize the OpenAI client with your API key
client = OpenAI(api_key="OPENAI_API_KEY")

def read_file(file_path: Path) -> str:
    """
    Read file content from .txt, .pdf, or .md.
    """
    if file_path.suffix.lower() in [".txt", ".md"]:
        return file_path.read_text(encoding="utf-8")
    elif file_path.suffix.lower() == ".pdf":
        text = ""
        with file_path.open("rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")


def clean_text(text: str) -> str:
    """
    Remove sections like 'Bibliography' or 'References' if present.
    """
    match = re.search(r"(Bibliography|References)", text, re.IGNORECASE)
    return text[:match.start()] if match else text


def chunk_text_by_character(text: str, max_chunk_length: int = 1000) -> list:
    """
    Divide texto extraído de PDF em chunks, priorizando quebras em linhas duplas ou pontos finais.
    """
    lines = text.split("\n")
    chunks = []
    current_chunk = ""

    for line in lines:
        # Se adicionar a linha ultrapassa o limite, tenta quebrar antes de um ponto
        if len(current_chunk) + len(line) + 1 > max_chunk_length:
            # Se possível, quebra no último ponto final
            if '.' in current_chunk:
                last_period = current_chunk.rfind('.')
                chunk_to_add = current_chunk[:last_period+1]
                chunks.append(chunk_to_add.strip())
                remaining_text = current_chunk[last_period+1:].strip()
                current_chunk = remaining_text + "\n" + line + "\n"
            else:
                chunks.append(current_chunk.strip())
                current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def chunk_text_by_semantic(text: str, max_chunk_length: int = 1000)-> list:
    max_characters = 200
    splitter = TextSplitter((200,1000))
    chuncks_no_model = splitter.chunks(text)
    '''
    for i, chunck in enumerate(chuncks_no_model):
        print(f"CHUNK {i+1}: ", chunck)'''
    return chuncks_no_model

def embed_chunks(chunks: list, embedder) -> np.ndarray:
    """
    Compute embedding for each chunk.
    """
    return np.array([embedder.encode(chunk) for chunk in chunks])


def retrieve_relevant_chunks(query: str, chunks: list, chunk_embeddings: np.ndarray,
                             embedder, top_k: int = 3) -> list:
    """
    Retrieve top_k chunks that are most similar to the query.
    """
    query_embedding = embedder.encode(query)
    similarities = cosine_similarity(np.array(chunk_embeddings), np.array(query_embedding).reshape(1,-1)).flatten()
    best_match_ids = similarities.argsort()[-top_k:][::-1]
    relevant_chunks = [chunks[i] for i in best_match_ids]
    return relevant_chunks


def rag_text(document_texts: list[str]) -> None:
    """
        Given a list of documents and a query, retrieve top relevant chunks and use them to prompt the LLM.
    """
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Armazena todos os chunks dos documentos
    all_chunks = []

    for doc in document_texts:
        cleaned_text = clean_text(doc)
        chunks = chunk_text_by_semantic(cleaned_text)
        #print(f"Documento dividido em {len(chunks)} chunks.")
        all_chunks.extend(chunks)

    # Gera embeddings para todos os chunks
    embeddings = embed_chunks(all_chunks, embedder)

    print(f"Total de chunks: {len(all_chunks)}")

    user_text = input('Amos: ')
    relevant_chunks = retrieve_relevant_chunks(
    user_text, all_chunks, embeddings, embedder, top_k=3)
    context = "\n".join(relevant_chunks)
    for idx, chunk in enumerate(relevant_chunks):
        print(f"Chunk {idx+1}: {chunk}\n")




def main():
    input_folder = Path("output_rag")
    supported_extensions = ["*.txt", "*.pdf", "*.md"]
    files = [
        file for ext in supported_extensions for file in input_folder.glob(ext)]
    if not files:
        print("No supported files found in the input folder.")
        return
    texts_vector =[]
    for file in files:
        print(f"\nProcessing file: {file.name} with RAG.")
        try:
            text = read_file(file)
        except Exception as e:
            print(f"Error reading {file.name}: {e}")
            return None
        texts_vector.append(text)
    rag_text(texts_vector)
    

if __name__ == "__main__":
    main()
