import os
from datetime import datetime
from pathlib import Path
import re
import shutil
import time

import numpy as np
import pandas as pd
import chromadb
import ollama
import PyPDF2
from openai import OpenAI, RateLimitError
from sentence_transformers import SentenceTransformer
from semantic_chunker.core import SemanticChunker
from semantic_text_splitter import TextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

chroma_client = chromadb.Client()
chroma_client = chromadb.PersistentClient(path="db")
collection = chroma_client.get_or_create_collection(name="artigo")


load_dotenv()

# Initialize the OpenAI client with your API key
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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


def chunk_text_by_semantic(text: str, max_chunk_length: int = 2500) -> list:
    min_characters = 200
    splitter = TextSplitter((min_characters, max_chunk_length))
    chunks_no_model = splitter.chunks(text)
    for i, pedaco in enumerate(chunks_no_model):
        collection.add(documents=pedaco, ids=[str(i)])
    '''
    for i, chunk in enumerate(chunks_no_model):
        print(f"CHUNK {i+1}: ", chunk)'''
    return chunks_no_model


def embed_chunks(chunks: list, embedder):
    """
    Compute embedding for each chunk.
    """
    for i, chunk in enumerate(chunks):
        '''
        embedding = embedder.encode(chunk)
        collection.add(
            documents=[chunk],
            embeddings=[embedding.tolist()],
            ids=[str(i)]
        )'''
        collection.add(documents=chunk, ids=[str(i)])


def retrieve_relevant_chunks(query: str, chunks: list, embeddings: np.array,
                             embedder, top_k: int = 3) -> list:
    """
    Retrieve top_k chunks that are most similar to the query.
    """
    query_embedding = embedder.encode(query)
    similarities = cosine_similarity(np.array(embeddings), np.array(
        query_embedding).reshape(1, -1)).flatten()
    best_match_ids = similarities.argsort()[-top_k:][::-1]

    return [chunks[i] for i in best_match_ids]


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
        # print(f"Documento dividido em {len(chunks)} chunks.")
        all_chunks.extend(chunks)

    # Gera embeddings para todos os chunks
    #embed_chunks(all_chunks, embedder)

    print(f"Total de chunks: {len(all_chunks)}")


def check_for_files(file_path: Path):
    supported_extensions = ["*.txt", "*.pdf", "*.md"]
    files = [
        file for ext in supported_extensions for file in file_path.glob(ext)]

    if not files:
        print("Nenhum arquivo suportado encontrado na pasta.")
        return

    texts_vector = []
    for file in files:
        print(f"\nProcessando arquivo: {file.name} com RAG.")

        try:
            text = read_file(file)
            texts_vector.append(text)

            # Move o arquivo para a pasta 'output_rag'
            output_folder = Path("output_rag")
            output_folder.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file), output_folder / file.name)
            print(f"{file.name} movido para {output_folder}")

        except Exception as e:
            print(f"Erro ao processar {file.name}: {e}")

            # Move o arquivo para a pasta 'input_rag_failed'
            error_folder = Path("input_rag_failed")
            error_folder.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file), error_folder / file.name)
            print(f"{file.name} movido para {error_folder}")

    if texts_vector:
        rag_text(texts_vector)


def preprocess_question(question):
    system_instruction = "Você é um assistente que reformula perguntas ou solicitações para que fiquem mais objetivas e claras no contexto de Guia Turístico do Estado do Amazonas. Deixe a pergunta mais direta sem mudar o sentido."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": question}
        ]
    )

    return response.choices[0].message.content.strip()

def rag_operation():
    prompt = """
    Você é um assistente de Guia Turístico do Estado do Amazonas.
    Use o seguinte contexto para responder a questão, não use nenhuma informação adicional,
    se não houver informacao no contexto, responda: Desculpe mas não consigo ajudar.
    Sempre termine a resposta com: Foi um prazer lhe atender.
    """

    while True:
        user_text = input('Usuário: ')
        #processed_question = preprocess_question(user_text)
        #print(f"Pergunta processada: {processed_question}")

        relevant_chunks = collection.query(query_texts=[user_text], n_results=5)

        context = ""
        for idx, chunk in enumerate(relevant_chunks["documents"][0]):
            context += chunk + "\n"
            print(f"Chunk {idx+1}: {chunk}\n")

        messages = [
            {"role": "system", "content": prompt},
            {"role": "system", "content": context},
            {"role": "user", "content": user_text}
        ]

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        print(f"Assistente: {completion.choices[0].message.content}")

        time.sleep(20)  



def main():
    input_folder = Path("input_rag")
    check_for_files(input_folder)
    rag_operation()


if __name__ == "__main__":
    main()
