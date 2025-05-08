from semantic_chunker.core import SemanticChunker

chunks = [
    "Artificial intelligence is a growing field.",
    "Machine learning is a subset of AI.",
    "Photosynthesis occurs in plants.",
    "Deep learning uses neural networks.",
    "Plants convert sunlight into energy.",
]

# Transforma lista de strings em dicionários esperados pelo chunker
chunk_dicts = [{"text": c} for c in chunks]

chunker = SemanticChunker(max_tokens=512)
merged_chunks = chunker.chunk(chunk_dicts)

# Depois pega só o texto de volta
final_chunks = [c["text"] for c in merged_chunks]

# Mostra os resultados
for i, merged in enumerate(final_chunks):
    print(f"Chunk {i}:")
    print(merged)
    print()

