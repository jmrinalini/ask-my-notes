"""
Phase 3: Vector storage and retrieval.

Phase 1 gave us chunks. Phase 2 proved embeddings capture meaning.
Now we combine them for real: embed EVERY chunk of the document, store
those vectors in a searchable database (ChromaDB), and then, given a
question, find the chunks whose meaning is closest to it.

This is "retrieval" - the R in RAG. Nothing generative happens yet;
we're only finding the right pieces of text. Generation (asking Claude/
Gemini to actually answer using these chunks) is Phase 4.

Why ChromaDB specifically?
    It's a vector database that runs locally (no server, no cost, no
    account) and is built for exactly this: store vectors + their
    original text, then query "give me the top-k closest vectors to
    this new vector." It saves everything to disk in chroma_db/, so you
    don't have to re-embed the document every time you run the script.
"""

import chromadb
from phase1_ingest import extract_text_from_pdf, chunk_text
from phase2_embeddings import embed_texts


def build_vector_store(pdf_path: str, collection_name: str = "notes"):
    """
    Full pipeline: PDF -> text -> chunks -> embeddings -> stored in ChromaDB.
    Returns the ChromaDB collection, ready to be queried.
    """
    # PersistentClient saves the database to disk (chroma_db/ folder) so it
    # survives between runs - we don't want to re-embed the PDF every time.
    client = chromadb.PersistentClient(path="chroma_db")

    # If we've already built this collection before, reuse it instead of
    # redoing all the work. Delete the chroma_db/ folder if you want a
    # fresh rebuild (e.g. after changing chunk size).
    existing_collections = [c.name for c in client.list_collections()]
    if collection_name in existing_collections:
        print(f"Collection '{collection_name}' already exists - reusing it.")
        return client.get_collection(collection_name)

    print(f"Building new collection '{collection_name}'...")
    collection = client.create_collection(collection_name)

    print(f"Reading and chunking: {pdf_path}")
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(raw_text, target_words=500)
    print(f"Got {len(chunks)} chunks. Embedding them now...")

    embeddings = embed_texts(chunks)

    # ChromaDB needs: a unique id, the vector, and the original text for
    # each chunk. We generate simple ids like "chunk_0", "chunk_1", ...
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=chunks
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB (saved to chroma_db/).\n")

    return collection


def search(collection, question: str, top_k: int = 3):
    """
    Embed the question, then ask ChromaDB for the top_k chunks whose
    embeddings are closest to it. Returns the matching chunk texts and
    how close each one was.
    """
    question_embedding = embed_texts([question])[0]

    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=top_k
    )

    # results is a dict of lists (one list per query - we only sent 1 query)
    matched_chunks = results["documents"][0]
    distances = results["distances"][0]  # lower distance = more similar

    return matched_chunks, distances


if __name__ == "__main__":
    pdf_path = "data/internship_report.pdf"

    collection = build_vector_store(pdf_path)

    question = "What obstacle detection method was used in the final Python phase?"
    print(f"Question: {question}\n")

    matched_chunks, distances = search(collection, question, top_k=6)

    for i, (chunk, distance) in enumerate(zip(matched_chunks, distances)):
        print(f"--- Match {i + 1} (distance: {distance:.4f}, lower = more relevant) ---")
        print(chunk[:300] + ("..." if len(chunk) > 300 else ""))
        print()