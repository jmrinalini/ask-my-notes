"""
Phase 2: Embeddings - turning text into vectors that capture MEANING.

What is an embedding, really?
    A model reads a sentence and outputs a list of numbers (e.g. 384 of them
    for the model we're using). That list is a coordinate in a very
    high-dimensional space. The key property: sentences with SIMILAR MEANING
    end up CLOSE TOGETHER in that space, even if they don't share any of the
    same words. "The dog ran fast" and "The canine sprinted quickly" would be
    close, even though zero words match.

Why does this matter for our project?
    This is what makes RAG "smart" instead of just keyword search. When you
    ask a question, we embed the question, then find which document chunks
    are numerically CLOSEST to it - i.e. closest in meaning, not closest in
    exact word overlap.

We use sentence-transformers here: a FREE model that runs entirely on your
own machine (no API call, no cost, no internet needed after the first
download). The first run will download the model (~90MB), then it's cached
locally forever.
"""

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# This loads a small, fast, well-regarded embedding model.
# "all-MiniLM-L6-v2" turns any sentence into a 384-number vector.
print("Loading embedding model (first run downloads ~90MB, then it's cached)...")
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: list[str]):
    """Turn a list of text strings into a list of embedding vectors."""
    return model.encode(texts)


def demo_semantic_similarity():
    """
    Prove embeddings capture MEANING, not just word overlap, by comparing
    distances between related vs. unrelated sentence pairs.
    """
    sentences = [
        "The vehicle platoon uses V2V communication.",     # 0
        "Cars in a convoy talk to each other wirelessly.",  # 1 - same MEANING as 0, different words
        "I had pasta for lunch today.",                     # 2 - unrelated
    ]

    embeddings = embed_texts(sentences)

    sim_0_1 = cos_sim(embeddings[0], embeddings[1]).item()
    sim_0_2 = cos_sim(embeddings[0], embeddings[2]).item()

    print("\n--- Semantic similarity demo ---")
    print(f'Sentence A: "{sentences[0]}"')
    print(f'Sentence B: "{sentences[1]}"  (different words, SAME meaning)')
    print(f'Sentence C: "{sentences[2]}"  (unrelated)')
    print()
    print(f"Similarity(A, B) = {sim_0_1:.4f}  <- should be HIGH (close in meaning)")
    print(f"Similarity(A, C) = {sim_0_2:.4f}  <- should be LOW (unrelated)")
    print()
    print("Similarity scores range from -1 to 1. Higher = more similar in meaning.")
    print("Notice: A and B share almost no words, but score high anyway.")
    print("That's the whole trick behind semantic search.")


if __name__ == "__main__":
    demo_semantic_similarity()

    # Bonus: show the actual shape of an embedding, so it's not just abstract
    sample_vector = embed_texts(["This is a test sentence."])[0]
    print(f"\n--- What an embedding actually looks like ---")
    print(f"Vector length: {len(sample_vector)} numbers")
    print(f"First 8 numbers: {sample_vector[:8]}")