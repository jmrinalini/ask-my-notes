"""
Phase 7: Hybrid search - combining semantic + keyword matching.

Recall the Phase 3 finding: the small embedding model (MiniLM) struggled
with dense technical text - acronyms like "YOLOv8" or "ACO-GWO" didn't
embed as distinctively as natural language does. Semantic search alone
missed some of these.

The fix: run TWO searches for every query, then merge the results:
    1. SEMANTIC search (what we already have) - good at "meaning",
       finds paraphrases and related concepts even with zero word overlap
    2. KEYWORD search (new, this phase) - good at exact terms, will
       NEVER miss a chunk that literally contains "YOLOv8" if the
       question mentions "YOLOv8"

This combination is called "hybrid search" and is standard practice in
real-world RAG systems for exactly this reason - each method covers the
other's blind spot.

Our keyword search is intentionally simple: count how many of the
question's words appear in each chunk (a basic form of what's formally
called "lexical" or "BM25-style" scoring, simplified for learning
purposes). Good enough to demonstrate the concept clearly.
"""

import re
from phase3_vector_store import build_vector_store, search as semantic_search


# Common English words that appear in almost every chunk regardless of topic.
# Without filtering these out, keyword scoring can't tell chunks apart -
# every chunk scores similarly high just from "the", "was", "in", etc.
STOPWORDS = {
    "the", "a", "an", "in", "on", "at", "was", "were", "is", "are", "of",
    "to", "for", "and", "or", "what", "which", "used", "final", "with",
    "this", "that", "it", "as", "be", "by", "from", "did", "do", "does",
}


def keyword_score(query: str, chunk: str) -> int:
    """
    Simple keyword scoring: count how many distinct MEANINGFUL query words
    (stopwords removed) appear in the chunk (case-insensitive). This is a
    simplified version of what's formally called "lexical" or "BM25-style"
    scoring - good enough to demonstrate the core idea.
    """
    query_words = set(re.findall(r'\w+', query.lower())) - STOPWORDS
    chunk_lower = chunk.lower()

    return sum(1 for word in query_words if word in chunk_lower)


def hybrid_search(collection, all_chunks: list[str], question: str, top_k: int = 3):
    """
    Combine semantic and keyword search results.

    Approach: get semantic search's top candidates (cast a slightly wider
    net than we need), ALSO score every chunk by keyword overlap, then
    merge by giving each chunk a combined score. This way a chunk that
    scores poorly semantically but contains an exact critical term (like
    "YOLOv8") still has a chance to surface.
    """
    # Cast a wide net with semantic search first (more than top_k, so
    # keyword scoring has room to re-rank things)
    semantic_matches, semantic_distances = semantic_search(collection, question, top_k=len(all_chunks))

    # Convert semantic distance (lower = better) into a similarity score
    # (higher = better) so we can combine it with keyword score the same way.
    # Distances here are roughly 0-2, so this keeps things on a comparable scale.
    combined_scores = []
    for chunk, distance in zip(semantic_matches, semantic_distances):
        semantic_similarity = 1 - (distance / 2)  # rough normalize to ~0-1
        kw_score = keyword_score(question, chunk)

        # Weighted combination: tweak these weights to favor one method
        # more. Equal weighting is a reasonable starting point.
        combined = (0.5 * semantic_similarity) + (0.5 * kw_score)
        combined_scores.append((chunk, combined, semantic_similarity, kw_score))

    # Sort by combined score, descending (higher = more relevant now)
    combined_scores.sort(key=lambda x: x[1], reverse=True)

    return combined_scores[:top_k]


if __name__ == "__main__":
    pdf_path = "data/internship_report.pdf"
    collection = build_vector_store(pdf_path)

    # We need the raw chunk list too, for reference (not strictly required
    # by hybrid_search above, but useful for comparison prints below)
    from phase1_ingest import extract_text_from_pdf, chunk_text
    all_chunks = chunk_text(extract_text_from_pdf(pdf_path), target_words=500)

    question = "What obstacle detection method was used in the final Python phase?"
    print(f"Question: {question}\n")

    print("--- HYBRID search results ---\n")
    results = hybrid_search(collection, all_chunks, question, top_k=3)

    for i, (chunk, combined, sem_sim, kw) in enumerate(results):
        print(f"Match {i + 1}: combined_score={combined:.2f}  "
              f"(semantic_sim={sem_sim:.2f}, keyword_hits={kw})")
        print(chunk[:250] + "...")
        print()

    print("=" * 70)
    print("\nCompare: this is what SEMANTIC-ONLY search gave us in Phase 3 "
          "(top match was the WORST-scoring chunk for this same question "
          "despite containing the literal answer - see Phase 3 notes).")