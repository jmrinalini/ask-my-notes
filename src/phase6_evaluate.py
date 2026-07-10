"""
Phase 6: Evaluation.

Up to now we've been checking quality by eye - reading one answer and
deciding "yeah that looks right." That doesn't scale, and it's easy to
fool yourself (we almost did, with the over-citation issue in Phase 5 -
the ANSWER looked perfect, and only manual digging revealed the citation
trail was partly wrong).

This phase replaces "looks right" with a real, repeatable test:
    - a small set of questions where WE know the correct answer
      (because it's our own document)
    - for each question, check TWO things separately:
        1. RETRIEVAL: did the correct chunk even get retrieved?
        2. ANSWER: does the final generated answer contain the right fact?

Why check these separately instead of just "is the final answer right"?
    If retrieval fails, the LLM never even SAW the right information - no
    amount of clever prompting fixes that. If retrieval succeeds but the
    answer is still wrong, that's a different problem (generation/prompt).
    Knowing WHICH one is failing tells you what to actually fix.

This is intentionally simple: we check for expected keywords/phrases in
the retrieved chunks and in the answer. Not perfect (a smarter eval would
use another LLM call to judge correctness), but honest, fast, and good
enough to catch real regressions as you keep changing the pipeline.
"""

from phase3_vector_store import build_vector_store, search
from phase4_generate import ask_with_context


# Each test case: a question about YOUR document, plus keywords that MUST
# appear somewhere in a correct answer. Keep these easy to verify by eye
# against the source PDF - that's what makes this trustworthy.
TEST_CASES = [
    {
        "question": "What was the Packet Delivery Ratio in the Python simulation?",
        "expected_keywords": ["95.9"],
    },
    {
        "question": "What obstacle detection method was used in the final Python phase?",
        "expected_keywords": ["YOLOv8", "webcam"],
    },
    {
        "question": "What three parameters does GWO tune for the ACO algorithm?",
        "expected_keywords": ["alpha", "beta", "rho"],
    },
    {
        "question": "What network simulator was used for wireless validation?",
        "expected_keywords": ["NS-3"],
    },
    {
        "question": "What was the average redundancy achieved in the Python simulation?",
        # Note: the source document itself reports two values for this -
        # "1.40" in the abstract/conclusion, "1.3735" in the Section 7
        # results table. Both are legitimately correct depending on which
        # part of the document is being referenced.
        "expected_keywords": ["1.40", "1.4", "1.37"],
    },
]


def keyword_found(text: str, keywords: list[str]) -> bool:
    """True if ANY of the expected keywords appear in text (case-insensitive)."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def run_evaluation(collection, top_k: int = 3):
    retrieval_hits = 0
    answer_hits = 0
    results = []

    for case in TEST_CASES:
        question = case["question"]
        expected = case["expected_keywords"]

        # Step 1: check retrieval - did the expected fact make it into the
        # retrieved chunks at all?
        matched_chunks, _ = search(collection, question, top_k=top_k)
        combined_context = " ".join(matched_chunks)
        retrieval_ok = keyword_found(combined_context, expected)

        # Step 2: check the actual generated answer
        answer = ask_with_context(collection, question, mode="qa", top_k=top_k)
        answer_ok = keyword_found(answer, expected)

        if retrieval_ok:
            retrieval_hits += 1
        if answer_ok:
            answer_hits += 1

        results.append({
            "question": question,
            "retrieval_ok": retrieval_ok,
            "answer_ok": answer_ok,
            "answer": answer,
        })

    return results, retrieval_hits, answer_hits


if __name__ == "__main__":
    pdf_path = "data/internship_report.pdf"
    collection = build_vector_store(pdf_path)

    print(f"Running evaluation on {len(TEST_CASES)} test cases...\n")
    results, retrieval_hits, answer_hits = run_evaluation(collection)

    for i, r in enumerate(results):
        retrieval_mark = "PASS" if r["retrieval_ok"] else "FAIL"
        answer_mark = "PASS" if r["answer_ok"] else "FAIL"
        print(f"Q{i + 1}: {r['question']}")
        print(f"  Retrieval: {retrieval_mark}   Answer: {answer_mark}")
        if not r["answer_ok"]:
            # Show the actual wrong/incomplete answer, so you can see WHY it failed
            print(f"  Got: {r['answer'][:200]}")
        print()

    total = len(TEST_CASES)
    print("=" * 50)
    print(f"Retrieval accuracy: {retrieval_hits}/{total} ({100 * retrieval_hits / total:.0f}%)")
    print(f"Answer accuracy:    {answer_hits}/{total} ({100 * answer_hits / total:.0f}%)")