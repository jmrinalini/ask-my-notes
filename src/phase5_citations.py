"""
Phase 5: Citations and grounding.

Phase 4 gave us answers grounded in retrieved context - but "grounded"
was really just us TRUSTING the model used the right chunks. We had no
way to verify it. This phase closes that gap: the model tags which
source(s) it used for its answer, and we print those sources alongside
the answer so you can check them yourself.

Why this matters:
    This is the difference between "an AI told me X" and "an AI told me X,
    and here's the exact paragraph from my own document that supports it."
    The second one is trustworthy. NotebookLM's citation feature is exactly
    this idea - it's what separates a real RAG tool from a chatbot that
    might be hallucinating.

How it works:
    1. Label each retrieved chunk as [Source 1], [Source 2], etc.
    2. Instruct the model to reference source numbers when making claims
    3. Print the answer AND the full text of each source, so the person
       reading it can verify the claim against the original wording.
"""

import os
from dotenv import load_dotenv
from google import genai

from phase3_vector_store import build_vector_store, search

load_dotenv()
client = genai.Client()


CITED_QA_INSTRUCTION = (
    "Answer the question using ONLY the numbered sources below. "
    "After each claim or sentence in your answer, cite the source(s) "
    "it came from like this: [Source 1]. If a sentence draws on multiple "
    "sources, cite all of them, like [Source 1][Source 2]. "
    "If the sources don't contain the answer, say so clearly - do not "
    "make anything up or use outside knowledge."
)


def build_cited_prompt(question: str, context_chunks: list[str]) -> str:
    """Label each chunk as a numbered source and build the instruction prompt."""
    labeled_sources = "\n\n".join(
        f"[Source {i + 1}]\n{chunk}"
        for i, chunk in enumerate(context_chunks)
    )

    return f"""{CITED_QA_INSTRUCTION}

SOURCES:
{labeled_sources}

QUESTION: {question}

ANSWER (with citations):"""


def ask_with_citations(collection, question: str, top_k: int = 3):
    """
    Full cited RAG pipeline:
    1. Retrieve top_k chunks
    2. Ask the model to answer WITH citations back to source numbers
    3. Return both the cited answer and the original chunks, so the
       caller can display "Source 1 says: ..." if the person wants to
       verify a specific citation.
    """
    matched_chunks, distances = search(collection, question, top_k=top_k)
    prompt = build_cited_prompt(question, matched_chunks)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text, matched_chunks


if __name__ == "__main__":
    pdf_path = "data/internship_report.pdf"
    collection = build_vector_store(pdf_path)

    question = "What were the final performance metrics achieved in the Python simulation?"
    print(f"Question: {question}\n")

    answer, sources = ask_with_citations(collection, question, top_k=3)

    print(f"Answer:\n{answer}\n")
    print("=" * 70)
    print("\nSOURCE TEXTS (to verify the citations above):\n")

    for i, source in enumerate(sources):
        print(f"--- Source {i + 1} ---")
        print(source[:350] + ("..." if len(source) > 350 else ""))
        print()