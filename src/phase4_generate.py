"""
Phase 4: Generation - turn retrieved chunks into an actual answer.

This is where retrieval (Phase 3) and generation (Phase 0) come together
into real RAG:

    question -> [Phase 3: find relevant chunks] -> [Phase 4: LLM answers
    using ONLY those chunks as context] -> grounded answer

Why pass chunks as context instead of just asking Gemini directly?
    Ask Gemini "What obstacle detection method did Mrinalini's internship
    use?" with no context, and it has no idea - it's not your document.
    By retrieving the relevant chunks and including them in the prompt, we
    force the model to answer using YOUR document, not its general
    knowledge. This is what makes answers grounded and trustworthy instead
    of hallucinated.

We also support multiple MODES here (not just Q&A) - matching the
"NotebookLM-style" product goal: ask a question, or ask for a section to
be simplified/explained/summarized. Same retrieval step, different prompt.
"""

import os
from dotenv import load_dotenv
from google import genai

from phase3_vector_store import build_vector_store, search

load_dotenv()
client = genai.Client()


# Each mode has its own instruction to the model. Retrieved context and the
# user's input get slotted in the same way for all of them - only the
# INSTRUCTION changes. This is a clean way to support multiple behaviors
# without duplicating the retrieval/prompt-building logic.
MODE_INSTRUCTIONS = {
    "qa": (
        "Answer the question using ONLY the context below. "
        "If the context doesn't contain the answer, say so clearly - "
        "do not make anything up."
    ),
    "simplify": (
        "Explain the following context in simple, plain language, "
        "as if to someone with no background in this field. "
        "Avoid jargon where possible, or briefly define it if unavoidable."
    ),
    "summarize": (
        "Summarize the following context in 3-5 concise sentences, "
        "capturing only the most important points."
    ),
}


def build_prompt(mode: str, user_input: str, context_chunks: list[str]) -> str:
    """Combine the mode's instruction, the retrieved context, and the user's
    input into one prompt to send to the model."""
    instruction = MODE_INSTRUCTIONS[mode]
    context = "\n\n---\n\n".join(context_chunks)

    return f"""{instruction}

CONTEXT:
{context}

USER INPUT: {user_input}

ANSWER:"""


def ask_with_context(collection, user_input: str, mode: str = "qa", top_k: int = 3) -> str:
    """
    Full RAG pipeline for one turn:
    1. Retrieve the top_k most relevant chunks for user_input
    2. Build a prompt that includes those chunks as context
    3. Send it to Gemini and return the answer
    """
    matched_chunks, distances = search(collection, user_input, top_k=top_k)

    prompt = build_prompt(mode, user_input, matched_chunks)

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )

    return response.text


if __name__ == "__main__":
    pdf_path = "data/internship_report.pdf"
    collection = build_vector_store(pdf_path)

    # Test 1: a direct question (Q&A mode)
    question = "What obstacle detection method was used in the final Python phase?"
    print(f"[MODE: qa] Question: {question}\n")
    answer = ask_with_context(collection, question, mode="qa")
    print(f"Answer:\n{answer}\n")
    print("=" * 70)

    # Test 2: simplify mode, on a more technical query
    topic = "the Grey Wolf Optimizer algorithm"
    print(f"\n[MODE: simplify] Topic: {topic}\n")
    simplified = ask_with_context(collection, topic, mode="simplify")
    print(f"Simplified explanation:\n{simplified}")