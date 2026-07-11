"""
Phase 8: Agent behavior - deciding WHEN to retrieve.

Every phase so far assumed every question needs document retrieval. But
think about how you'd actually use this app: sometimes you ask something
about your document ("what was the PDR?"), and sometimes you ask something
totally unrelated ("hi", "what's 2+2", "who are you?"). Running full
retrieval for those wastes time and can even confuse the answer (imagine
retrieving platoon chunks to answer "what's 2+2").

This is a first taste of "agentic" behavior: instead of a fixed pipeline
(always retrieve -> always generate), the system makes a DECISION first -
"does this question need my document, or can I just answer directly?" -
and only takes the retrieval path when needed.

How we implement the decision:
    We ask the model itself to classify the question, with a very
    constrained prompt (answer ONLY "YES" or "NO"). This is a common,
    simple pattern for building decision points into an LLM pipeline -
    force a narrow, parseable output instead of open-ended text.
"""

import os
from dotenv import load_dotenv
from google import genai

from phase3_vector_store import build_vector_store
from phase4_generate import ask_with_context

load_dotenv()
client = genai.Client()


CLASSIFY_PROMPT = """You are deciding whether a question requires looking up a \
specific personal document (an internship report about vehicle platoon \
communication, ACO-GWO algorithms, and obstacle detection) to answer accurately.

Answer with ONLY one word: YES or NO.

YES if the question is about the specific document's content, findings, \
methods, or results.
NO if the question is general knowledge, small talk, or unrelated to the \
document (even if it sounds technical in general).

Question: {question}

Answer (YES or NO only):"""


def needs_retrieval(question: str) -> bool:
    """Ask the model to classify whether this question needs document
    retrieval. Returns True/False."""
    prompt = CLASSIFY_PROMPT.format(question=question)

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )

    decision = response.text.strip().upper()
    return decision.startswith("YES")


def answer_directly(question: str) -> str:
    """No retrieval - just answer from the model's general knowledge,
    with a note that this wasn't grounded in the document."""
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=question
    )
    return response.text


def agent_answer(collection, question: str) -> str:
    """
    The agent's main loop:
    1. Decide if retrieval is needed
    2. Route to either the RAG pipeline (Phase 4) or a direct answer
    3. Label which path was taken, so it's transparent what happened
    """
    if needs_retrieval(question):
        print("[agent decision: RETRIEVE - question relates to the document]")
        answer = ask_with_context(collection, question, mode="qa")
    else:
        print("[agent decision: SKIP RETRIEVAL - general question, answering directly]")
        answer = answer_directly(question)

    return answer


import time

if __name__ == "__main__":
    pdf_path = "data/internship_report.pdf"
    collection = build_vector_store(pdf_path)

    test_questions = [
        "What obstacle detection method was used in the final Python phase?",  # should RETRIEVE
        "Hi, how are you?",                                                     # should SKIP
        "What is 15 times 7?",                                                  # should SKIP
        "What was the Packet Delivery Ratio achieved?",                         # should RETRIEVE
    ]

    for question in test_questions:
        print(f"\nQuestion: {question}")
        answer = agent_answer(collection, question)
        print(f"Answer: {answer}\n")
        print("-" * 70)

        # Free tier allows only 5 requests/minute, and each question here
        # makes 2 calls (classify + answer). A short pause avoids hitting
        # that limit when running several questions back to back.
        time.sleep(15)