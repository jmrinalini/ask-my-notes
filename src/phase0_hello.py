"""
Phase 0: Prove the plumbing works.

This script does ONE thing: send a message to Gemini and print the reply.
No RAG, no PDFs, no vector DB yet. Just: can we talk to the model?

We're using Google's Gemini API here because it has a genuine free tier
(no credit card, no charges) -- good for learning without cost pressure.

Concepts you're learning here:
- Loading secrets from a .env file (never hardcode API keys in code)
- Making your first API call
- Reading a response object
"""

import os
from dotenv import load_dotenv
from google import genai

# load_dotenv() reads the .env file and makes its variables available
# via os.environ, as if you'd set them in your terminal.
load_dotenv()

# genai.Client() automatically looks for GEMINI_API_KEY in the environment.
# We don't type the key anywhere in code - that's the whole point of .env.
client = genai.Client()


def ask_gemini(question: str) -> str:
    """Send a single question to Gemini and return the text reply."""
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=question
    )
    return response.text


if __name__ == "__main__":
    question = "In one sentence, what is Retrieval-Augmented Generation (RAG)?"
    print(f"Asking: {question}\n")

    answer = ask_gemini(question)
    print(f"Gemini says: {answer}")