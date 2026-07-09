"""
Phase 1: Ingest a PDF and split it into chunks.

Two jobs here:
1. extract_text_from_pdf() - pull raw text out of a PDF file
2. chunk_text()            - split that text into ~N-word pieces WITHOUT
                              cutting a sentence in half

Why not just split every 500 characters?
    Because "The failure rate was reduced to 4.1% under the ACO-G" | "WO
    hybrid algorithm..." is two broken, meaningless fragments. If we later
    embed and search over broken fragments, retrieval quality suffers -
    the model gets to see a half-sentence instead of a complete idea.

Our approach: split into sentences first, then GROUP sentences together
until we hit a target word count, then start a new chunk. This keeps every
chunk made of whole sentences.
"""

import re
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> str:
    """Read a PDF file and return all its text as one big string."""
    reader = PdfReader(pdf_path)
    all_text = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:  # some pages (e.g. pure images) may return empty text
            all_text.append(page_text)

    # Join pages with a space so words at page boundaries don't get glued together
    return " ".join(all_text)


def split_into_sentences(text: str) -> list[str]:
    """
    Naive sentence splitter: breaks on '.', '?', '!' followed by a space
    and a capital letter or newline. Not perfect (technical PDFs are messy -
    abbreviations, decimals like "4.1%", bullet points), but good enough to
    start with. We'll improve this if Phase 6 evaluation shows it's hurting
    retrieval quality.
    """
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.?!])\s+', text)
    # Remove empty/whitespace-only fragments
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, target_words: int = 500, overlap_sentences: int = 1) -> list[str]:
    """
    Group sentences into chunks of roughly `target_words` words each.

    overlap_sentences: how many sentences from the END of one chunk get
    repeated at the START of the next chunk. This helps when an important
    idea spans a chunk boundary - without overlap, a question about that
    idea might miss half the context. Small overlap (1-2 sentences) is a
    common practical trick in RAG systems.
    """
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk_sentences = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())

        # If adding this sentence would exceed our target, close off the
        # current chunk and start a new one.
        if current_word_count + sentence_word_count > target_words and current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

            # Start the new chunk with the last few sentences of the old one (overlap)
            overlap = current_chunk_sentences[-overlap_sentences:] if overlap_sentences else []
            current_chunk_sentences = overlap + [sentence]
            current_word_count = sum(len(s.split()) for s in current_chunk_sentences)
        else:
            current_chunk_sentences.append(sentence)
            current_word_count += sentence_word_count

    # Don't forget the last chunk
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks


if __name__ == "__main__":
    pdf_path = "data/internship_report.pdf"

    print(f"Reading: {pdf_path}\n")
    raw_text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(raw_text)} characters, {len(raw_text.split())} words total.\n")

    chunks = chunk_text(raw_text, target_words=500)
    print(f"Split into {len(chunks)} chunks.\n")

    # Show the first 2 chunks so you can SEE what a chunk actually looks like
    for i, chunk in enumerate(chunks[:2]):
        word_count = len(chunk.split())
        print(f"--- Chunk {i + 1} ({word_count} words) ---")
        print(chunk[:400] + ("..." if len(chunk) > 400 else ""))
        print()