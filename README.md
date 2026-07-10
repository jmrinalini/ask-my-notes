# Ask My Notes

A NotebookLM-style app: upload a document, chat with it — ask questions,
get things explained or simplified, all grounded in and cited from the
actual document.

Built from scratch to learn how RAG (Retrieval-Augmented Generation) works
under the hood.

## Progress
- [x] Phase 0: Setup & first API call
- [ ] Phase 1: Ingest & chunk documents
- [ ] Phase 2: Embeddings
- [ ] Phase 3: Vector search
- [ ] Phase 4: Multi-mode generation (Q&A / Simplify / Summarize)
- [x] Phase 5: Citations & grounding
      Note: found the model sometimes over-cites (tags a source that doesn't
      actually support a claim) even when the underlying facts are correct.
      Known limitation of prompt-based citation; something to measure
      properly in Phase 6.
- [ ] Phase 6: Evaluation
- [ ] Phase 7: Better retrieval (stretch)
- [ ] Phase 8: Agent behavior (stretch)
- [ ] Phase 9: UI (Streamlit)

## Setup

1. Create a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Get an API key from https://console.anthropic.com and add it:
   ```
   cp .env.example .env
   ```
   Then open `.env` and paste your real key in.

4. Run the Phase 0 test:
   ```
   python src/phase0_hello.py
   ```

   If it prints an answer about RAG, your setup works.