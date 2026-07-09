\# Ask My Notes



A NotebookLM-style app: upload a document, chat with it — ask questions,

get things explained or simplified, all grounded in and cited from the

actual document.



Built from scratch to learn how RAG (Retrieval-Augmented Generation) works

under the hood.



\## Progress

\- \[x] Phase 0: Setup \& first API call

\- \[ ] Phase 1: Ingest \& chunk documents

\- \[ ] Phase 2: Embeddings

\- \[ ] Phase 3: Vector search

\- \[ ] Phase 4: Multi-mode generation (Q\&A / Simplify / Summarize)

\- \[ ] Phase 5: Citations \& grounding

\- \[ ] Phase 6: Evaluation

\- \[ ] Phase 7: Better retrieval (stretch)

\- \[ ] Phase 8: Agent behavior (stretch)

\- \[ ] Phase 9: UI (Streamlit)



\## Setup



1\. Create a virtual environment:

&#x20;  ```

&#x20;  python3 -m venv venv

&#x20;  source venv/bin/activate   # on Windows: venv\\Scripts\\activate

&#x20;  ```



2\. Install dependencies:

&#x20;  ```

&#x20;  pip install -r requirements.txt

&#x20;  ```



3\. Get an API key from https://console.anthropic.com and add it:

&#x20;  ```

&#x20;  cp .env.example .env

&#x20;  ```

&#x20;  Then open `.env` and paste your real key in.



4\. Run the Phase 0 test:

&#x20;  ```

&#x20;  python src/phase0\_hello.py

&#x20;  ```



&#x20;  If it prints an answer about RAG, your setup works.

