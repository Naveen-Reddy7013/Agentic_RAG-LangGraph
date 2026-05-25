# Agentic RAG — LangGraph

A production-ready **Corrective RAG (CRAG)** system built with LangGraph that evaluates, corrects, and verifies every step of the pipeline before returning an answer.

---

## What Makes This Different from Basic RAG

| Basic RAG | This Project |
|---|---|
| Retrieve → Generate | Retrieve → Grade → Correct → Generate → Verify |
| No quality checks | LLM-as-judge at every step |
| Hallucinates silently | Detects and rejects hallucinations |
| No self-correction | Rewrites query when retrieval fails |
| Single pass | Self-correcting loop with retry limit |

---

## Architecture

```
START
  ↓
retrieve
  ↓
grade_documents
  ↓ (conditional)
  ├── web_search = "No"  → generate
  └── web_search = "Yes" → transform_query → retrieve (loop back)
                                    ↓ (conditional)
                            generate
                                ↓ (conditional)
                    ├── hallucination = "no"  → generate (retry)
                    ├── answer = "no"         → transform_query
                    └── both pass             → END ✓
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM | Groq API — llama-3.3-70b-versatile |
| Embeddings | HuggingFace — sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS (local) |
| Language | Python 3.11+ |

> **100% free stack** — no OpenAI billing, no cloud vector database.

---

## Production Features

- **Relevance Grading** — LLM judges each retrieved chunk before generation. Irrelevant chunks are filtered out.
- **Hallucination Detection** — Verifies every claim in the answer is grounded in retrieved documents.
- **Query Rewriting** — Rewrites the user question to improve semantic match when retrieval fails.
- **Answer Grading** — Validates the answer actually resolves the user question.
- **Retry Limiting** — Prevents infinite loops with a max retry counter (default: 3).
- **State Management** — Shared TypedDict state flows across all nodes.

---

## Project Structure

```
Agentic_RAG-LangGraph/
├── data/
│   └── rag_guide.docx          # Source document
├── graph_state.py              # Shared state schema (TypedDict)
├── ingestion.py                # Load, chunk, embed, save to FAISS
├── nodes.py                    # All node functions
├── graph.py                    # LangGraph workflow + routing
├── main.py                     # Entry point with user input loop
├── Agentic_RAG_LangGraph.pdf   # Architecture PDF
├── requirements.txt
├── pyproject.toml
└── .env.example                # API key template
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Naveen-Reddy7013/Agentic_RAG-LangGraph.git
cd Agentic_RAG-LangGraph
```

### 2. Create virtual environment and install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 4. Build the vector store

```bash
python ingestion.py
```

This loads `data/rag_guide.docx`, splits it into chunks, embeds them using HuggingFace, and saves the FAISS index to `vectorstore/faiss_index/`.

### 5. Run the RAG

```bash
python main.py
```

---

## Example

```
Ask your question (or type 'exit' to quit): what is hallucination in RAG?

==================================================
Question: what is hallucination in RAG?
==================================================

--- Output from node: retrieve ---
--- NODE: GRADE DOCUMENTS ---
Found 4 relevant documents
--- ROUTER: DECIDE TO GENERATE ---
Decision: generate answer
--- NODE: GENERATE ---
--- ROUTER: GRADE GENERATION ---
Decision: answer is grounded, checking if it resolves question
Decision: answer is useful, returning to user

==================================================
FINAL ANSWER:
==================================================
Hallucination in RAG refers to when the LLM generates facts not present
in the retrieved documents. The hallucination grader checks if every claim
in the answer is grounded in the retrieved context before returning it to the user.
```

---

## How It Works

### State

All nodes share a `GraphState` TypedDict:

```python
class GraphState(TypedDict):
    question: str       # User question
    documents: List     # Retrieved + filtered chunks
    generation: str     # Final LLM answer
    web_search: str     # "Yes"/"No" flag for query rewrite
    retries: int        # Loop prevention counter
```

### Nodes

| Node | LLM Call | Purpose |
|---|---|---|
| `retrieve` | No | Vector similarity search, top-4 chunks |
| `grade_documents` | Yes | Filter irrelevant chunks |
| `generate` | Yes | Produce grounded answer |
| `transform_query` | Yes | Rewrite question for better retrieval |
| `decide_to_generate` | No | Route based on web_search flag |
| `grade_generation` | Yes (x2) | Hallucination + answer quality check |

---

## Requirements

```
langchain
langchain-groq
langchain-community
langchain-huggingface
faiss-cpu
python-dotenv
sentence-transformers
langgraph
pydantic
docx2txt
```

---

## Architecture PDF

A detailed 6-page PDF covering the full architecture, tech stack, workflow diagram, and production features is included: `Agentic_RAG_LangGraph.pdf`

---

## License

MIT
