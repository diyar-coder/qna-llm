# Project Concepts & Tech Stack

This document explains **what this project is**, **how the pieces fit together**, and **what each major tool/concept means**.

For setup and run instructions, see [`README.md`](./README.md).

---

## What this project is

A **study chatbot** where a student uploads their own course materials (PDF, DOCX, or TXT). The app:

1. Breaks the document into smaller pieces (“chunks”)
2. Converts those chunks into numerical vectors (“embeddings”)
3. Stores them in a vector database (**Chroma**)
4. When the student asks a question, finds the most relevant chunks
5. Sends those chunks + the question to an **LLM** (GPT) to generate an answer

The big idea: answers are grounded in **documents the student chose**, so the source is more trustworthy for studying.

---

## File structure

```text
qna-llm/
├── trivia_with_docs.py   # Main app: UI + RAG pipeline (load → chunk → embed → ask)
├── requirements.txt      # Python packages this project needs
├── README.md             # How to install and run
├── CONCEPTS.md           # This file — concepts & technologies
├── .devcontainer/        # Optional cloud/VS Code container setup
└── .streamlit/           # Local only (not committed): secrets like API keys
    └── secrets.toml
```

### What each important file does

| File | Role |
|---|---|
| `trivia_with_docs.py` | The whole Streamlit app: document upload, embeddings, Q&A, Trivia |
| `requirements.txt` | Lists libraries to install (`pip install -r requirements.txt`) |
| `.streamlit/secrets.toml` | Stores `OPENAI_API_KEY` locally (keep private; don’t push to GitHub) |
| `.devcontainer/` | Helps run the project in a consistent Dev Container / Codespaces environment |

---

## Core concepts

### 1. LLM (Large Language Model)

An LLM is an AI model trained to understand and generate text (here: **OpenAI GPT**, e.g. `gpt-3.5-turbo`).

In this project the LLM:
- Writes answers in **Q&A** mode
- Generates practice questions in **Trivia** mode

On its own, an LLM only knows what it was trained on. It does **not** automatically know your uploaded PDF. That’s why we use RAG.

### 2. RAG (Retrieval-Augmented Generation)

RAG = **retrieve** relevant document text first, then **generate** an answer using that text.

Flow in this app:

```text
User question
    ↓
Search Chroma for similar document chunks
    ↓
Give those chunks + question to the LLM
    ↓
LLM answer grounded in the uploaded document
```

Why it matters for studying: the model is guided by the student’s materials instead of inventing from general knowledge alone.

### 3. Tokens

Text is split into **tokens** (pieces of words) before models process it.

- More text ≈ more tokens ≈ higher API cost
- Embedding cost and chat cost both depend on tokens
- The app estimates embedding cost with **tiktoken**

### 4. Chunking

Documents are too long to embed/search as one giant blob, so we split them into **chunks**.

In code this uses LangChain’s `RecursiveCharacterTextSplitter`:
- `chunk_size` — how big each chunk is
- `chunk_overlap` — repeated text between chunks so meaning isn’t cut awkwardly at boundaries

Smaller chunks = more precise retrieval, sometimes less context  
Larger chunks = more context, sometimes noisier matches

### 5. Embeddings (vectorization)

An **embedding** turns text into a list of numbers (a vector) that represents meaning.

Similar meanings → vectors that are close together.

This app uses OpenAI’s `text-embedding-3-small`.

Example intuition:
- “constitution” and “founding document” → nearby vectors
- “constitution” and “banana bread recipe” → far apart

### 6. Vector database (Chroma)

**Chroma** stores embeddings and lets you search by similarity.

When you ask a question:
1. The question is embedded into a vector
2. Chroma finds the `k` closest document-chunk vectors
3. Those chunks are passed to the LLM

`k` (sidebar setting) = how many chunks to retrieve. Higher `k` → more context, usually more tokens/cost.

### 7. Similarity search

“Find chunks whose meaning is closest to this question.”

This project uses **similarity** search through LangChain’s retriever API on top of Chroma.

---

## Technologies used

### Python
The programming language for the whole project.  
Scripts, package installs, and the Streamlit app all run in Python.

### Streamlit
A Python framework for building interactive web apps quickly.

Used here for:
- File upload UI
- Sidebar controls (chunk size, `k`, menu)
- Q&A / Trivia pages
- Chat history display

Run command:
```bash
streamlit run trivia_with_docs.py
```

### LangChain
A framework that helps connect LLMs, document loaders, splitters, retrievers, and vector stores.

Used here for:
- Loading PDF / DOCX / TXT (`PyPDFLoader`, `Docx2txtLoader`, `TextLoader`)
- Splitting text into chunks
- Building a retrieval QA chain (`RetrievalQA`)
- Talking to OpenAI chat + embeddings models

### OpenAI
Provides:
- **Embeddings model** (`text-embedding-3-small`) — turns chunks into vectors
- **Chat model** (`gpt-3.5-turbo`) — generates answers / trivia

Requires an API key in `.streamlit/secrets.toml`.

### ChromaDB (`chromadb`)
The local vector database used to store and search document embeddings.

### Supporting packages

| Package | Why it’s here |
|---|---|
| `langchain` / `langchain_openai` / `langchain_community` | RAG building blocks + OpenAI integrations |
| `openai` | OpenAI API client (used under the hood) |
| `chromadb` | Vector store backend |
| `pypdf` | Read PDF files |
| `docx2txt` | Read Word `.docx` files |
| `tiktoken` | Count tokens / estimate embedding cost |
| `numpy` | Numerical support (used by ML/vector libraries) |
| `streamlit_option_menu` | Sidebar menu for Q&A vs Trivia |

---

## End-to-end path through the code

Main file: `trivia_with_docs.py`

| Step | Function / UI | What happens |
|---|---|---|
| 1 | `load_document()` | Reads uploaded PDF/DOCX/TXT into LangChain documents |
| 2 | `chunk_data()` | Splits documents into overlapping chunks |
| 3 | `calculate_embedding_cost()` | Estimates token/cost before embedding |
| 4 | `create_embeddings()` | Creates OpenAI embeddings and stores them in Chroma |
| 5 | Sidebar **Add Data** | Runs steps 1–4 and saves the vector store in `st.session_state` |
| 6 | `ask_and_get_answer()` | Retrieves top-`k` chunks and asks GPT for an answer |
| 7 | **Q&A mode** | User question → retrieved context → answer + history |
| 8 | **Trivia mode** | Uses retrieved document text to generate practice questions |

`st.session_state` keeps the vector store (and chat history) available while the Streamlit app is running.

---

## Why this architecture (study-bot angle)

| Approach | Pros | Cons |
|---|---|---|
| Plain ChatGPT (no docs) | Easy | May hallucinate; not tied to class materials |
| This RAG study bot | Grounded in student-chosen sources | Needs API key; embedding/chat cost; quality depends on chunking/`k` |

The product advantage is **source control**: students hand-pick what the bot can use.

---

## Glossary (quick)

- **Prompt** — instructions/text sent to the LLM  
- **Hallucination** — when a model invents facts  
- **Retriever** — component that finds relevant chunks  
- **Vector** — list of numbers representing meaning  
- **API key** — secret credential that authorizes OpenAI calls  
- **Virtual environment (`.venv`)** — isolated Python package install for this project  

---

## Related reading in this repo

- [`README.md`](./README.md) — install, run, troubleshooting
- `trivia_with_docs.py` — implementation of everything above
- `requirements.txt` — exact libraries installed for the app
