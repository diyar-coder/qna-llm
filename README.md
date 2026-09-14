# Study Q&A Chatbot (RAG)

An AI study assistant that answers questions **only from documents you upload**.

Students can choose their own PDFs, Word docs, or text files, so answers come from sources they trust — not the open internet.

Built with **Streamlit**, **LangChain**, **OpenAI**, and **Chroma** (RAG: retrieve relevant document chunks, then generate an answer).

For a deeper explanation of the file structure, LLM/RAG/Chroma concepts, and every major dependency, see **[`CONCEPTS.md`](./CONCEPTS.md)**.

- Stack: Streamlit, LangChain, OpenAI embeddings + chat, Chroma vector store
- Goal: let students ground study Q&A in self-selected course materials
- To run locally: clone → create venv → install requirements → add Streamlit secret → `streamlit run trivia_with_docs.py`
---

## Features

- Upload a `.pdf`, `.docx`, or `.txt` file
- Chunk + embed the document into a Chroma vector store
- Ask questions about the uploaded content (Q&A mode)
- Generate trivia questions from the document (Trivia mode)
- Adjust chunk size and retrieval `k` in the sidebar

---

## What you’ll need

1. **Python 3.10–3.12** (recommended)  
   - Avoid very new Python versions (e.g. 3.14) — some packages like Chroma may not install cleanly.
   - Check your version:
     ```bash
     python3 --version
     ```
   - Download from: https://www.python.org/downloads/

2. **An OpenAI API key**  
   - Create one at: https://platform.openai.com/api-keys  
   - You need a funded OpenAI account (embeddings + chat use paid API calls).

3. **Git** (to clone the repo)  
   - https://git-scm.com/downloads

4. A terminal (Terminal on Mac, PowerShell/Command Prompt on Windows)

---

## Setup (step by step)

### 1. Clone the repository

```bash
git clone https://github.com/diyar-coder/qna-llm.git
cd qna-llm
```

### 2. Create a virtual environment

**Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

You should see `(.venv)` at the start of your terminal prompt.

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If `chromadb` fails to install, confirm you’re on Python **3.10–3.12**, recreate the venv, and try again.

### 4. Add your OpenAI API key (Streamlit secrets)

This app reads the key from Streamlit secrets (`st.secrets["OPENAI_API_KEY"]`).

Create a folder and file:

```bash
mkdir -p .streamlit
```

Create `.streamlit/secrets.toml` with:

```toml
OPENAI_API_KEY = "sk-your-key-here"
```

Replace `sk-your-key-here` with your real key.

**Important:** never commit this file to GitHub. Keep your API key private.

### 5. Run the app

```bash
streamlit run trivia_with_docs.py
```

Streamlit will open in your browser (usually `http://localhost:8501`).

To stop the app, press `Ctrl + C` in the terminal.

---

## How to use the app

1. In the sidebar, upload a `.pdf`, `.docx`, or `.txt` file.
2. (Optional) Adjust **Chunk size** and **k**.
3. Click **Add Data** and wait for chunking/embedding to finish.
4. Choose a mode:
   - **Q&A** — ask a question about your document
   - **Trivia** — generate practice questions from the document

### Example document

This repo includes a sample file you can try right away:

- **`us_constitution.pdf`** — upload it, click **Add Data**, then ask a question (for example: “What are the three branches of government?”) or generate trivia.

Tips:
- Larger `k` retrieves more chunks (can improve answers, costs more tokens).
- Start with a short document while testing so embedding is cheap/fast.

---

## Project structure

```text
qna-llm/
├── trivia_with_docs.py   # Main Streamlit app (RAG chatbot)
├── requirements.txt      # Python dependencies
├── us_constitution.pdf   # Example document to upload and try
├── .devcontainer/        # Optional VS Code / Codespaces setup
└── README.md
```

---

## How it works (high level)

1. **Load** the uploaded document (PDF / DOCX / TXT)
2. **Split** it into overlapping text chunks
3. **Embed** chunks with OpenAI (`text-embedding-3-small`) and store them in **Chroma**
4. On each question, **retrieve** the most similar chunks
5. Send those chunks + the question to **GPT** to generate an answer

This is Retrieval-Augmented Generation (RAG): the model answers from *your* documents.

---

## Troubleshooting

| Problem | What to try |
|---|---|
| `No OpenAI API key` / secrets error | Make sure `.streamlit/secrets.toml` exists and contains `OPENAI_API_KEY` |
| `chromadb` install errors | Use Python 3.10–3.12 and a fresh venv |
| App opens but answers fail | Check that your OpenAI key is valid and your account has billing enabled |
| “Upload a document” warning | Upload a file **and** click **Add Data** before asking questions |
| Wrong Python / packages | Confirm the venv is activated (`(.venv)` in your prompt) |


---

## License / API costs

Running this app calls OpenAI APIs (embeddings + chat), which are billed to your OpenAI account. Keep an eye on usage while testing.
