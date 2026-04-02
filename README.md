# 🧠 SLM Data Management Tool

A local AI-powered document and data querying tool built with **FastAPI**, **Ollama**, and **FAISS**. Upload your files, ask questions in plain English, and get accurate, data-backed answers — all running locally on your machine with no cloud dependencies.

---

## 🖥 Developer Hardware

This project was built and tested on:

| Component | Spec                             |
| --------- | -------------------------------- |
| CPU       | Intel Core i9                    |
| RAM       | 64 GB                            |
| GPU       | NVIDIA RTX ADA 3500 — 12 GB VRAM |
| OS        | Windows 11                       |

The default model (`wen2.5:14b`) runs comfortably on this setup using Ollama's GPU offloading. If your hardware differs, see the [Model Selection Guide](#-model-selection-guide) below.

---

## ✨ What It Does

This tool handles two types of files with two different pipelines:

| File Type                      | Pipeline            | How It Works                                                                |
| ------------------------------ | ------------------- | --------------------------------------------------------------------------- |
| `.csv`, `.xlsx`, `.xls`        | **Pandas Pipeline** | LLM generates pandas code → executes against full dataset → narrates result |
| `.pdf`, `.txt`, `.docx`, `.md` | **RAG Pipeline**    | Text chunked → embedded with FAISS → top chunks retrieved → LLM answers     |

The key advantage over standard RAG for tabular data: **the entire dataset is used for computation**, not just a few retrieved rows. This means aggregations, counts, percentages, and comparisons are always accurate.

---

## 🧠 How The Approach Works

Understanding why this tool uses two different pipelines is important — they solve fundamentally different problems.

### The Problem With Using RAG on CSV Files

Standard RAG works by splitting a document into chunks, embedding them, and retrieving the most "similar" chunks to the query. This works well for documents where the answer lives in a specific passage.

For tabular data, this breaks down completely:

- A question like _"what is the drop-off rate?"_ requires scanning **all 220 rows** and computing a percentage
- RAG only retrieves the top 8 most similar rows — which will be biased toward rows where the value is `1` (present), completely missing rows where the value is `0` (absent)
- The LLM then sees only positive examples and incorrectly concludes there is no drop-off

This is not a model quality problem — it is a **fundamental architectural mismatch** between retrieval-based systems and aggregation-based queries.

---

### Pipeline 1 — Tabular Data (CSV / Excel)

```
User Query: "What is the avatar creation drop-off?"
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  STEP 1: Build Schema Context                       │
│                                                     │
│  Scan the dataframe once at upload time.            │
│  For each column, extract:                          │
│    - Column name and data type                      │
│    - Null count                                     │
│    - Min / max / mean  (numeric columns)            │
│    - Sample unique values  (string columns)         │
│                                                     │
│  This gives the LLM enough knowledge to write       │
│  correct pandas code without seeing the raw data.   │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  STEP 2: LLM Generates Pandas Code                 │
│                                                     │
│  The schema context + user query are sent to the    │
│  LLM with a strict prompt:                          │
│    - "df is already loaded, do not reload it"       │
│    - "assign your answer to a variable: result"     │
│    - "write only code, no explanation"              │
│                                                     │
│  LLM output (example):                             │
│    result = (df['Avatar Created'] == 0).sum()       │
│             / len(df) * 100                         │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  STEP 3: Execute in Isolated Sandbox                │
│                                                     │
│  The generated code runs in a subprocess:           │
│    - Completely separate from the FastAPI process   │
│    - Hard timeout (10 seconds) — no infinite loops  │
│    - Uses sys.executable so the virtualenv is       │
│      inherited (pandas, numpy available)            │
│    - df serialized to a temp CSV → loaded fresh     │
│      inside subprocess → temp dir deleted after     │
│                                                     │
│  If execution fails (SyntaxError, wrong column      │
│  name, etc.) the error is sent back to the LLM      │
│  and it tries again. Maximum 3 attempts.            │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  STEP 4: Narrate the Result                         │
│                                                     │
│  The raw computed result (e.g. 35.45) is sent to    │
│  the LLM with a strict narration prompt:            │
│    - "report ONLY what the numbers say"             │
│    - "do NOT add context or interpretation"         │
│    - "do NOT say 'this suggests' or 'this means'"   │
│                                                     │
│  Output: "The Avatar Creation drop-off is 35.45%"   │
└─────────────────────────────────────────────────────┘
```

**Why this works:** The LLM never sees raw data rows. It only sees the schema (to write code) and the computed result (to narrate). All actual computation happens deterministically in pandas against the full dataset.

---

### Pipeline 2 — Document Data (PDF / TXT / DOCX)

```
User Query: "How do I reset a user password?"
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  STEP 1: Chunk the Document                         │
│                                                     │
│  Text is extracted from the file and split into     │
│  overlapping chunks of ~300 characters each.        │
│  Smaller chunks = more precise retrieval.           │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  STEP 2: Embed and Index with FAISS                 │
│                                                     │
│  Each chunk is converted to a vector embedding      │
│  using the embedding model (qwen3-embedding:4b).    │
│                                                     │
│  Vectors are stored in a FAISS HNSW index:          │
│    - HNSW = Hierarchical Navigable Small World      │
│    - Approximate nearest-neighbor search            │
│    - Fast even with thousands of chunks             │
│                                                     │
│  Index is saved to disk — rebuilt only when the     │
│  file changes (cached on subsequent requests).      │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  STEP 3: Retrieve Relevant Chunks                   │
│                                                     │
│  The user query is embedded using the same model.   │
│  FAISS finds the top-8 chunks with highest cosine   │
│  similarity to the query vector.                    │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  STEP 4: LLM Answers From Context                   │
│                                                     │
│  The 8 retrieved chunks are injected into a         │
│  strict system prompt:                              │
│    - "your ONLY source is the context below"        │
│    - "if not in context, say so explicitly"         │
│    - "do not use outside knowledge"                 │
│                                                     │
│  The LLM generates an answer grounded only in       │
│  the retrieved document passages.                   │
└─────────────────────────────────────────────────────┘
```

**Why this works:** Document questions are about finding the right passage, not computing across all passages. Retrieval-based search is fast and accurate for this, and the strict prompt prevents the LLM from going beyond what the document says.

---

## 🗂 Supported File Types

- **Tabular**: `.csv`, `.xlsx`, `.xls`
- **Documents**: `.pdf`, `.txt`, `.docx`, `.md`
- Mixed uploads work — tabular and document files can be sent in the same request

---

## 🚀 Use Cases

### 📊 Tabular / CSV Data

- _"What is the Avatar Creation drop-off rate?"_
- _"Which hero was selected the most?"_
- _"How many users had invalid scans?"_
- _"What percentage of users completed all their games?"_
- _"What is the average pack purchase per user?"_

### 📄 Document / PDF Data

- _"What are the steps to reset a password in the admin panel?"_
- _"Summarize the key points of this report"_
- _"What does the document say about onboarding?"_

### 🔀 Mixed Queries

Send both a CSV and a PDF together:

- _"Based on the engagement data and the product guide, what features had low adoption?"_

---

## 🛠 Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- Pull the required models:

```bash
ollama pull qwen3-embedding:4b
ollama pull wen2.5:14b
```

---

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/slm-data-management-tool.git
cd slm-data-management-tool

# 2. Create and activate a virtual environment
python -m venv myenv

# Windows
myenv\Scripts\activate

# macOS / Linux
source myenv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
fastapi
uvicorn[standard]
faiss-cpu
numpy
ollama
pandas
openpyxl
python-docx
pymupdf
pydantic
python-multipart
```

---

### Running the Server

```bash
python app.py
```

Server starts at: `http://localhost:8000`

---

## 📡 API Usage

### Endpoint

```
POST /query
Content-Type: multipart/form-data
```

### Fields

| Field   | Type    | Description                        |
| ------- | ------- | ---------------------------------- |
| `files` | File(s) | One or more files to query against |
| `query` | String  | Your question in plain English     |

### Example — cURL

```bash
curl -X POST http://localhost:8000/query \
  -F "files=@data.csv" \
  -F "query=What is the avatar creation drop-off rate?"
```

### Example — Multiple Files

```bash
curl -X POST http://localhost:8000/query \
  -F "files=@engagement_data.csv" \
  -F "files=@product_guide.pdf" \
  -F "query=What features had the lowest engagement?"
```

### Example — Postman

1. Set method to `POST`, URL to `http://localhost:8000/query`
2. Go to **Body** → select **form-data**
3. Add key `files` (type: File) → attach your file
4. Add key `query` (type: Text) → type your question
5. Hit **Send**

### Response

```json
{
  "answer": "The Avatar Creation drop-off is 35.45%."
}
```

---

## ⚙️ Configuration

All key settings are at the top of `app.py`:

```python
EMBEDDING_MODEL     = "qwen3-embedding:4b"   # Ollama embedding model
LANGUAGE_MODEL      = "wen2.5:14b"      # Ollama language model
MAX_CHARS_PER_CHUNK = 300                    # RAG chunk size (for documents)
EMBED_BATCH_SIZE    = 16                     # Embedding batch size
SANDBOX_TIMEOUT     = 10                     # Max seconds for pandas code execution
```

---

## 🔄 Model Selection Guide

This tool works with **any model available in Ollama** — just change `LANGUAGE_MODEL` and/or `EMBEDDING_MODEL` in `app.py`. You are not locked into any specific model.

```python
# Examples — swap either of these to any Ollama model you have pulled
LANGUAGE_MODEL  = "wen2.5:14b"   # change this
EMBEDDING_MODEL = "qwen3-embedding:4b" # change this
```

### Language Model Recommendations

| Hardware                       | Recommended Model               | Notes                                        |
| ------------------------------ | ------------------------------- | -------------------------------------------- |
| 12 GB VRAM (e.g. RTX ADA 3500) | `wen2.5:14b`                    | Best reasoning quality, used by this project |
| 8 GB VRAM                      | `llama3.1:8b` or `qwen2.5:7b`   | Good balance of speed and accuracy           |
| 4–6 GB VRAM                    | `qwen2.5:3b` or `phi3:mini`     | Lightweight, faster responses                |
| CPU only (no GPU)              | `llama3.2:1b` or `gemma2:2b`    | Slow but functional                          |
| High-end GPU (24 GB+)          | `qwen2.5:72b` or `llama3.1:70b` | Maximum accuracy                             |

### Embedding Model Recommendations

| Model                | Notes                                           |
| -------------------- | ----------------------------------------------- |
| `qwen3-embedding:4b` | Used by this project, good multilingual support |
| `mxbai-embed-large`  | Strong English performance                      |
| `nomic-embed-text`   | Lightweight, fast                               |
| `bge-m3`             | Best for multilingual documents                 |

### How to Switch

```bash
# Pull any model you want
ollama pull llama3.1:8b

# Then update app.py
LANGUAGE_MODEL = "llama3.1:8b"
```

The rest of the code requires no changes — the pipeline is model-agnostic.

---

## 🔒 Security Notes

- LLM-generated pandas code runs in an **isolated subprocess** with a hard timeout
- The subprocess uses a temp directory that is deleted after each request
- Each user request gets a **unique UUID workspace** — files are not shared between requests
- The subprocess uses `sys.executable` so it stays within the virtualenv

> ⚠️ This tool is intended for **local/internal use**. Do not expose port 8000 to the public internet without adding authentication.

---

## 📁 Project Structure

```
slm-data-management-tool/
├── app.py                  # Main FastAPI application
├── requirements.txt        # Python dependencies
├── user_data/              # Auto-created: per-request file workspaces
│   └── <uuid>/
│       └── <filename>/
│           ├── *_chunks.db       # SQLite chunk store (RAG)
│           └── *_hnsw.index      # FAISS vector index (RAG)
└── README.md
```

---

## 🐛 Troubleshooting

**`ModuleNotFoundError: No module named 'pandas'` in subprocess**
Make sure you're running the server with your virtualenv activated. The subprocess inherits `sys.executable` from the running Python.

**`ollama.ResponseError: model not found`**
Pull the required models first:

```bash
ollama pull qwen3-embedding:4b
ollama pull wen2.5:14b
```

**Slow responses**
`wen2.5:14b` is a large model. Switch to `llama3.1:8b` in `app.py` for faster responses on lower-end hardware.

**Server running old code after edits**
If using `reload=True`, touch the file to trigger a reload:

```powershell
# Windows
(Get-Item .\app.py).LastWriteTime = Get-Date
```

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

MIT
