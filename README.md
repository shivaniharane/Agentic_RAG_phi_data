# 📄🤖 Agentic RAG — PDF Knowledge Agent with LanceDB & Phidata

> Build an intelligent agent that reads your PDF documents, stores them in a vector database, and answers questions by combining document knowledge with real-time web search — all with persistent conversation memory.

---

## 📌 Table of Contents

- [What is this project?](#-what-is-this-project)
- [What problem does it solve?](#-what-problem-does-it-solve)
- [RAG vs Agentic RAG — Key Difference](#-rag-vs-agentic-rag--key-difference)
- [How it works — Full Pipeline](#-how-it-works--full-pipeline)
- [Flowcharts](#-flowcharts)
- [Project Structure](#-project-structure)
- [Tech Stack & Tools](#-tech-stack--tools)
- [Component Deep Dive](#-component-deep-dive)
- [Setup & Installation](#-setup--installation)
- [Running on Google Colab](#-running-on-google-colab)
- [Configuration & Customization](#-configuration--customization)
- [Sample Outputs](#-sample-outputs)
- [Limitations](#-limitations)
- [How Limitations Can Be Resolved](#-how-limitations-can-be-resolved)
- [Key Concepts for Beginners](#-key-concepts-for-beginners)
- [Use Cases](#-use-cases)
- [Contributing](#-contributing)

---

## 🧠 What is this project?

This project is an **Agentic RAG (Retrieval-Augmented Generation) system** built with the [Phidata](https://docs.phidata.com/) framework. It creates an AI agent that:

1. **Reads a PDF** (the GPT-4 research paper in this example)
2. **Chunks and indexes** the document into a local vector database (LanceDB)
3. **Answers questions** by retrieving relevant chunks from the PDF
4. **Searches the web** via DuckDuckGo when PDF knowledge is insufficient
5. **Remembers conversations** across multiple queries using SQLite storage

Think of it as a **personal research assistant** — give it a document, and it can answer any question about it intelligently, knowing when to look at the document vs. when to search the internet.

---

## 💡 What problem does it solve?

### The core problem with standard LLMs:
LLMs like GPT-4 have a **knowledge cutoff date** and cannot read your private documents. If you ask GPT-4 "What are the exact benchmark scores in this research paper?", it cannot answer unless the paper was in its training data.

### The core problem with naive search:
Simple keyword search over documents misses context. Searching for "visual input" in a PDF won't find paragraphs that say "multimodal capabilities" — even though they mean the same thing.

### What Agentic RAG solves:

| Problem | Solution |
|---|---|
| LLM doesn't know your document | PDF is chunked and indexed → agent retrieves relevant sections |
| LLM knowledge is outdated | DuckDuckGo tool provides real-time web search |
| Agent forgets previous questions | SQLite storage persists full conversation history |
| Single retrieval step misses nuance | Agent decides dynamically whether to use document, web, or both |

---

## 🔄 RAG vs Agentic RAG — Key Difference

```
Standard RAG:
─────────────
User Query
    │
    ▼
Vector DB Search ──── always triggered ──── fixed retrieval
    │
    ▼
Stuff chunks into LLM prompt
    │
    ▼
LLM generates answer
    │
    ▼
Response (no memory, no web search, no decision-making)


Agentic RAG (this project):
───────────────────────────
User Query
    │
    ▼
Agent THINKS: "Do I need the PDF? The web? Both? Neither?"
    │
    ├──── PDF retrieval (if question is about the document)
    ├──── Web search via DuckDuckGo (if real-time info needed)
    ├──── Both (if question spans document + current context)
    └──── Neither (if answer is already in conversation memory)
    │
    ▼
Agent synthesizes all sources into a coherent answer
    │
    ▼
Response + conversation saved to SQLite for future reference
```

**The key difference**: In standard RAG, retrieval is always triggered mechanically. In Agentic RAG, the agent **decides** what to retrieve, from where, and how to combine sources — just like a human researcher would.

---

## ⚙️ How it works — Full Pipeline

### Phase 1: Ingestion (runs once)
```
PDF file on disk
      │
      ▼
PDFReader (chunk=True)
      │  Splits PDF into smaller overlapping text chunks
      │  Each chunk ≈ a paragraph or section
      ▼
LanceDB Vector Database
      │  Stores chunks as searchable records
      │  Table name: "documents"
      │  Storage: /tmp/lancedb (local file)
      ▼
Knowledge Base Ready
```

### Phase 2: Querying (runs on every user question)
```
User Question
      │
      ▼
Phidata Agent (gpt-4o-mini)
      │
      │  Step 1: Checks conversation history (SQLite)
      │           "Have I answered something related before?"
      │
      │  Step 2: Searches PDF Knowledge Base (LanceDB)
      │           Keyword search across all stored chunks
      │           Returns top-N most relevant chunks
      │
      │  Step 3: Decides if web search needed (DuckDuckGo)
      │           "Is there recent/external info I should check?"
      │
      ▼
LLM synthesizes: history + PDF chunks + web results
      │
      ▼
Streamed markdown response to user
      │
      ▼
Conversation saved to SQLite (data.db)
```

---

## 🗺️ Flowcharts

### Complete System Architecture

```
╔══════════════════════════════════════════════════════════════╗
║              Agentic RAG — Full Architecture                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ┌─────────────────────────────────────────────────────┐    ║
║  │              INGESTION PIPELINE (once)               │    ║
║  │                                                      │    ║
║  │  PDF File ──▶ PDFReader ──▶ Chunker ──▶ LanceDB     │    ║
║  │  (gpt-4.pdf)   (parse)    (split)    (index/store)  │    ║
║  └─────────────────────────────────────────────────────┘    ║
║                          │                                   ║
║                          │ pdf_knowledge_base.load()         ║
║                          ▼                                   ║
║  ┌─────────────────────────────────────────────────────┐    ║
║  │              QUERY PIPELINE (each question)          │    ║
║  │                                                      │    ║
║  │  User Question                                       │    ║
║  │       │                                              │    ║
║  │       ▼                                              │    ║
║  │  ┌─────────────────────────────────────────────┐    │    ║
║  │  │           Phidata Agent                      │    │    ║
║  │  │          (gpt-4o-mini)                       │    │    ║
║  │  │                                              │    │    ║
║  │  │  ┌──────────────┐  ┌──────────────────────┐ │    │    ║
║  │  │  │  SQLite      │  │  Decision Engine      │ │    │    ║
║  │  │  │  Storage     │  │                       │ │    │    ║
║  │  │  │              │  │  Use PDF? ──▶ LanceDB │ │    │    ║
║  │  │  │  Conversation│  │  Use Web? ──▶ DuckDDG │ │    │    ║
║  │  │  │  History     │  │  Use Both?            │ │    │    ║
║  │  │  └──────────────┘  └──────────────────────┘ │    │    ║
║  │  │                                              │    │    ║
║  │  └─────────────────────────────────────────────┘    │    ║
║  │       │                                              │    ║
║  │       ▼                                              │    ║
║  │  Streamed Markdown Response                          │    ║
║  └─────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════╝
```

---

### PDF Ingestion Pipeline (Detailed)

```
  gpt-4.pdf (on disk)
        │
        ▼
  ┌─────────────────────────────────────────┐
  │           PDFReader(chunk=True)          │
  │                                          │
  │  Page 1 ──┐                             │
  │  Page 2 ──┼──▶ Text extraction          │
  │  Page 3 ──┘    │                        │
  │                ▼                        │
  │         Chunking                        │
  │         ┌─────────────────────────┐     │
  │         │ Chunk 1: "GPT-4 is a   │     │
  │         │  large multimodal..."   │     │
  │         ├─────────────────────────┤     │
  │         │ Chunk 2: "We evaluate  │     │
  │         │  GPT-4 on a variety..."│     │
  │         ├─────────────────────────┤     │
  │         │ Chunk 3: "GPT-4 can    │     │
  │         │  accept image inputs..."│     │
  │         └─────────────────────────┘     │
  └─────────────────────────────────────────┘
        │
        ▼ Each chunk stored as a record
  ┌─────────────────────────────────────────┐
  │              LanceDB                     │
  │  Table: "documents"                      │
  │  Path:  /tmp/lancedb                     │
  │  Search: Keyword (BM25-style)            │
  │                                          │
  │  ┌─────┬────────────────────────────┐   │
  │  │ id  │ content                    │   │
  │  ├─────┼────────────────────────────┤   │
  │  │  1  │ "GPT-4 is a large multi..." │  │
  │  │  2  │ "We evaluate GPT-4 on..."  │   │
  │  │  3  │ "GPT-4 can accept image..." │  │
  │  │ ... │ ...                         │  │
  │  └─────┴────────────────────────────┘   │
  └─────────────────────────────────────────┘
```

---

### Agent Decision Flow (Query Time)

```
  User: "Does GPT-4 accept visual inputs?"
                    │
                    ▼
          ┌─────────────────┐
          │  Check SQLite   │
          │  History        │
          │  (any context?) │
          └────────┬────────┘
                   │ No prior context
                   ▼
          ┌─────────────────┐
          │  Search LanceDB │
          │  for "visual    │
          │   inputs"       │
          └────────┬────────┘
                   │ Returns: Chunk 3
                   │ "GPT-4 can accept image..."
                   ▼
          ┌──────────────────────────────┐
          │  Is web search needed?        │
          │                               │
          │  This question is about the   │
          │  PDF content specifically →   │
          │  NO, PDF chunk is sufficient  │
          └────────┬─────────────────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  LLM synthesizes│
          │  answer from    │
          │  PDF chunk      │
          └────────┬────────┘
                   │
                   ▼
  "Yes! GPT-4 is a multimodal model that
   can accept both text and image inputs..."
                   │
                   ▼
          ┌─────────────────┐
          │  Save to SQLite │
          │  data.db        │
          └─────────────────┘
```

---

### Memory & Persistence Flow

```
  First Session:
  ──────────────
  Q1: "Does GPT-4 accept visual inputs?"
      → Agent answers → saved to data.db

  Q2: "What is the comparison of GPT-4 and GPT-3.5?"
      → Agent checks data.db → sees Q1 context
      → Searches PDF for comparison data
      → Synthesizes answer with full context
      → Saved to data.db

  Second Session (restart notebook):
  ────────────────────────────────────
  Q3: "What did we discuss earlier?"
      → Agent loads data.db
      → Sees Q1 and Q2 history
      → "We discussed GPT-4's visual inputs
         and its comparison with GPT-3.5..."
```

---

## 📁 Project Structure

```
agentic-rag-phidata/
│
├── Agentic_RAG_phi_data.ipynb   # Main Colab notebook
├── requirements.txt              # All Python dependencies
├── README.md                     # This file
│
├── Data/                         # Auto-created by notebook
│   └── gpt-4.pdf                 # Downloaded GPT-4 paper (auto-downloaded)
│
├── data.db                       # SQLite agent memory (auto-created)
│   └── table: "data"             # Conversation history
│
└── /tmp/lancedb/                 # LanceDB vector store (auto-created)
    └── table: "documents"        # Chunked PDF content
```

> ⚠️ `data.db` and `/tmp/lancedb/` are auto-generated at runtime. Do not commit them to GitHub — add to `.gitignore`.

---

## 🛠️ Tech Stack & Tools

| Tool / Library | Version | Purpose |
|---|---|---|
| **Phidata** | ≥ 2.7.0 | Agent framework — orchestrates LLM + tools + memory |
| **LanceDB** | ≥ 0.6.0 | Local vector database for PDF chunk storage |
| **OpenAI** | ≥ 1.30.0 | GPT-4o-mini LLM for reasoning and generation |
| **DuckDuckGo Search** | ≥ 5.0.0 | Free real-time web search tool |
| **pypdf** | ≥ 4.0.0 | PDF parsing and text extraction |
| **tantivy** | ≥ 0.21.0 | Rust-based full-text search (powers LanceDB keyword search) |
| **pylance** | latest | LanceDB Python interface |
| **SQLite** | built-in | Persistent agent conversation history |
| **Python** | 3.10+ | Runtime environment |
| **Google Colab** | — | Cloud notebook execution environment |

---

### Why Phidata?

Phidata is an **agent framework** that makes it easy to build production-ready AI agents without boilerplate. Key advantages over raw LangChain or OpenAI SDK:

- **Built-in RAG**: Pass a `knowledge` base to any agent — retrieval happens automatically
- **Built-in storage**: Pass a `storage` backend — conversation history is persisted automatically
- **Built-in tool use**: Pass `tools=[]` — the agent decides when to call them
- **Minimal code**: The entire agent in this project is defined in ~10 lines

| Feature | Phidata | Raw OpenAI SDK | LangChain |
|---|---|---|---|
| Built-in RAG | ✅ | ❌ Manual | ⚠️ Complex setup |
| Built-in memory | ✅ | ❌ Manual | ⚠️ Manual |
| Tool use | ✅ Automatic | ❌ Manual | ⚠️ Complex |
| Agent storage | ✅ SQLite/PG | ❌ None | ❌ None |
| Lines of code | ~10 | ~100+ | ~50+ |

---

### Why LanceDB?

LanceDB is an **embedded vector database** — it runs locally as a file, with no server required. Benefits:

- ✅ **Zero setup** — just point to a directory path
- ✅ **Embedded** — no Docker, no server, no cloud account
- ✅ **Fast** — written in Rust, powered by Apache Arrow
- ✅ **Supports both keyword and vector (embedding) search**
- ✅ **Free and open-source**

In this project, `SearchType.keyword` is used — LanceDB performs BM25-style full-text search (via `tantivy`) over the stored chunks.

---

### Why DuckDuckGo?

DuckDuckGo is a **privacy-focused search engine** with a free, unofficial Python API. It is used here because:
- ✅ **No API key required** — works immediately out of the box
- ✅ **No usage limits** (within reason)
- ✅ **Privacy-respecting** — no tracking
- ✅ Natively supported by Phidata via `phi.tools.duckduckgo`

---

### Why SQLite for Agent Storage?

SQLite is used to **persist conversation history** between agent runs. Phidata's `SqlAgentStorage` automatically:
- Creates a `data.db` file in the working directory
- Saves every message (user + agent) to a table
- Loads history on the next agent call via `add_history_to_messages=True`

This means the agent **remembers** what was discussed even if you restart the notebook.

---

## 🔍 Component Deep Dive

### 1. LanceDB Vector Store

```python
vector_db = LanceDb(
    table_name="documents",        # Table where chunks are stored
    uri="/tmp/lancedb",            # Local directory — no server needed
    search_type=SearchType.keyword # BM25 keyword search (no embeddings needed)
)
```

**SearchType options:**
| Type | How it works | When to use |
|---|---|---|
| `keyword` | BM25 full-text search (TF-IDF style) | Simple, fast, no embedding cost |
| `vector` | Cosine similarity on embeddings | Better semantic matching |
| `hybrid` | Combines keyword + vector | Best results, highest cost |

---

### 2. PDF Knowledge Base

```python
pdf_knowledge_base = PDFKnowledgeBase(
    path="/content/Data/",        # Can be a folder — loads ALL PDFs inside
    vector_db=vector_db,          # Where to store processed chunks
    reader=PDFReader(chunk=True)  # chunk=True splits PDFs into pieces
)

pdf_knowledge_base.load()         # Processes PDFs and populates LanceDB
```

**Chunking explained:**
- `chunk=True` splits each PDF page into smaller text segments
- Smaller chunks = more precise retrieval (fewer irrelevant sentences)
- Each chunk is stored as a separate searchable record in LanceDB

---

### 3. Phidata Agent

```python
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"), # The brain — LLM for reasoning
    knowledge=pdf_knowledge_base,        # PDF knowledge — searched automatically
    tools=[DuckDuckGo()],               # Web search — used when needed
    show_tool_calls=True,               # Shows which tools were called
    markdown=True,                       # Response formatted as Markdown
    storage=SqlAgentStorage(            # Persistent memory
        table_name="data",
        db_file="data.db"
    ),
    add_history_to_messages=True,       # Injects chat history into each prompt
)
```

**What `add_history_to_messages=True` does:**
Every time the agent is called, it reads the full conversation from `data.db` and includes it in the system prompt — so the agent always has full context of everything discussed.

---

### 4. Streaming Response

```python
agent.print_response(
    "Does GPT-4 accept visual inputs?",
    stream=True   # Tokens printed as they are generated (like ChatGPT)
)
```

`stream=True` uses Server-Sent Events (SSE) to print tokens progressively rather than waiting for the full response — much better user experience.

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10 or higher
- An [OpenAI API key](https://platform.openai.com/api-keys)
- pip

### Step 1: Clone the repository
```bash
git clone https://github.com/your-username/agentic-rag-phidata.git
cd agentic-rag-phidata
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Set up your API key

**Option A — `.env` file (recommended for local use):**
```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```
Then load it in the notebook:
```python
from dotenv import load_dotenv
load_dotenv()
```

**Option B — Environment variable:**
```bash
export OPENAI_API_KEY=sk-your-key-here
```

### Step 4: Run the notebook
```bash
jupyter notebook Agentic_RAG_phi_data.ipynb
```

---

## ☁️ Running on Google Colab

### Step 1: Upload the notebook
Go to [colab.research.google.com](https://colab.research.google.com) → File → Upload Notebook

### Step 2: Add your OpenAI API key to Colab Secrets
1. Click the 🔑 **Secrets** icon in the left sidebar
2. Click **+ Add new secret**
3. Name: `OpenAI`, Value: your OpenAI API key
4. Toggle **Notebook access** ON

### Step 3: Install dependencies (Cell 1)
```python
!pip install phidata openai duckduckgo-search lancedb tantivy pypdf
!pip install pylance
```

### Step 4: Run all cells
Runtime → Run all (`Ctrl+F9`)

> ⏱️ **First run:** The `pdf_knowledge_base.load()` step downloads the GPT-4 PDF and indexes it — takes ~30–60 seconds.
> **Subsequent runs:** LanceDB is already populated — `load()` is near-instant.

---

## ⚙️ Configuration & Customization

### Use your own PDF
```python
# Replace the wget command with your own file
# Then update the path:
pdf_knowledge_base = PDFKnowledgeBase(
    path="/content/Data/",    # Put your PDF(s) in this folder
    vector_db=vector_db,
    reader=PDFReader(chunk=True),
)
```

### Use a folder of multiple PDFs
```python
# Just put multiple PDFs in the Data/ directory
# PDFKnowledgeBase automatically loads ALL .pdf files in the folder
! cp my_paper1.pdf my_paper2.pdf Data/
```

### Switch to semantic (vector/embedding) search
```python
from phi.vectordb.search import SearchType
from phi.embedder.openai import OpenAIEmbedder

vector_db = LanceDb(
    table_name="documents",
    uri="/tmp/lancedb",
    search_type=SearchType.vector,          # semantic search
    embedder=OpenAIEmbedder(id="text-embedding-3-small")  # embedding model
)
```

### Switch to a different LLM
```python
# Use GPT-4o for higher quality
from phi.model.openai import OpenAIChat
model = OpenAIChat(id="gpt-4o")

# OR use Anthropic Claude
from phi.model.anthropic import Claude
model = Claude(id="claude-3-haiku-20240307")

# OR use a local Ollama model (free, no API key)
from phi.model.ollama import Ollama
model = Ollama(id="llama3")
```

### Add more tools
```python
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.arxiv import ArxivTools
from phi.tools.wikipedia import WikipediaTools

agent = Agent(
    ...
    tools=[DuckDuckGo(), ArxivTools(), WikipediaTools()],
)
```

---

## 📄 Sample Outputs

### Query 1: PDF-based question
```
User: Does GPT-4 accept visual inputs?

Agent: [Searching knowledge base for: visual inputs]
[Tool call: knowledge_base.search("visual inputs")]

Yes! **GPT-4 is a multimodal model** that can accept both text and
image inputs 🖼️. According to the GPT-4 technical report:

- GPT-4 accepts inputs consisting of **interleaved text and images**
- It can process images alongside text prompts
- The model produces text outputs based on both modalities

This makes it distinct from GPT-3.5, which only handles text inputs.
```

---

### Query 2: Comparison question (uses PDF + memory)
```
User: What is the comparison of GPT-4 and GPT-3.5?

Agent: [Checking conversation history...]
[Tool call: knowledge_base.search("GPT-4 GPT-3.5 comparison")]

Based on the GPT-4 technical report, here's a detailed comparison:

| Feature         | GPT-3.5        | GPT-4            |
|-----------------|----------------|------------------|
| Input modality  | Text only      | Text + Images    |
| MMLU score      | 70.0%          | 86.4%            |
| Bar exam        | ~10th pctile   | ~90th pctile     |
| Context window  | 16K tokens     | 128K tokens      |
| Reliability     | Lower          | Higher (RLHF++)  |

GPT-4 significantly outperforms GPT-3.5 across nearly all benchmarks...
```

---

## ⚠️ Limitations

### 1. Keyword Search Misses Semantic Meaning
**What happens:** `SearchType.keyword` performs BM25-style text matching. It finds chunks that contain the exact search terms but misses semantically related content.

**Example:** Searching "visual input" will NOT find chunks that say "multimodal capabilities" — even though they mean the same thing. This leads to missed relevant content.

---

### 2. In-Memory / Local LanceDB is Not Production-Ready
**What happens:** LanceDB stores data at `/tmp/lancedb`, which is a **temporary directory** on Colab. When the Colab session ends, this folder is deleted and all indexed PDF data is lost.

**Why it matters:** Every time you start a new Colab session, you must run `pdf_knowledge_base.load()` again, re-indexing the entire PDF from scratch.

---

### 3. SQLite Storage is Session-Dependent on Colab
**What happens:** `data.db` is created in the Colab working directory (`/content/`). Like all Colab files, it is deleted when the session disconnects, erasing all conversation memory.

---

### 4. No Custom Chunk Size Control
**What happens:** `PDFReader(chunk=True)` uses default chunk sizes that may not be optimal. Chunks that are too large return too much irrelevant text; chunks too small lose surrounding context.

---

### 5. DuckDuckGo Rate Limiting
**What happens:** DuckDuckGo's unofficial API can block requests if used too frequently in quick succession. There is no built-in retry or backoff logic.

---

### 6. Single PDF Indexed in This Demo
**What happens:** The notebook downloads and indexes only the GPT-4 paper. Questions about any other topic will either rely on DuckDuckGo or the LLM's internal knowledge.

---

### 7. No Re-ranking of Retrieved Chunks
**What happens:** LanceDB returns the top-N chunks based on keyword match score alone. It does not re-rank them by relevance to the full question context, so the most relevant chunk is not always first.

---

### 8. `show_tool_calls=True` Exposes Internals
**What happens:** Tool call details are printed to the console. For a production-facing application, you typically would not want to expose the agent's internal reasoning steps to end users.

---

### 9. No Evaluation or Quality Metrics
**What happens:** There is no way to measure whether the RAG pipeline is retrieving the right chunks or generating accurate answers. For production use, you need a RAG evaluation framework.

---

## 🔧 How Limitations Can Be Resolved

### Fix 1: Upgrade to Semantic (Vector) Search for Better Relevance
```python
from phi.vectordb.search import SearchType
from phi.embedder.openai import OpenAIEmbedder

# Switch from keyword → vector search
vector_db = LanceDb(
    table_name="documents",
    uri="/tmp/lancedb",
    search_type=SearchType.vector,
    embedder=OpenAIEmbedder(id="text-embedding-3-small")
    # Now "visual inputs" WILL match "multimodal capabilities"
)

# OR use hybrid search (best of both worlds)
vector_db = LanceDb(
    table_name="documents",
    uri="/tmp/lancedb",
    search_type=SearchType.hybrid,  # keyword + vector combined
    embedder=OpenAIEmbedder(id="text-embedding-3-small")
)
```

---

### Fix 2: Persist LanceDB to Google Drive on Colab
```python
from google.colab import drive
drive.mount('/content/drive')

# Store LanceDB in Drive — survives session restarts
vector_db = LanceDb(
    table_name="documents",
    uri="/content/drive/MyDrive/lancedb",  # persistent location
    search_type=SearchType.keyword,
)

# Now pdf_knowledge_base.load() only needs to run ONCE ever
```

---

### Fix 3: Persist SQLite Storage to Google Drive
```python
agent = Agent(
    ...
    storage=SqlAgentStorage(
        table_name="data",
        db_file="/content/drive/MyDrive/agent_data.db"  # Drive-backed
    ),
    add_history_to_messages=True,
)
```

---

### Fix 4: Control Chunk Size for Better Retrieval
```python
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.document.reader.pdf import PDFReader

pdf_knowledge_base = PDFKnowledgeBase(
    path="/content/Data/",
    vector_db=vector_db,
    reader=PDFReader(
        chunk=True,
        chunk_size=1000,     # characters per chunk (default is often too large)
        overlap=200,         # overlap between chunks to preserve context
    ),
)
```

---

### Fix 5: Use Tavily Instead of DuckDuckGo for Reliable Web Search
```python
# Tavily is more reliable and returns cleaner results
from phi.tools.tavily import TavilyTools
import os

os.environ["TAVILY_API_KEY"] = userdata.get("tavily")
# Free tier: 1,000 searches/month at https://tavily.com

agent = Agent(
    ...
    tools=[TavilyTools()],  # replace DuckDuckGo
)
```

---

### Fix 6: Load Multiple PDFs for a Richer Knowledge Base
```python
# Just add more PDFs to the Data/ directory before loading
! wget "https://arxiv.org/pdf/2303.08774.pdf" -O Data/gpt-4.pdf
! wget "https://arxiv.org/pdf/2005.14165.pdf" -O Data/gpt-3.pdf
! wget "https://arxiv.org/pdf/1706.03762.pdf" -O Data/attention.pdf

# PDFKnowledgeBase automatically indexes ALL PDFs in the folder
pdf_knowledge_base = PDFKnowledgeBase(
    path="/content/Data/",   # all 3 PDFs will be indexed
    vector_db=vector_db,
    reader=PDFReader(chunk=True),
)
pdf_knowledge_base.load()
```

---

### Fix 7: Add RAG Evaluation with RAGAS
```python
# pip install ragas
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

# Evaluate your RAG pipeline quality
results = evaluate(
    dataset=your_eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_recall]
)
print(results)
```

---

### Fix 8: Switch to PostgreSQL for Production Storage
```python
# For a production deployment with multiple users
from phi.storage.agent.postgres import PgAgentStorage

agent = Agent(
    ...
    storage=PgAgentStorage(
        table_name="agent_sessions",
        db_url="postgresql://user:password@host:5432/dbname"
    ),
    add_history_to_messages=True,
)
```

---

## 📚 Key Concepts for Beginners

### What is RAG?
**RAG (Retrieval-Augmented Generation)** is a technique that gives LLMs access to external knowledge by:
1. **Retrieving** relevant text from a knowledge base (your documents)
2. **Augmenting** the LLM's prompt with that text
3. **Generating** an answer using both the retrieved text and the LLM's training

Without RAG, the LLM can only answer from its training data. With RAG, it can answer from *your* documents.

### What is a Vector Database?
A vector database stores text as **mathematical vectors** (arrays of numbers). These vectors capture the *meaning* of text — not just the words. This allows searching by meaning rather than exact keyword match.

```
"visual input"       → [0.23, -0.11, 0.87, ...]  (embedding vector)
"multimodal input"   → [0.24, -0.10, 0.85, ...]  (very similar vector)
"banana"             → [-0.45, 0.62, -0.23, ...] (very different vector)
```

When you search for "visual input", the database returns "multimodal input" as relevant — because their vectors are mathematically close.

### What is Chunking?
PDFs can be hundreds of pages long. Sending an entire PDF to an LLM at once would exceed its context window. **Chunking** splits the document into smaller pieces:

```
Full PDF (100 pages)
    │
    ▼
Chunk 1: "Introduction — GPT-4 is a large..."   (300 words)
Chunk 2: "Methods — We trained GPT-4 using..."   (300 words)
Chunk 3: "Results — GPT-4 scored 86.4% on..."    (300 words)
...
Chunk N: "Conclusion — In this work, we..."       (300 words)
```

Only the **relevant chunks** are sent to the LLM — not the whole document.

### What is BM25 / Keyword Search?
BM25 is a classic information retrieval algorithm used by search engines. It scores documents based on:
- How many times query terms appear in the document (**term frequency**)
- How rare the query terms are across all documents (**inverse document frequency**)

It is fast and requires no embeddings, but it cannot understand synonyms or meaning.

### What is an Embedding?
An embedding is a numerical representation of text that captures semantic meaning. The text is converted into a vector (list of numbers) by an embedding model. Two texts with similar meanings have vectors that are mathematically close to each other.

### What is Agentic vs Standard RAG?
- **Standard RAG**: Retrieve → Stuff into prompt → Generate. Always retrieves, always from one source.
- **Agentic RAG**: The LLM *decides* whether to retrieve, from which source, how many times, and how to combine results. It treats retrieval as a tool it can choose to use.

---

## 🎯 Use Cases

This project pattern (PDF + Vector DB + Web Search + Memory) can be applied to:

| Use Case | PDF Knowledge Base | Web Search Use |
|---|---|---|
| 📄 Legal document QA | Contracts, case law | Look up recent rulings |
| 📚 Research assistant | Academic papers | Find related recent work |
| 🏥 Medical reference | Clinical guidelines | Check latest drug info |
| 🏢 HR policy chatbot | Employee handbook | Find updated regulations |
| 📊 Financial analysis | Annual reports | Get current market data |
| 🎓 Study assistant | Textbooks, notes | Supplement with examples |
| 🔧 Technical support | Product manuals | Search community forums |

---

## 🤝 Contributing

Ideas for extending this project:

- 🔍 Switch to **hybrid search** (keyword + vector) for better retrieval
- 💾 Add **Google Drive persistence** for LanceDB and SQLite on Colab
- 🖥️ Build a **Gradio or Streamlit UI** for a proper chat interface
- 📊 Add **RAGAS evaluation** to measure retrieval and answer quality
- 📄 Support more document types: **DOCX, HTML, CSV, YouTube transcripts**
- 🔄 Add a **re-ranking step** after retrieval to sort chunks by relevance
- 🌐 Replace DuckDuckGo with **Tavily** for more reliable web search
- 🤖 Try **different LLMs**: Claude, Llama3 via Ollama, Mixtral via Groq

To contribute:
```bash
git checkout -b feature/add-gradio-ui
git commit -m "Add Gradio chat interface for document QA"
git push origin feature/add-gradio-ui
# Then open a Pull Request on GitHub
```

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [Phidata](https://docs.phidata.com/) — for the clean, powerful agent framework
- [LanceDB](https://lancedb.github.io/lancedb/) — for the embedded vector database
- [OpenAI](https://openai.com/) — for the gpt-4o-mini model
- [DuckDuckGo](https://duckduckgo.com/) — for the free web search API
- [OpenAI](https://cdn.openai.com/papers/gpt-4.pdf) — for the GPT-4 technical report used as demo data

---

*Built with ❤️ using Phidata, LanceDB, and OpenAI*
