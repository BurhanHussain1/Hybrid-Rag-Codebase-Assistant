# Hybrid RAG — Codebase Intelligence Assistant · Implementation Plan

> **This file is a self-contained handoff.** A fresh session can implement the whole
> project from this file alone. Build it in **this folder**
> (`C:\Users\burhan.hussain\Desktop\Rags\Hybrid Rag`), phase by phase, and confirm each
> phase's "Done when" before moving on.

---

## 0. What we're building

An AI assistant that answers questions about any **Git repository** by combining two
retrievers and fusing their results:

- **BM25 keyword search** — nails exact identifiers (`UserSessionManager`, `JWT_SECRET`).
- **Semantic vector search** — nails meaning ("where does login happen?").
- **Reciprocal Rank Fusion (RRF)** — merges both rankings into one.
- **Cross-encoder reranking** — reorders the fused shortlist for precision.
- **LLM answer** — grounded in the retrieved code, with `file:line` citations.

The headline feature is a **benchmark page** comparing **BM25-only vs Vector-only vs
Hybrid** on the same queries, proving hybrid retrieval wins.

Example queries it should handle well:
- Natural language: *"How does authentication work?"*, *"Explain the payment flow."*
- Exact identifiers: `UserSessionManager`, `JWT_SECRET`, `AUTH_SERVICE_TIMEOUT`

---

## 1. Confirmed decisions (do NOT re-litigate these)

- **Scope:** lean, finishable portfolio build — NOT the full SaaS in the original plan doc.
- **UI:** Streamlit (single Python app; no Next.js/React).
- **Embeddings:** free & local — ChromaDB's built-in `all-MiniLM-L6-v2` (ONNX, no torch needed).
- **Generation:** **OpenAI `gpt-4o-mini`** (user chose OpenAI — do NOT use Claude/Gemini/etc.).
- **Reranker:** INCLUDED — cross-encoder via `sentence-transformers` (this is the one part
  that needs `torch`).
- **One paid key only:** `OPENAI_API_KEY`. Everything else runs locally and free.

### Deliberately NOT building (cut from the original plan — don't add these)
Next.js/React/Tailwind/shadcn/Monaco · FastAPI backend · background workers · Postgres ·
Redis · Qdrant/FAISS · auth/users · analytics tables · multi-LLM · Tree-sitter (optional
later) · GraphRAG · agentic RAG · AI code review · GitLab/ZIP/private/multi-repo (start with
one repo).

---

## 2. Tech stack

| Layer          | Tool                                                        |
| -------------- | ----------------------------------------------------------- |
| Language / UI  | Python + Streamlit                                          |
| Repo input     | GitPython (clone a GitHub URL) or a local folder path       |
| Code chunking  | `langchain-text-splitters` (`RecursiveCharacterTextSplitter.from_language`) |
| Keyword search | `rank_bm25` (BM25Okapi — pure Python, free)                 |
| Vector search  | ChromaDB + `all-MiniLM-L6-v2` (local, free)                 |
| Fusion         | Reciprocal Rank Fusion (small custom function)              |
| Reranking      | `sentence-transformers` CrossEncoder `cross-encoder/ms-marco-MiniLM-L-6-v2` (needs torch) |
| Generation     | OpenAI `gpt-4o-mini`                                         |
| Config/env     | `python-dotenv`                                             |

### `requirements.txt`
```
openai
chromadb
rank-bm25
sentence-transformers
langchain-text-splitters
gitpython
streamlit
python-dotenv
```

---

## 3. Repo / GitHub setup

- **Repo name:** `hybrid-rag-codebase-assistant`  (alt: `codefusion-hybrid-rag`)
- **Description:** *A Hybrid RAG codebase-intelligence assistant that answers questions about
  any Git repo — combining BM25 keyword search and semantic vector search with Reciprocal
  Rank Fusion and cross-encoder reranking — with file-and-line citations.*
- **Visibility:** public. Don't add README/.gitignore/license on GitHub (created locally).
- **Topics:** `hybrid-rag` `rag` `bm25` `vector-search` `reciprocal-rank-fusion`
  `cross-encoder` `code-search` `openai` `chromadb` `streamlit` `python`
- `.gitignore` must ignore: `.env`, `venv/`, `__pycache__/`, `chroma_db/`, `bm25_index/`,
  `repos/`, `*.sqlite3`, `.gstack/`, editor/OS junk.
- `.env` holds `OPENAI_API_KEY` (gitignored). Commit `.env.example` only.
- **Commits:** human-friendly imperative messages, one per phase. The **user runs commits/pushes
  themselves** — provide the commands, don't commit for them.
  - Phase commit messages: `Set up project structure and dependencies` · `Add repo ingestion
    and chunking` · `Build vector and BM25 indexes` · `Add hybrid retrieval with RRF and
    reranking` · `Add cited answers and Streamlit chat UI` · `Add BM25-vs-vector-vs-hybrid
    benchmark` · `Add repo analysis and evaluation`

### GitHub workflow — push after each phase

**One-time, after Phase 0** (make the first commit, then connect the remote — pick A or B):
```bash
git add .
git commit -m "Set up project structure and dependencies"

# A) GitHub CLI (easiest, if `gh` is installed):
gh repo create hybrid-rag-codebase-assistant --public --source=. --remote=origin --push

# B) Manual: create an EMPTY repo on github.com (no README/.gitignore/license), then:
git remote add origin https://github.com/<your-username>/hybrid-rag-codebase-assistant.git
git push -u origin main
```

**After every later phase** (repeat this each time a phase's "Done when" passes):
```bash
git status     # SAFETY CHECK: .env must NOT appear here (only .env.example is committed)
git add .
git commit -m "<the phase's message from the list above>"
git push
```

Notes:
- `.env` is gitignored — it must never show up in `git status`. Your API key stays local.
- Generated dirs (`chroma_db/`, `bm25_index/`, `repos/`, `venv/`) are gitignored; only code +
  docs get pushed, so the repo stays lean.
- **All `git` commits and pushes are done manually by the user.** The implementing session
  must NOT run `git commit` or `git push` — it only surfaces the exact command for reference
  after each phase, and the user runs it. (Scaffolding files and `git init` are fine for the
  session to do; committing and pushing are not.)
- For commits to show on the GitHub contribution graph, the git email must be verified on the
  user's GitHub account.

---

## 4. Project structure (flat, like the Basic RAG project)

```
Hybrid Rag/
├── config.py         # repo source, model names, retrieval params (single source of truth)
├── ingest.py         # clone/read repo → filter files → code-aware chunk → build BOTH indexes
├── retrieve.py       # vector + BM25 → RRF → cross-encoder rerank → top chunks  ← the core
├── rag.py            # retrieve → OpenAI → cited answer  (+ retrieval-only fallback)
├── app.py            # Streamlit chat UI (retrieved files + hybrid scores + citations)
├── benchmark.py      # BM25-only vs Vector-only vs Hybrid comparison + results table
├── requirements.txt
├── .env.example
├── README.md
├── chroma_db/        # vector store (gitignored, regenerated by ingest.py)
├── bm25_index/       # pickled BM25 index + chunk store (gitignored)
└── repos/            # cloned repositories (gitignored)
```

---

## 5. Key parameters (defaults — put these in `config.py`)

- Vector embedding: ChromaDB `DefaultEmbeddingFunction` = `all-MiniLM-L6-v2` (384-dim, ONNX).
- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- OpenAI model: `gpt-4o-mini`, `temperature=0`.
- Chunking: `chunk_size=800`, `chunk_overlap=150`, language chosen per file extension.
- Retrieval: vector top-k = 20, BM25 top-k = 20 → RRF (`k=60`) → take top 20 → **rerank →
  top 6** passed to the LLM.
- Supported file types: `.py .js .ts .tsx .jsx .java .go .cs .rb .rs .md .rst .yaml .yml
  .json .toml .tf` + `Dockerfile`, `README*`. Skip: `.git/`, `node_modules/`, `venv/`,
  `dist/`, `build/`, `.next/`, `__pycache__/`, lockfiles, minified files, anything > ~1 MB,
  and binary files.
- Each chunk stores metadata: `repo`, `path` (relative), `start_line`, `language`, `chunk_id`.
  Compute `start_line` by locating the chunk's offset in the file text and counting newlines
  before it — citations are `path:start_line`.

### BM25 tokenization (important for code)
Tokenize by lowercasing and splitting on non-alphanumerics, **and** split
`camelCase`/`snake_case`/`PascalCase` into subtokens (e.g. `UserSessionManager` →
`user session manager` + keep the whole token). This lets BM25 match both the exact
identifier and its word parts.

### RRF (reference)
```python
def rrf(rankings, k=60):
    # rankings: list of lists of chunk_ids, each already ordered best->worst
    scores = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
```

---

## 6. Phases

### Phase 0 — Setup (½ day)
- Create folder scaffolding: `.gitignore`, `.env.example`, `requirements.txt`, `README.md`,
  `config.py`. `git init -b main`.
- Create venv (`python -m venv venv`), install deps. **`torch`/`sentence-transformers` is a
  large, slow install on this machine's flaky network — run it in the background with a
  generous timeout and verify with an import.** Consider installing crawl/light deps first,
  then torch-heavy ones.
- **Done when:** every dependency imports; `OPENAI_API_KEY` loads from `.env`.

### Phase 1 — Repo ingest & chunking (1–2 days)
- `ingest.py`: given a GitHub URL, clone into `repos/<name>` with GitPython (or accept a local
  path). Walk files, apply the include/exclude filters, read text, code-aware chunk with
  `RecursiveCharacterTextSplitter.from_language`, attach metadata (`path`, `start_line`,
  `language`, `chunk_id`).
- Pick ONE demo repo and get it fully working before generalizing. A medium Python/JS repo
  with auth/config/services makes the example queries shine (e.g. a well-known FastAPI-based
  project). Ask the user or pick a sensible default.
- **Done when:** running ingest produces a clean in-memory list of chunks with correct
  `path:start_line` metadata; print counts per language and a sample chunk.

### Phase 2 — Build both indexes (1 day)
- Vector: ChromaDB `PersistentClient(path="chroma_db")`, one collection, `upsert` chunks
  (Chroma embeds via the default MiniLM). Make it **resumable/idempotent** (upsert by
  `chunk_id`).
- BM25: build `BM25Okapi` over the tokenized chunks; pickle the index + the aligned chunk list
  (text + metadata) into `bm25_index/`.
- **Done when:** a test query returns sensible results from **each** index independently.

### Phase 3 — Hybrid retrieval — THE CORE (2–3 days)
- `retrieve.py`: `search(query, top_k=6, repo=None)`:
  1. Vector search (top 20) and BM25 search (top 20) — run both.
  2. **RRF merge** the two rankings.
  3. **Cross-encoder rerank** the top ~20 fused chunks → return top `top_k`.
  4. Return chunks with their component scores (bm25 / vector / fused / rerank) for the UI.
- **Done when:** an exact-identifier query (e.g. a class name) AND a natural-language query
  ("how does X work?") both return the right file in the top results — and hybrid beats
  either retriever alone on at least one query where the other fails.

### Phase 4 — Cited answers + Streamlit UI (2 days)
- `rag.py`: `answer(query, repo=None) -> (text, sources)`. Build a context block from the top
  chunks (each labelled `[n] path:start_line`), prompt OpenAI to answer **only** from context
  with inline `[n]` citations, `temperature=0`. Fall back to retrieval-only if no API key.
- `app.py`: Streamlit chat — question box, streamed/rendered answer with citations, an
  expandable "Retrieved files" panel showing each chunk's path:line and its hybrid scores,
  chat history, clear-chat. Code renders in fenced blocks (native copy button).
- **Done when:** asking a question in the browser returns a correct, cited answer and shows
  the retrieved files with scores.

### Phase 5 — Benchmark page — THE DIFFERENTIATOR (1–2 days)
- `benchmark.py` + a Streamlit page/section: a small labelled query set (each query tagged
  with the file/identifier that *should* be retrieved). Run each query three ways —
  **BM25-only**, **Vector-only**, **Hybrid (+rerank)** — and score whether the right file
  appears in top-k. Show a comparison table (hit-rate / MRR per method) and per-query results.
- **Done when:** the table clearly shows Hybrid ≥ each single method, with at least one query
  where BM25 wins (exact identifier) and one where vectors win (natural language) — the whole
  point of hybrid.

### Phase 6 — Optional polish
- Repo auto-analysis (detect language/framework/DB from files) shown in a sidebar.
- Multiple repos (add a `repo` filter to retrieval — metadata already supports it).
- Eval harness (like Basic RAG's `eval.py`) scoring retrieval + citation + groundedness.

---

## 7. Environment gotchas (LEARNED THE HARD WAY — read before running anything)

This is a **Windows** machine, Python **3.13**, with a **flaky/slow network**.

- Use a venv at `venv/`; call the interpreter as `venv/Scripts/python.exe ...` (Git Bash) or
  activate it. Check `python --version` first.
- **Large model / package downloads stall** on this connection. torch + sentence-transformers
  and the cross-encoder model (~80 MB) may need **resumable downloads** and **background
  installs with long timeouts**. If a download aborts mid-stream, retry with a resumable
  (HTTP Range) streaming download rather than assuming the network is blocked.
- **ChromaDB (1.5.x) locking:** chroma leaves lingering Python processes that lock `chroma_db/`
  and the model cache. **NEVER kill a chroma process mid-write** — it corrupts the HNSW index
  (`Error loading hnsw index`). If the store corrupts, **delete `chroma_db/` and re-run**
  ingest (it's resumable).
- Set `os.environ["ANONYMIZED_TELEMETRY"] = "False"` before importing/using chromadb to
  silence telemetry noise.
- On Windows, reconfigure stdout to UTF-8 at the top of CLI scripts
  (`sys.stdout.reconfigure(encoding="utf-8")`) so emoji/box-drawing output doesn't crash.
- Make `ingest.py` resumable and wrap per-file work in try/except so one bad file doesn't kill
  the run.
- The MiniLM embedding model caches under `~/.cache/chroma/onnx_models/`; the first ingest
  downloads it once.

---

## 8. Reference implementation

A **completed, working** sibling project lives at
`C:\Users\burhan.hussain\Desktop\Rags\Basic rag` (a multi-framework docs RAG:
`ingest.py`, `rag.py`, `app.py`, `eval.py`, Streamlit UI, ChromaDB, local MiniLM embeddings,
OpenAI `gpt-4o-mini`). Reuse its patterns for: resumable ChromaDB ingest, the `rag.answer()
-> (text, sources)` shape, the Streamlit chat UI with citations and a stats sidebar, the
retrieval-only fallback when no API key, and the eval harness. **Hybrid RAG = Basic RAG + a
BM25 retriever + RRF fusion + cross-encoder reranking + a benchmark page.**

---

## 9. Two rules

1. **One repo end-to-end first.** Get the full pipeline working on a single demo repo before
   supporting more.
2. **Retrieval quality decides everything.** The Phase 5 benchmark is how we *prove* hybrid
   beats either search alone — it's the most important deliverable for the portfolio.

---

## How to start (for the implementing session)

1. Read this whole file.
2. Do **Phase 0** (scaffold + git init + venv + deps; background the torch install).
3. Ask the user for a demo GitHub repo URL (or pick a sensible medium-sized one) and do
   **Phase 1** end-to-end.
4. Proceed phase by phase, confirming each "Done when". Give the user the commit command after
   each phase (they commit/push themselves).
