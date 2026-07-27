# Hybrid RAG — Codebase Intelligence Assistant

Ask questions about any Git repository and get answers grounded in the actual source,
with `file:line` citations.

Most code-search RAG demos use one retriever and quietly fail half the time. Semantic
search understands *"where does login happen?"* but fumbles an exact symbol like
`JWT_SECRET`. Keyword search nails the symbol but has no idea what "login" means. This
project runs **both**, fuses the two rankings with **Reciprocal Rank Fusion**, reranks the
shortlist with a **cross-encoder**, and ships a benchmark page that proves the combination
beats either retriever alone.

```
                    ┌──────────────┐
  question ────────▶│  BM25 (top20)│──┐
                    └──────────────┘  │   ┌─────┐   ┌──────────────┐   ┌────────┐
                                      ├──▶│ RRF │──▶│ cross-encoder│──▶│ OpenAI │──▶ cited answer
                    ┌──────────────┐  │   └─────┘   │   rerank     │   └────────┘
             ──────▶│ Vector(top20)│──┘             └──────────────┘
                    └──────────────┘              top 20 fused → top 6
```

## Why hybrid

| Query | BM25 alone | Vector alone | Hybrid |
| --- | --- | --- | --- |
| `InvalidPasswordException` | ✅ exact token match | ❌ no semantic signal in a symbol | ✅ |
| *"How does confirming an email address work?"* | ❌ no literal overlap | ✅ understands intent | ✅ |

## Measured results

17 labelled queries against [`fastapi-users`](https://github.com/fastapi-users/fastapi-users)
(854 chunks), scored on whether the file that genuinely answers the query lands in the top 6.
Run it yourself with `python benchmark.py`.

**Strict** — must return the implementing *source* file; prose excluded:

| Method | hit-rate@6 | MRR | identifiers | natural language |
| --- | --- | --- | --- | --- |
| BM25 only | 65% | 0.363 | 88% | 44% |
| Vector only | 76% | 0.536 | 88% | 67% |
| RRF (no rerank) | 71% | 0.391 | 88% | 56% |
| **Hybrid (full)** | **82%** | **0.603** | **100%** | 67% |

**Lenient** — the source file *or* its documentation page counts:

| Method | hit-rate@6 | MRR | identifiers | natural language |
| --- | --- | --- | --- | --- |
| BM25 only | 41% | 0.211 | 38% | 44% |
| Vector only | 59% | 0.424 | 62% | 56% |
| RRF (no rerank) | 59% | 0.377 | 38% | 78% |
| **Hybrid (full)** | **82%** | **0.592** | **88%** | 78% |

Hybrid wins in both modes, and the failure modes are exactly the predicted ones:
BM25 alone recovers `InvalidPasswordException` where the vector index misses it entirely,
vector alone recovers several natural-language queries BM25 can't touch, and fusion recovers
three queries that *both* single retrievers miss. The cross-encoder is worth ~0.21 MRR on top
of plain RRF — it mostly reorders what fusion already found rather than finding more.

### What the benchmark taught us

The first run scored 22% on natural-language queries for *every* method, which looked like a
retrieval failure. It wasn't. `fastapi-users` ships a documentation site mirroring its
modules, and only **24% of the index is actual source code** (37% tests, 35% docs). The
retrievers were correctly returning `docs/configuration/routers/register.md` for *"how does a
new user sign up?"* — a genuinely correct answer that strict labels scored as a miss.

Two things came out of that: chunks are now tagged `source` / `docs` / `tests` / `examples` so
you can ask for implementations specifically, and the benchmark reports both scoring modes
instead of only the flattering one. Restricting to source lifts BM25 from 18% → 65%.

There is also a real limitation worth naming: **BM25 finds usages, not definitions.** The
chunk defining `class JWTStrategy` mentions the name once; its tests and docs mention it three
times, so term frequency ranks them higher. Fusion plus reranking is what pulls the definition
back to #1.

## Stack

| Layer | Tool |
| --- | --- |
| UI | Streamlit |
| Repo input | GitPython (clone a URL) or a local folder path |
| Chunking | `langchain-text-splitters`, code-aware per language |
| Keyword search | `rank_bm25` (BM25Okapi) |
| Vector search | ChromaDB + `all-MiniLM-L6-v2` (local, free) |
| Fusion | Reciprocal Rank Fusion |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | OpenAI `gpt-4o-mini` |

Only one paid dependency: `OPENAI_API_KEY`. Embeddings and reranking run locally on CPU.
Without a key the app still works in retrieval-only mode.

## Quickstart

Requires Python 3.11+ (built and tested on 3.13.5).

```bash
python -m venv venv
venv\Scripts\activate            # Windows;  source venv/bin/activate on macOS/Linux

# Linux only — see the note below. Skip this line on Windows and macOS.
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

copy .env.example .env           # then paste your OpenAI key into .env

python download_model.py         # fetch the reranker (resumable; ~90 MB, once)
python ingest.py https://github.com/fastapi-users/fastapi-users
streamlit run app.py
```

**Why the extra torch line on Linux.** `sentence-transformers` depends on torch, and PyPI's
default torch wheel for Linux is the CUDA build — about 2.5 GB, downloaded to run a 90 MB
reranker that never touches a GPU. Installing the CPU wheel first gets you ~200 MB instead.
Windows and macOS already default to CPU wheels, so the plain install is fine there.

Versions in `requirements.txt` are pinned exactly. ChromaDB and `langchain-text-splitters`
have both shipped breaking changes across minor releases, and an unpinned clone a year from
now is a clone that doesn't run.

`ingest.py` with no argument uses the default demo repo from `config.py`. Point it at a
local folder instead of a URL to index a project you already have on disk.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

43 tests, ~7 seconds, no network and no API key — they cover the pure logic that everything
else rests on: BM25 tokenization, chunk metadata and line numbers, the RRF arithmetic, and
the benchmark/eval scoring rules.

The line-number tests get the most attention, because `start_line` is the one field that can
be wrong without anything appearing broken — citations keep rendering, they just point at the
wrong code. One test specifically covers duplicated code blocks, the case that forced line
numbers to be derived from the splitter's offset rather than by searching for the chunk text.

The suite was checked by mutation: introducing an off-by-one in `start_line`, shifting the RRF
rank offset, and disabling camelCase splitting each turn it red.

## Command line

```bash
python retrieve.py "JWTStrategy"                     # raw retrieval + per-stage scores
python retrieve.py --method bm25 "JWTStrategy"       # one retriever at a time
python retrieve.py --kind source "JWTStrategy"       # skip docs/tests, code only
python rag.py "How does authentication work?"        # cited answer in the terminal
python benchmark.py                                  # both scoring modes
python benchmark.py --mode strict --k 3              # harder: source only, top 3
```

## Files

| File | Role |
| --- | --- |
| `config.py` | every tunable: models, chunk sizes, top-k values, file filters, BM25 tokenizer |
| `ingest.py` | clone/read a repo → filter → code-aware chunk → build both indexes |
| `retrieve.py` | vector + BM25 → RRF → cross-encoder rerank — **the core** |
| `rag.py` | retrieve → OpenAI → cited answer |
| `app.py` | Streamlit chat UI + benchmark page |
| `benchmark.py` | BM25 vs Vector vs Hybrid comparison, labelled query set |
| `analyze.py` | detects languages, frameworks, databases, tooling from a repo |
| `eval.py` | end-to-end eval: retrieval + citation + groundedness |
| `download_model.py` | resumable reranker download (the Hub client won't resume) |
| `tests/` | pytest suite — tokenization, chunk line numbers, RRF, scoring |

## End-to-end evaluation

`benchmark.py` scores retrieval. `eval.py` scores the whole pipeline, including what the
model does with what it retrieved:

```bash
python eval.py                    # retrieval + citation + groundedness
python eval.py --retrieval-only   # no API calls, no cost
python eval.py -v                 # show any identifier not found in context
```

Current results on 8 implementation questions (`gpt-4o-mini`, hybrid, k=6, source-only):

| Metric | Score | What it checks |
| --- | --- | --- |
| Retrieval | 88% | the file that answers the question reached the context |
| Citation | 75% | that file is actually cited — 8/8 answers cite something, 8/8 markers valid |
| Groundedness | 100% | every `backticked` identifier appears in the retrieved context |
| Content | 75% | the answer mentions the expected key terms |

Groundedness is the cheap hallucination detector: across 113 identifiers named in the eight
answers, every one was present in the retrieved code. It catches fabricated API names, not
misinterpretation of real ones — an answer can cite real symbols and still describe them
wrongly, so treat it as a floor, not a guarantee.

## Multiple repositories

```bash
python ingest.py https://github.com/fastapi-users/fastapi-users https://github.com/psf/requests
python analyze.py requests          # what is this codebase?
python retrieve.py --repo requests "how are redirects handled?"
```

Each repo gets its own BM25 corpus (scores are corpus-relative and can't be concatenated)
and its own metadata tag. The Streamlit sidebar grows a repository selector once more than
one is indexed.

`benchmark.py` always scopes to `fastapi-users`, the repo its labels were written against.
That's deliberate: adding a second repo unscoped moved strict hit-rate for BM25 from 65% →
59% purely because unrelated chunks compete for the same top-6 slots. A benchmark whose
result depends on what else happens to be in the index isn't measuring the retriever.

That interference is itself a useful result — under cross-repo noise, **hybrid held at 82%
and 100% on identifiers while BM25 lost 6 points and RRF-without-rerank lost 12**. The
reranker is what absorbs the extra noise.

## Notes on the design

**Line numbers are exact.** Chunk offsets come from the splitter's `add_start_index`, then
newlines before the offset are counted. Searching for the chunk text afterwards would break
on repeated code.

**BM25 tokenizes code, not prose.** Each identifier is kept whole *and* split on
camelCase/snake_case — `UserSessionManager` indexes as `usersessionmanager` + `user` +
`session` + `manager`, so both the exact symbol and "session manager" find it.

**RRF fuses ranks, not scores.** BM25 scores and cosine distances aren't on comparable
scales, and any normalization factor you pick changes per query. RRF only reads positions.

**Everything degrades instead of breaking.** No API key → retrieval-only mode. No reranker
→ RRF ordering. Interrupted ingest → re-run, it resumes.

## License

MIT
