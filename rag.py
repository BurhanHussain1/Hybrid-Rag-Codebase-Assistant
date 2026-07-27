"""
Phase 4 — Cited answer generation.

Wraps the hybrid retriever with an LLM call: retrieve the best chunks, hand them
to OpenAI as numbered context, and require the answer to cite them inline as [n].

The grounding rules matter more than the model here. Every claim has to trace
back to a chunk that was actually retrieved, and every source carries a real
`path:start_line` so a reader can open the file and check. If the context does
not contain the answer, the model is told to say so rather than fill the gap
from memory — a confident wrong answer about someone's codebase is worse than
"not in the retrieved code".

With no API key configured the pipeline still runs and returns the retrieved
sources, just without a generated answer.

Usage:
    python rag.py "How does authentication work?"
    python rag.py --method vector "Explain the password hashing flow"
"""

import argparse

import config
import retrieve

SYSTEM_PROMPT = """You are a codebase intelligence assistant. You answer questions about a \
specific Git repository using ONLY the code and documentation retrieved from it.

Rules:
- Answer strictly from the provided context. Never invent functions, classes, parameters,
  or file paths that do not appear in it.
- If the context does not contain the answer, say so plainly and name what you would need.
  Do not fall back on general knowledge about similar libraries.
- CITATIONS ARE MANDATORY. Every sentence that states a fact about the code must end with
  one or more source markers like [1] or [2][5], matching the numbered sources above each
  context block. A sentence asserting anything about this repository without a [n] marker
  is a failed answer. Example of the required style:
      The backend pairs a transport with a strategy [2], and `login()` delegates token
      creation to the strategy's `write_token()` [5].
- Cite the source you actually took the claim from. Do not cite everything at the end.
- Quote short code snippets from the context in fenced blocks when they make the answer
  concrete. Keep them short — point at the code, don't paste whole files.
- Explain how the pieces connect (what calls what), not just what each one is.
- Be direct and technical. No preamble, no summary of the question."""


def build_context(chunks):
    """
    Render retrieved chunks as a numbered context block.

    Each block is labelled with its citation target so the model can only cite
    things that are really there, and the [n] it emits maps to a source we can
    display back to the user.
    """
    blocks = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c['path']}:{c['start_line']}  ({c['language']})"
        blocks.append(f"{header}\n```{c['language']}\n{c['text']}\n```")
    return "\n\n---\n\n".join(blocks)


def to_sources(chunks):
    """Chunks -> the source records the UI renders, numbered to match the [n] citations."""
    return [
        {
            "n": i,
            "path": c["path"],
            "start_line": c["start_line"],
            "language": c["language"],
            "kind": c.get("kind", "source"),
            "repo": c["repo"],
            "text": c["text"],
            "bm25_score": c.get("bm25_score"),
            "vector_score": c.get("vector_score"),
            "fused_score": c.get("fused_score"),
            "rerank_score": c.get("rerank_score"),
        }
        for i, c in enumerate(chunks, 1)
    ]


def answer(question, repo=None, k=None, method="hybrid", kinds=None):
    """
    Retrieve, then generate a cited answer.

    Returns (answer_text, sources). Retrieval always happens; generation is
    skipped when there is no API key, so the app degrades to a code search
    engine instead of breaking.

    `kinds` narrows retrieval to e.g. ("source",) — useful on repos whose
    documentation out-competes the code it describes.
    """
    chunks = retrieve.search(question, top_k=k, repo=repo, method=method, kinds=kinds)
    if not chunks:
        return "Nothing relevant found in the index. Has `python ingest.py` been run?", []

    sources = to_sources(chunks)

    if not config.has_openai_key():
        return ("_No `OPENAI_API_KEY` in `.env` — showing retrieved code only. "
                "Add a key to get generated answers._", sources)

    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        temperature=config.OPENAI_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question}\n\nRetrieved code from the repository:\n\n"
                           f"{build_context(chunks)}",
            },
        ],
    )
    return response.choices[0].message.content, sources


def index_stats():
    """(total_chunks, {repo: n}, {language: n}, {kind: n}) — for the app sidebar."""
    try:
        store = retrieve.get_store()
    except Exception:
        return 0, {}, {}, {}
    repos, languages, kinds = {}, {}, {}
    for record in store["chunks"].values():
        repos[record["repo"]] = repos.get(record["repo"], 0) + 1
        languages[record["language"]] = languages.get(record["language"], 0) + 1
        kind = record.get("kind", "source")
        kinds[kind] = kinds.get(kind, 0) + 1
    return len(store["chunks"]), repos, languages, kinds


def main():
    config.use_utf8_stdout()

    parser = argparse.ArgumentParser(description="Ask a question about the indexed repo.")
    parser.add_argument("question", nargs="+")
    parser.add_argument("--method", default="hybrid", choices=["hybrid", "bm25", "vector", "rrf"])
    parser.add_argument("--k", type=int, default=config.FINAL_TOP_K)
    parser.add_argument("--repo", default=None)
    parser.add_argument("--kind", action="append", choices=list(config.CHUNK_KINDS),
                        help="restrict retrieval to a chunk kind (repeatable)")
    args = parser.parse_args()

    text, sources = answer(" ".join(args.question), repo=args.repo, k=args.k,
                           method=args.method, kinds=args.kind)

    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)
    if sources:
        print("\nSources:")
        for s in sources:
            score = s["rerank_score"]
            suffix = f"   (rerank {score:.2f})" if score is not None else ""
            print(f"  [{s['n']}] {s['path']}:{s['start_line']}{suffix}")


if __name__ == "__main__":
    main()
