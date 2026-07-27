"""
Phase 4/5 — Streamlit UI.

Two pages:
  Chat      — ask questions, get cited answers, inspect exactly which chunks were
              retrieved and how each retriever scored them.
  Benchmark — the differentiator: BM25 vs Vector vs Hybrid on a labelled query
              set, showing hit-rate and MRR side by side.

The "Retrieved files" panel is deliberately prominent. A RAG demo that only shows
the final answer asks you to take retrieval on faith; showing the per-stage scores
makes it possible to see *why* an answer came out the way it did — and to catch it
when retrieval, not the model, is what went wrong.

Run with:
    streamlit run app.py
"""

import config
import rag
import retrieve

import streamlit as st

st.set_page_config(page_title="Hybrid RAG — Codebase Assistant", page_icon="🔍", layout="wide")

EXAMPLES = [
    "How does authentication work?",
    "JWTStrategy",
    "How are passwords hashed and verified?",
    "Explain the OAuth login flow",
    "What does the UserManager do?",
    "How is a user created and registered?",
]

METHODS = {
    "Hybrid (BM25 + Vector + RRF + rerank)": "hybrid",
    "RRF only (no reranker)": "rrf",
    "BM25 only (keyword)": "bm25",
    "Vector only (semantic)": "vector",
}


@st.cache_data(show_spinner=False)
def cached_stats():
    return rag.index_stats()


def fmt(value, places=3):
    return f"{value:.{places}f}" if value is not None else "—"


def render_sources(sources):
    """The retrieved-files panel: citation targets, per-stage scores, and the code."""
    if not sources:
        return

    with st.expander(f"📎 Retrieved {len(sources)} chunks — scores and code", expanded=False):
        st.caption(
            "`bm25` keyword score · `vector` cosine similarity · `fused` RRF score · "
            "`rerank` cross-encoder relevance. A dash means that retriever never "
            "surfaced this chunk — which is exactly what fusion is for."
        )
        st.dataframe(
            [
                {
                    "#": s["n"],
                    "location": f"{s['path']}:{s['start_line']}",
                    "kind": s.get("kind", "—"),
                    "bm25": fmt(s["bm25_score"], 2),
                    "vector": fmt(s["vector_score"]),
                    "fused": fmt(s["fused_score"], 4),
                    "rerank": fmt(s["rerank_score"], 2),
                }
                for s in sources
            ],
            width="stretch",
            hide_index=True,
        )
        for s in sources:
            st.markdown(f"**[{s['n']}] `{s['path']}:{s['start_line']}`**")
            st.code(s["text"], language=s["language"] or "text")


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    page = st.radio("Page", ["💬 Chat", "📊 Benchmark"], label_visibility="collapsed")

    st.markdown("---")
    method_label = st.selectbox(
        "Retrieval method",
        list(METHODS),
        help="Switch retrievers to feel the difference. Hybrid is the real pipeline; "
             "the others are the baselines the benchmark scores against.",
    )
    method = METHODS[method_label]

    top_k = st.slider("Chunks passed to the LLM", 3, 12, config.FINAL_TOP_K)

    total, repos, languages, kinds_count = cached_stats()

    selected_kinds = st.multiselect(
        "Search in",
        list(config.CHUNK_KINDS),
        default=[],
        help="Leave empty to search everything. On repos with a big doc site, the prose "
             "often outranks the code it describes — pick `source` to get implementations.",
        format_func=lambda k: f"{k} ({kinds_count.get(k, 0)})",
    )
    kinds = selected_kinds or None

    st.markdown("---")
    if total:
        st.caption(f"**Index:** {total:,} chunks")
        for name, count in sorted(repos.items()):
            st.caption(f"📦 `{name}` · {count:,}")
        st.caption(" · ".join(
            f"{kind} {count}" for kind, count in sorted(kinds_count.items(), key=lambda kv: -kv[1])
        ))
        top_langs = sorted(languages.items(), key=lambda kv: -kv[1])[:5]
        st.caption(" · ".join(f"{lang} {count}" for lang, count in top_langs))
    else:
        st.error("No index found. Run `python ingest.py` first.")

    st.caption(f"Answers: `{config.OPENAI_MODEL}`")
    if not config.has_openai_key():
        st.warning("No `OPENAI_API_KEY` — retrieval works, answers are disabled.")

    if page == "💬 Chat" and st.button("🗑️ Clear chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------------------------------
# Chat page
# --------------------------------------------------------------------------
if page == "💬 Chat":
    st.title("🔍 Hybrid RAG — Codebase Assistant")
    st.caption(
        "Keyword search **and** semantic search, fused with Reciprocal Rank Fusion and "
        "reranked by a cross-encoder. Every answer cites `file:line`."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_sources(msg.get("sources", []))

    if not st.session_state.messages:
        st.markdown("**Try one of these** — note how the exact-identifier and "
                    "natural-language queries stress different retrievers:")
        cols = st.columns(3)
        for i, example in enumerate(EXAMPLES):
            if cols[i % 3].button(example, width="stretch", key=f"ex{i}"):
                st.session_state.pending = example
                st.rerun()

    question = st.chat_input("Ask about the codebase…")
    if not question and "pending" in st.session_state:
        question = st.session_state.pop("pending")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner(f"Retrieving with {method_label.split(' (')[0]}…"):
            text, sources = rag.answer(question, k=top_k, method=method, kinds=kinds)
        st.session_state.messages.append(
            {"role": "assistant", "content": text, "sources": sources}
        )
        st.rerun()


# --------------------------------------------------------------------------
# Benchmark page
# --------------------------------------------------------------------------
else:
    import benchmark

    st.title("📊 Does hybrid retrieval actually win?")
    st.caption(
        "Every query below is labelled with the file that *should* be retrieved. "
        "Each one runs three ways and is scored on whether that file shows up."
    )

    queries = benchmark.QUERIES
    st.markdown(
        f"**{len(queries)} labelled queries** · "
        f"{sum(1 for q in queries if q['kind'] == 'identifier')} exact-identifier, "
        f"{sum(1 for q in queries if q['kind'] == 'natural')} natural-language"
    )

    mode = st.radio(
        "Scoring mode",
        ["Lenient — code *or* its docs page counts", "Strict — source files only"],
        help="This repo ships a doc site mirroring its modules, so most questions have two "
             "correct answers. Strict mode excludes prose and demands the implementation.",
    )
    strict = mode.startswith("Strict")

    if st.button("▶️ Run benchmark", type="primary"):
        progress = st.progress(0.0, text="Starting…")

        def on_progress(done, total, label):
            progress.progress(done / total, text=f"[{done}/{total}] {label}")

        results = benchmark.run(strict=strict, on_progress=on_progress)
        progress.empty()
        st.session_state.benchmark = results

    if "benchmark" in st.session_state:
        results = st.session_state.benchmark
        summary = results["summary"]

        st.subheader("Results")
        st.caption(
            ("**Strict** — scored against the implementing source file, searching source "
             "chunks only." if results["strict"] else
             "**Lenient** — the implementing source file *or* its documentation page counts.")
            + f"  k={results['k']}"
        )
        best = max(summary, key=lambda m: summary[m]["hit_rate"])
        cols = st.columns(len(summary))
        for col, (name, row) in zip(cols, summary.items()):
            col.metric(
                benchmark.METHOD_LABELS[name],
                f"{row['hit_rate']:.0%}",
                delta="best" if name == best else None,
                help=f"MRR {row['mrr']:.3f}",
            )

        st.dataframe(
            [
                {
                    "method": benchmark.METHOD_LABELS[name],
                    "hit-rate@k": f"{row['hit_rate']:.0%}",
                    "hits": f"{row['hits']}/{row['total']}",
                    "MRR": f"{row['mrr']:.3f}",
                    "identifier queries": f"{row['by_kind']['identifier']:.0%}",
                    "natural-language queries": f"{row['by_kind']['natural']:.0%}",
                }
                for name, row in summary.items()
            ],
            width="stretch",
            hide_index=True,
        )

        st.subheader("Per-query breakdown")
        st.caption("✅ expected file retrieved (with its rank) · ❌ missed")
        st.dataframe(
            [
                {
                    "query": r["query"],
                    "kind": r["kind"],
                    "expects": r["expect"],
                    **{
                        benchmark.METHOD_LABELS[m]: (
                            f"✅ #{r['ranks'][m]}" if r["ranks"][m] else "❌"
                        )
                        for m in summary
                    },
                }
                for r in results["per_query"]
            ],
            width="stretch",
            hide_index=True,
        )

        for note in benchmark.narrate(results):
            st.info(note)
