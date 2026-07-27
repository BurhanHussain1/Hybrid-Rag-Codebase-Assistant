"""
Phase 3 — Hybrid retrieval. This is the core of the project.

    query
      ├─ vector search (ChromaDB / MiniLM)  -> top 20 by meaning
      └─ BM25 search   (rank_bm25)          -> top 20 by exact tokens
                    │
              Reciprocal Rank Fusion         -> one merged ranking (top 20)
                    │
            cross-encoder rerank             -> top 6, scored by actual relevance
                    │
                 chunks + per-stage scores

Why each stage exists:

* Two retrievers, because neither is enough on its own. Ask for `JWT_SECRET` and
  the embedding model has almost nothing to work with — a constant name carries
  no semantics. Ask "how does login work?" and BM25 has nothing to match, because
  the code says `authenticate`, not "login".

* RRF instead of blending scores, because BM25 scores and cosine distances live
  on incomparable scales — normalizing them means inventing a conversion factor
  that changes per query. RRF ignores magnitudes and fuses *ranks*, so a chunk
  that both retrievers put near the top wins regardless of scale.

* A cross-encoder last, because both retrievers score the query and the chunk
  separately. The cross-encoder reads them *together* and can tell that a chunk
  merely mentioning "password" isn't the one that hashes it. It's far too slow
  to run over the whole corpus, which is exactly why it goes last — 20 chunks in,
  6 out.

Usage:
    python retrieve.py "how does authentication work"
    python retrieve.py --method bm25 "JWTStrategy"
    python retrieve.py --k 10 "password hashing"
"""

import argparse

import config
import ingest


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------

def rrf(rankings, k=None):
    """
    Reciprocal Rank Fusion: merge several ranked id lists into one.

    Each list contributes 1/(k + rank) per item. The constant k (60 is the value
    from the original TREC paper) flattens the curve near the top so that rank 1
    doesn't utterly dominate rank 2 — that tolerance is the point, since either
    retriever can be confidently wrong.

    Returns [(chunk_id, fused_score), ...] ordered best first.
    """
    k = config.RRF_K if k is None else k
    scores = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


# --------------------------------------------------------------------------
# Chunk store — every chunk's text + metadata, keyed by chunk_id
# --------------------------------------------------------------------------

_store = None  # {"chunks": {chunk_id: record}, "indexes": [bm25 payloads]}


def get_store(repo=None):
    """Load the pickled BM25 indexes and their chunk records once, then reuse."""
    global _store
    if _store is None:
        payloads = ingest.load_bm25(repo)
        if not payloads:
            raise RuntimeError(
                "No BM25 index found in bm25_index/. Run:  python ingest.py"
            )
        chunks = {}
        for payload in payloads:
            for record in payload["chunks"]:
                chunks[record["chunk_id"]] = record
        _store = {"chunks": chunks, "indexes": payloads}
    return _store


def get_chunk(chunk_id):
    return get_store()["chunks"].get(chunk_id)


# --------------------------------------------------------------------------
# Retriever 1 — BM25 keyword search
# --------------------------------------------------------------------------

def bm25_search(query, k=None, repo=None, kinds=None):
    """
    Rank chunks by BM25 over the code-aware tokenization in config.tokenize.

    Returns [(chunk_id, bm25_score), ...] best first.

    Each repo keeps its own BM25 corpus (scores are corpus-relative), so with
    several repos ingested we take each one's top-k and merge. Fine in practice
    because RRF downstream only cares about order.

    `kinds` filters *before* truncating to k, so asking for source-only returns
    k source chunks rather than whatever survives from a docs-dominated top-k.
    """
    k = config.BM25_TOP_K if k is None else k
    tokens = config.tokenize(query)
    if not tokens:
        return []

    results = []
    for payload in get_store(repo)["indexes"]:
        if repo and payload["repo"] != repo:
            continue
        scores = payload["bm25"].get_scores(tokens)
        chunks = payload["chunks"]
        eligible = [
            i for i in range(len(scores))
            if scores[i] > 0 and (not kinds or chunks[i].get("kind") in kinds)
        ]
        eligible.sort(key=lambda i: scores[i], reverse=True)
        results.extend((chunks[i]["chunk_id"], float(scores[i])) for i in eligible[:k])

    results.sort(key=lambda kv: kv[1], reverse=True)
    return results[:k]


# --------------------------------------------------------------------------
# Retriever 2 — semantic vector search
# --------------------------------------------------------------------------

_collection = None


def get_vector_collection():
    global _collection
    if _collection is None:
        _collection = ingest.get_collection(create=False)
    return _collection


def build_where(repo=None, kinds=None):
    """Chroma metadata filter for repo / kind (needs $and once there are two)."""
    clauses = []
    if repo:
        clauses.append({"repo": repo})
    if kinds:
        clauses.append({"kind": {"$in": list(kinds)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def vector_search(query, k=None, repo=None, kinds=None):
    """
    Rank chunks by cosine similarity of MiniLM embeddings.

    Returns [(chunk_id, similarity), ...] best first, where similarity is
    1 - distance so that, like BM25, bigger is better.
    """
    k = config.VECTOR_TOP_K if k is None else k
    where = build_where(repo, kinds)
    res = get_vector_collection().query(query_texts=[query], n_results=k, where=where)
    if not res["ids"] or not res["ids"][0]:
        return []
    return [
        (cid, 1.0 - float(dist))
        for cid, dist in zip(res["ids"][0], res["distances"][0])
    ]


# --------------------------------------------------------------------------
# Reranker — cross-encoder
# --------------------------------------------------------------------------

_reranker = None


def get_reranker():
    """
    Load the cross-encoder once, preferring the local copy from download_model.py.

    Returns None if the model can't be loaded — no network, torch missing — so
    that retrieval degrades to RRF-only instead of failing outright. That
    fallback is what keeps a stalled model download from taking the app down
    with it; results get slightly worse, nothing breaks.
    """
    global _reranker
    if _reranker is None:
        try:
            # transformers prints a weight-loading progress bar to stderr on
            # every load; silence it so CLI output stays readable.
            import logging

            logging.getLogger("transformers").setLevel(logging.ERROR)

            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder(config.cross_encoder_source())
        except Exception as e:
            print(f"warning: cross-encoder unavailable ({type(e).__name__}); "
                  f"falling back to RRF order", flush=True)
            _reranker = False  # cache the failure; don't retry on every query
    return _reranker or None


def rerank(query, candidates, top_k):
    """
    Reorder candidate chunks by cross-encoder relevance and keep the best top_k.

    `candidates` is a list of chunk dicts; each gets a "rerank_score". Falls back
    to the incoming (RRF) order when the model isn't available.
    """
    model = get_reranker()
    if model is None or not candidates:
        for c in candidates:
            c["rerank_score"] = None
        return candidates[:top_k]

    pairs = [(query, f"{c['path']}\n{c['text']}") for c in candidates]
    scores = model.predict(pairs)
    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]


# --------------------------------------------------------------------------
# The hybrid pipeline
# --------------------------------------------------------------------------

def search(query, top_k=None, repo=None, method="hybrid", kinds=None):
    """
    Retrieve the most relevant chunks for a query.

    method:
        "hybrid" — both retrievers + RRF + cross-encoder rerank (the real thing)
        "bm25"   — keyword only          } the two baselines the Phase 5
        "vector" — semantic only         } benchmark compares hybrid against
        "rrf"    — both + RRF, no rerank (isolates the reranker's contribution)

    kinds: restrict to chunk kinds, e.g. ("source",) to exclude prose. Applied
    inside each retriever rather than as a post-filter, so a source-only search
    still gets a full candidate pool instead of the leftovers of a docs-heavy one.

    Every returned chunk carries the scores from each stage it went through
    (bm25_score / vector_score / fused_score / rerank_score), which is what the
    UI's "Retrieved files" panel displays. A None score means "this retriever
    never surfaced this chunk".
    """
    top_k = config.FINAL_TOP_K if top_k is None else top_k

    # Run only the retrievers this method needs, exactly once each.
    bm25_ranked = (bm25_search(query, repo=repo, kinds=kinds)
                   if method in ("hybrid", "bm25", "rrf") else [])
    vec_ranked = (vector_search(query, repo=repo, kinds=kinds)
                  if method in ("hybrid", "vector", "rrf") else [])
    bm25_hits, vec_hits = dict(bm25_ranked), dict(vec_ranked)

    fused_scores = {}
    if method == "bm25":
        ordered = [cid for cid, _ in bm25_ranked]
    elif method == "vector":
        ordered = [cid for cid, _ in vec_ranked]
    else:
        fused = rrf([[cid for cid, _ in bm25_ranked], [cid for cid, _ in vec_ranked]])
        ordered = [cid for cid, _ in fused]
        fused_scores = dict(fused)

    # Materialize candidates, dropping ids whose chunk record is missing (can
    # happen if chroma_db/ and bm25_index/ drift out of sync).
    limit = config.FUSED_TOP_K if method in ("hybrid", "rrf") else top_k
    candidates = []
    for cid in ordered[:limit]:
        record = get_chunk(cid)
        if record is None:
            continue
        candidates.append({
            **record,
            "bm25_score": bm25_hits.get(cid),
            "vector_score": vec_hits.get(cid),
            "fused_score": fused_scores.get(cid),
            "rerank_score": None,
        })

    if method == "hybrid":
        return rerank(query, candidates, top_k)
    return candidates[:top_k]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def format_score(value, width=7):
    return f"{value:>{width}.3f}" if value is not None else " " * (width - 1) + "-"


def main():
    config.use_utf8_stdout()

    parser = argparse.ArgumentParser(description="Hybrid code search.")
    parser.add_argument("query", nargs="+")
    parser.add_argument("--method", default="hybrid", choices=["hybrid", "bm25", "vector", "rrf"])
    parser.add_argument("--k", type=int, default=config.FINAL_TOP_K)
    parser.add_argument("--repo", default=None)
    parser.add_argument("--kind", action="append", choices=list(config.CHUNK_KINDS),
                        help="restrict to a chunk kind (repeatable), e.g. --kind source")
    parser.add_argument("--show", action="store_true", help="print each chunk's code")
    args = parser.parse_args()

    query = " ".join(args.query)
    results = search(query, top_k=args.k, repo=args.repo, method=args.method, kinds=args.kind)

    scope = f"  kinds={','.join(args.kind)}" if args.kind else ""
    print(f"\nQuery : {query}")
    print(f"Method: {args.method}{scope}    ({len(results)} results)")
    print("=" * 96)
    print(f"{'#':<3} {'bm25':>7} {'vector':>7} {'fused':>7} {'rerank':>8}  {'kind':<9} location")
    print("-" * 96)
    for i, c in enumerate(results, 1):
        print(f"{i:<3} {format_score(c['bm25_score'])} {format_score(c['vector_score'])} "
              f"{format_score(c['fused_score'])} {format_score(c['rerank_score'], 8)}  "
              f"{c.get('kind', '?'):<9} {c['path']}:{c['start_line']}")
        if args.show:
            print()
            for line in c["text"].splitlines()[:15]:
                print(f"      | {line}")
            print()
    print("=" * 96)


if __name__ == "__main__":
    main()
