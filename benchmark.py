"""
Phase 5 — Does hybrid retrieval actually beat either retriever alone?

The claim behind this whole project is that keyword search and semantic search
fail in *different* places, so fusing them covers both gaps. That's easy to
assert and easy to get wrong, so this measures it.

Method
------
Each query below is labelled with the file that genuinely answers it — the file
that *defines* the thing, not one that merely mentions it. Every query runs
through four retrieval configurations and is scored on whether that file appears
in the top-k chunks:

    bm25    keyword only          — the exact-identifier specialist
    vector  semantic only         — the natural-language specialist
    rrf     both, fused, no rerank — isolates what fusion alone buys
    hybrid  fused + cross-encoder — the full pipeline

Metrics
-------
hit-rate@k  fraction of queries where the expected file made the top k.
            "Did the user get their answer at all?"
MRR         mean of 1/rank of the first correct chunk. Rewards ranking the right
            file 1st over 6th, which hit-rate can't see.

Two scoring modes
-----------------
The demo repo ships a documentation site that mirrors its modules, so most
questions have two genuinely correct answers: the code, and the page describing
it. Which one counts depends on what the user meant, so both are reported:

    lenient (default)  the implementing source file OR its canonical docs page.
                       "How does signing up work?" is answered by
                       router/register.py *or* docs/configuration/routers/register.md.
    strict             the source file only, searching source chunks only.
                       "Show me the implementation" — prose doesn't count.

Honesty notes
-------------
* The query set is small (17) and hand-labelled, so treat the numbers as a
  directional signal, not a leaderboard entry.
* The queries were written from the repo's public API surface before any results
  were looked at. The `also_ok` docs paths were added *after* a first run, when
  it turned out the retrievers were returning correct doc pages that strict
  labels scored as misses. That was a labelling error, not a retrieval failure —
  but it is a post-hoc widening, so both modes are reported rather than only the
  friendlier one. The widening applies identically to every method, so it cannot
  favour hybrid.
* Identifier queries are never widened. Typing a class name means "take me to
  its definition", so only the defining source file counts.
* Ground truth stays tight otherwise — usually one file per query. Accepting any
  plausibly related file would push every method toward 100% and measure nothing.

Usage:
    python benchmark.py
    python benchmark.py --k 3
"""

import argparse

import config
import retrieve

METHODS = ["bm25", "vector", "rrf", "hybrid"]

METHOD_LABELS = {
    "bm25": "BM25 only",
    "vector": "Vector only",
    "rrf": "RRF (no rerank)",
    "hybrid": "Hybrid (full)",
}

# kind="identifier" — an exact symbol a developer would paste in. Lexical match
#                    is available; semantics barely are.
# kind="natural"    — a question in English. The words in the query mostly do
#                    NOT appear in the code that answers it.
QUERIES = [
    # ---- exact identifiers -------------------------------------------------
    {"query": "JWTStrategy", "kind": "identifier",
     "expect": ["fastapi_users/authentication/strategy/jwt.py"]},
    {"query": "PasswordHelper", "kind": "identifier",
     "expect": ["fastapi_users/password.py"]},
    {"query": "BaseUserManager", "kind": "identifier",
     "expect": ["fastapi_users/manager.py"]},
    {"query": "CookieTransport", "kind": "identifier",
     "expect": ["fastapi_users/authentication/transport/cookie.py"]},
    {"query": "RedisStrategy", "kind": "identifier",
     "expect": ["fastapi_users/authentication/strategy/redis.py"]},
    {"query": "InvalidPasswordException", "kind": "identifier",
     "expect": ["fastapi_users/exceptions.py"]},
    {"query": "generate_state_token", "kind": "identifier",
     "expect": ["fastapi_users/router/oauth.py"]},
    {"query": "BearerTransport", "kind": "identifier",
     "expect": ["fastapi_users/authentication/transport/bearer.py"]},

    # ---- natural language --------------------------------------------------
    # `also_ok` = the repo's own documentation page for the same subject. Counted
    # in lenient mode only; see the module docstring for why.
    {"query": "How are passwords hashed and checked?", "kind": "natural",
     "expect": ["fastapi_users/password.py"],
     "also_ok": ["docs/configuration/password-hash.md"]},
    {"query": "How does the app decide whether a request is from a logged-in user?",
     "kind": "natural",
     "expect": ["fastapi_users/authentication/authenticator.py"],
     "also_ok": ["docs/usage/current-user.md"]},
    {"query": "Where are the login and logout endpoints defined?", "kind": "natural",
     "expect": ["fastapi_users/router/auth.py"],
     "also_ok": ["docs/configuration/routers/auth.md"]},
    {"query": "How does a brand new user sign up for an account?", "kind": "natural",
     "expect": ["fastapi_users/router/register.py"],
     "also_ok": ["docs/configuration/routers/register.md"]},
    {"query": "What happens when someone forgets their password?", "kind": "natural",
     "expect": ["fastapi_users/router/reset.py"],
     "also_ok": ["docs/configuration/routers/reset.md"]},
    {"query": "How is a JSON web token created and decoded?", "kind": "natural",
     "expect": ["fastapi_users/jwt.py"],
     "also_ok": ["docs/configuration/authentication/strategies/jwt.md"]},
    {"query": "How does signing in with a third-party social account work?",
     "kind": "natural",
     "expect": ["fastapi_users/router/oauth.py"],
     "also_ok": ["docs/configuration/oauth.md"]},
    {"query": "How does confirming an email address work?", "kind": "natural",
     "expect": ["fastapi_users/router/verify.py"],
     "also_ok": ["docs/configuration/routers/verify.md"]},
    {"query": "Where are access tokens persisted to the database?", "kind": "natural",
     "expect": ["fastapi_users/authentication/strategy/db/strategy.py",
                "fastapi_users/authentication/strategy/db/adapter.py"],
     "also_ok": ["docs/configuration/authentication/strategies/database.md"]},
]


def accepted_paths(spec, strict):
    """The set of paths that count as a correct answer for this query."""
    if strict:
        return set(spec["expect"])
    return set(spec["expect"]) | set(spec.get("also_ok", []))


def first_match(results, accepted):
    """(1-based rank, matched path) of the first accepted result, else (None, None)."""
    for rank, chunk in enumerate(results, 1):
        if chunk["path"] in accepted:
            return rank, chunk["path"]
    return None, None


def run(k=None, repo=None, methods=None, strict=False, on_progress=None):
    """
    Run every query through every method.

    strict=True scores against the implementing source file only, and searches
    source chunks only — the "show me the code, not the guide" test.

    Returns {"k", "strict", "per_query", "summary"}. `on_progress(done, total,
    label)` is called as it goes so the Streamlit page can show a progress bar.
    """
    k = config.FINAL_TOP_K if k is None else k
    methods = methods or METHODS
    kinds = ("source",) if strict else None

    total_steps = len(QUERIES) * len(methods)
    done = 0
    per_query = []

    for spec in QUERIES:
        accepted = accepted_paths(spec, strict)
        ranks, matched = {}, {}
        for method in methods:
            results = retrieve.search(
                spec["query"], top_k=k, repo=repo, method=method, kinds=kinds
            )
            ranks[method], matched[method] = first_match(results, accepted)
            done += 1
            if on_progress:
                on_progress(done, total_steps, f"{METHOD_LABELS[method]} — {spec['query'][:48]}")
        per_query.append({
            "query": spec["query"],
            "kind": spec["kind"],
            "expect": ", ".join(sorted(accepted)),
            "ranks": ranks,
            "matched": matched,
        })

    summary = {}
    for method in methods:
        hits = [r for r in per_query if r["ranks"][method]]
        by_kind = {}
        for kind in ("identifier", "natural"):
            of_kind = [r for r in per_query if r["kind"] == kind]
            hit_of_kind = [r for r in of_kind if r["ranks"][method]]
            by_kind[kind] = len(hit_of_kind) / len(of_kind) if of_kind else 0.0
        summary[method] = {
            "hits": len(hits),
            "total": len(per_query),
            "hit_rate": len(hits) / len(per_query),
            "mrr": sum(1.0 / r["ranks"][method] for r in hits) / len(per_query),
            "by_kind": by_kind,
        }

    return {"k": k, "strict": strict, "per_query": per_query, "summary": summary}


def narrate(results):
    """
    Turn the numbers into the observations that matter, computed from the data
    rather than asserted — so this stays honest even if hybrid loses.
    """
    per_query, summary = results["per_query"], results["summary"]
    notes = []

    def missed(method, row):
        return row["ranks"].get(method) is None

    bm25_saves = [r["query"] for r in per_query
                  if r["ranks"].get("bm25") and missed("vector", r)]
    vector_saves = [r["query"] for r in per_query
                    if r["ranks"].get("vector") and missed("bm25", r)]
    hybrid_saves = [r["query"] for r in per_query
                    if r["ranks"].get("hybrid") and missed("bm25", r) and missed("vector", r)]

    if bm25_saves:
        notes.append(
            f"**Keyword search wins {len(bm25_saves)} quer{'y' if len(bm25_saves) == 1 else 'ies'} "
            f"the vector index misses entirely** — e.g. `{bm25_saves[0]}`. "
            "An identifier carries almost no semantic signal to embed."
        )
    if vector_saves:
        notes.append(
            f"**Vector search wins {len(vector_saves)} quer{'y' if len(vector_saves) == 1 else 'ies'} "
            f"BM25 misses entirely** — e.g. \"{vector_saves[0]}\". "
            "The question and the code that answers it share almost no words."
        )
    if hybrid_saves:
        notes.append(
            f"**Hybrid recovers {len(hybrid_saves)} quer{'y' if len(hybrid_saves) == 1 else 'ies'} "
            f"that *both* single retrievers miss** — e.g. \"{hybrid_saves[0]}\". "
            "Fusion promotes chunks each retriever ranked mid-pack."
        )

    if "rrf" in summary and "hybrid" in summary:
        delta = summary["hybrid"]["mrr"] - summary["rrf"]["mrr"]
        if abs(delta) >= 0.005:
            direction = "improves" if delta > 0 else "hurts"
            notes.append(
                f"The cross-encoder {direction} MRR by {abs(delta):.3f} over plain RRF "
                f"({summary['rrf']['mrr']:.3f} → {summary['hybrid']['mrr']:.3f}) — "
                "reranking mostly reorders what fusion already found rather than finding more."
            )

    best = max(summary, key=lambda m: (summary[m]["hit_rate"], summary[m]["mrr"]))
    singles = [m for m in ("bm25", "vector") if m in summary]
    if singles and "hybrid" in summary:
        best_single = max(singles, key=lambda m: summary[m]["hit_rate"])
        h, s = summary["hybrid"]["hit_rate"], summary[best_single]["hit_rate"]
        if h > s:
            notes.append(
                f"**Hybrid beats the best single retriever** "
                f"({h:.0%} vs {s:.0%} for {METHOD_LABELS[best_single]}) — the point of the project."
            )
        elif h == s:
            notes.append(
                f"Hybrid ties {METHOD_LABELS[best_single]} on hit-rate ({h:.0%}); "
                f"compare MRR ({summary['hybrid']['mrr']:.3f} vs {summary[best_single]['mrr']:.3f}) "
                "for the ranking-quality difference."
            )
        else:
            notes.append(
                f"⚠️ On this query set {METHOD_LABELS[best_single]} ({s:.0%}) edges out "
                f"hybrid ({h:.0%}). Reported as measured."
            )
    notes.append(f"Best overall: **{METHOD_LABELS[best]}**.")
    return notes


def main():
    config.use_utf8_stdout()

    parser = argparse.ArgumentParser(description="BM25 vs Vector vs Hybrid.")
    parser.add_argument("--k", type=int, default=config.FINAL_TOP_K)
    parser.add_argument("--repo", default=None)
    parser.add_argument("--mode", default="both", choices=["lenient", "strict", "both"])
    args = parser.parse_args()

    modes = ["lenient", "strict"] if args.mode == "both" else [args.mode]
    for mode in modes:
        report(run(k=args.k, repo=args.repo, strict=(mode == "strict")))


def report(results):
    """Print one mode's table, per-query breakdown, and observations."""
    summary = results["summary"]
    k = results["k"]

    banner = ("STRICT — source files only, prose excluded" if results["strict"]
              else "LENIENT — the source file or its documentation page counts")
    print("\n" + "#" * 78)
    print(f"# {banner}")
    print("#" * 78)

    print("=" * 78)
    print(f"{'method':<18} {'hit-rate':>9} {'hits':>8} {'MRR':>7} {'ident':>7} {'natural':>8}")
    print("-" * 78)
    for method in METHODS:
        row = summary[method]
        print(f"{METHOD_LABELS[method]:<18} {row['hit_rate']:>8.0%} "
              f"{row['hits']:>5}/{row['total']:<2} {row['mrr']:>7.3f} "
              f"{row['by_kind']['identifier']:>6.0%} {row['by_kind']['natural']:>8.0%}")
    print("=" * 78)

    print(f"\nPer-query (rank of the expected file, '-' = not in top {k}):\n")
    header = f"{'kind':<11} {'query':<52}" + "".join(f"{METHOD_LABELS[m][:9]:>11}" for m in METHODS)
    print(header)
    print("-" * len(header))
    for row in results["per_query"]:
        cells = "".join(
            f"{('#' + str(row['ranks'][m])) if row['ranks'][m] else '-':>11}" for m in METHODS
        )
        print(f"{row['kind']:<11} {row['query'][:50]:<52}{cells}")

    print("\n" + "=" * 78)
    for note in narrate(results):
        clean = note.replace("**", "").replace("`", "")
        print(f"\n* {clean}")
    print()


if __name__ == "__main__":
    main()
