"""
Phase 6 — Evaluation harness.

The benchmark in benchmark.py measures *retrieval* only. This measures the whole
pipeline end to end, including what the model does with what it was given, and
scores three things per question:

  retrieval     Did a chunk from the file that actually answers the question make
                it into the context at all? Nothing downstream can succeed if this
                fails, so it's scored separately from the model's behaviour.

  citation      Does the answer carry inline [n] markers, do they all point at
                sources that exist, and is the right file among the ones cited?
                An answer that's correct but uncited is unverifiable, and a
                citation pointing at [7] when six sources were supplied is worse
                than no citation at all.

  groundedness  Of the code identifiers the answer names in `backticks`, how many
                actually appear in the retrieved context? This is the cheap
                hallucination detector: a model inventing a plausible-sounding
                method like `get_user_token()` gets caught here even when the
                prose around it reads perfectly. Ungrounded terms are printed,
                not just counted, so a failure is diagnosable.

Groundedness is a proxy, not proof — an answer can cite real identifiers and
still describe them wrongly. It catches fabrication, not misinterpretation.

Every case asks how something is implemented and is labelled with the source file
that implements it, so retrieval defaults to source chunks only. Pass --all-kinds
to let the repo's own documentation compete for the same slots.

Usage:
    python eval.py                    # full eval (uses OpenAI)
    python eval.py --retrieval-only   # no API calls, retrieval scoring only
    python eval.py --method vector    # score a different retriever
    python eval.py --all-kinds        # let docs and tests compete
    python eval.py -v                 # show ungrounded terms per case
"""

import argparse
import re

import config
import rag
import retrieve

# Each case names the file that genuinely answers the question. `must_mention`
# is a light content check — terms the answer should contain if it understood
# the question, kept loose so it tests comprehension rather than phrasing.
EVAL_SET = [
    {"q": "How does a user log in and get a token?",
     "file": "fastapi_users/router/auth.py",
     "must_mention": ["login", "strategy"]},
    {"q": "How are passwords hashed and verified?",
     "file": "fastapi_users/password.py",
     "must_mention": ["hash", "verify"]},
    {"q": "What does JWTStrategy do?",
     "file": "fastapi_users/authentication/strategy/jwt.py",
     "must_mention": ["token", "jwt"]},
    {"q": "How does the authentication backend combine transport and strategy?",
     "file": "fastapi_users/authentication/backend.py",
     "must_mention": ["transport", "strategy"]},
    {"q": "How is a new user registered?",
     "file": "fastapi_users/router/register.py",
     "must_mention": ["register", "create"]},
    {"q": "What happens when a password reset is requested?",
     "file": "fastapi_users/router/reset.py",
     "must_mention": ["reset", "token"]},
    {"q": "How does BaseUserManager validate a password?",
     "file": "fastapi_users/manager.py",
     "must_mention": ["validate", "password"]},
    {"q": "How are access tokens stored in a database strategy?",
     "file": "fastapi_users/authentication/strategy/db/strategy.py",
     "must_mention": ["token", "database"]},
]

# `code` spans that are prose or too generic to be evidence of anything.
STOPWORDS = {
    "true", "false", "none", "null", "self", "cls", "str", "int", "bool",
    "dict", "list", "get", "set", "post", "put", "id", "the", "and", "or",
    "not", "if", "else", "return", "async", "await", "def", "class", "type",
}

_BACKTICKED = re.compile(r"`([^`\n]{2,80})`")
_CITATION = re.compile(r"\[(\d{1,2})\]")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def cited_indices(text):
    """Every [n] marker in the answer, as ints."""
    return [int(m) for m in _CITATION.findall(text)]


def grounded_terms(text, context):
    """
    Split the answer's backticked identifiers into (grounded, ungrounded).

    A term counts as grounded when it appears verbatim in the retrieved context.
    Dotted paths are checked component-wise, so `strategy.write_token` passes if
    the context contains `write_token` — the attribute is what's being claimed.
    """
    grounded, ungrounded = [], []
    for span in _BACKTICKED.findall(text):
        for token in _IDENTIFIER.findall(span):
            if token.lower() in STOPWORDS:
                continue
            (grounded if token in context else ungrounded).append(token)
    return grounded, ungrounded


def score_case(case, method, k, retrieval_only, kinds):
    """Run one eval case and return a result dict."""
    chunks = retrieve.search(case["q"], top_k=k, method=method, kinds=kinds)
    paths = [c["path"] for c in chunks]
    result = {
        "q": case["q"],
        "file": case["file"],
        "retrieval": case["file"] in paths,
        "paths": paths,
    }
    if retrieval_only:
        return result

    text, sources = rag.answer(case["q"], k=k, method=method, kinds=kinds)
    context = "\n".join(c["text"] + "\n" + c["path"] for c in chunks)

    markers = cited_indices(text)
    valid_range = markers and all(1 <= n <= len(sources) for n in markers)
    cited_files = {sources[n - 1]["path"] for n in markers if 1 <= n <= len(sources)}

    grounded, ungrounded = grounded_terms(text, context)
    total_terms = len(grounded) + len(ungrounded)
    ratio = len(grounded) / total_terms if total_terms else 1.0

    low = text.lower()
    result.update({
        "has_citations": bool(markers),
        "citations_valid": bool(valid_range),
        "cites_expected": case["file"] in cited_files,
        "citation": bool(valid_range) and case["file"] in cited_files,
        "grounded_ratio": ratio,
        "ungrounded": sorted(set(ungrounded)),
        "n_terms": total_terms,
        "groundedness": ratio >= 0.8,
        "mentions": all(m.lower() in low for m in case["must_mention"]),
        "answer": text,
    })
    return result


def main():
    config.use_utf8_stdout()

    parser = argparse.ArgumentParser(description="End-to-end RAG evaluation.")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="skip generation — no OpenAI calls, no cost")
    parser.add_argument("--method", default="hybrid",
                        choices=["hybrid", "bm25", "vector", "rrf"])
    parser.add_argument("--k", type=int, default=config.FINAL_TOP_K)
    parser.add_argument("--all-kinds", action="store_true",
                        help="search docs and tests too (default: source only)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="list ungrounded terms per case")
    args = parser.parse_args()

    # Every case here asks how something is *implemented* and is labelled with a
    # source file, so source-only is the matching default. --all-kinds shows what
    # happens when the repo's own prose is allowed to compete.
    kinds = None if args.all_kinds else ("source",)

    retrieval_only = args.retrieval_only or not config.has_openai_key()
    if args.retrieval_only is False and retrieval_only:
        print("No OPENAI_API_KEY found — running retrieval-only.\n")

    n = len(EVAL_SET)
    scope = "all content" if args.all_kinds else "source only"
    print(f"Evaluating {n} cases  (method={args.method}, k={args.k}, {scope})\n")

    results = []
    for case in EVAL_SET:
        r = score_case(case, args.method, args.k, retrieval_only, kinds)
        results.append(r)

        if retrieval_only:
            print(f"  [retrieval {'PASS' if r['retrieval'] else 'fail'}]  {r['q']}")
            continue

        print(f"  [retr {'PASS' if r['retrieval'] else 'fail'} | "
              f"cite {'PASS' if r['citation'] else 'fail'} | "
              f"grnd {'PASS' if r['groundedness'] else 'fail'} "
              f"({r['grounded_ratio']:.0%} of {r['n_terms']})]  {r['q']}")
        if args.verbose and r["ungrounded"]:
            print(f"         not in context: {', '.join(r['ungrounded'][:8])}")

    print("\n" + "=" * 72)
    retr = sum(r["retrieval"] for r in results)
    print(f"Retrieval    {retr}/{n} ({retr / n:.0%})   "
          f"expected file present in the top {args.k}")

    if not retrieval_only:
        cite = sum(r["citation"] for r in results)
        grnd = sum(r["groundedness"] for r in results)
        ment = sum(r["mentions"] for r in results)
        any_cite = sum(r["has_citations"] for r in results)
        valid = sum(r["citations_valid"] for r in results)
        mean_ratio = sum(r["grounded_ratio"] for r in results) / n

        print(f"Citation     {cite}/{n} ({cite / n:.0%})   "
              f"expected file actually cited  "
              f"[{any_cite}/{n} cite anything, {valid}/{n} all markers valid]")
        print(f"Groundedness {grnd}/{n} ({grnd / n:.0%})   "
              f">=80% of backticked identifiers found in context  "
              f"(mean {mean_ratio:.0%})")
        print(f"Content      {ment}/{n} ({ment / n:.0%})   "
              f"answer mentions the expected key terms")

        stray = sorted({t for r in results for t in r["ungrounded"]})
        if stray:
            print(f"\nIdentifiers named but not present in any retrieved chunk "
                  f"({len(stray)}):\n  {', '.join(stray[:20])}")
            print("  (usually paraphrase or a stdlib name — inspect with -v if it looks wrong)")
    print("=" * 72)


if __name__ == "__main__":
    main()
