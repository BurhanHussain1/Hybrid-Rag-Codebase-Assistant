"""
Scoring helpers for the benchmark, the eval harness, and prompt assembly.

These decide what gets *reported*, so a bug here doesn't break the app — it
quietly produces flattering numbers. That makes them worth pinning down: the
lenient/strict distinction in particular exists to keep the benchmark honest,
and it's only honest if strict actually stays strict.
"""

import benchmark
import rag


# --------------------------------------------------------------------------
# Benchmark ground truth
# --------------------------------------------------------------------------

SPEC = {
    "query": "how does signup work?",
    "kind": "natural",
    "expect": ["pkg/register.py"],
    "also_ok": ["docs/register.md"],
}


def test_strict_mode_excludes_documentation():
    assert benchmark.accepted_paths(SPEC, strict=True) == {"pkg/register.py"}


def test_lenient_mode_accepts_the_docs_page_too():
    assert benchmark.accepted_paths(SPEC, strict=False) == {
        "pkg/register.py", "docs/register.md"
    }


def test_identifier_queries_are_never_widened():
    # Typing a class name means "take me to the definition", so identifier specs
    # carry no also_ok and both modes must agree.
    for spec in benchmark.QUERIES:
        if spec["kind"] == "identifier":
            assert "also_ok" not in spec, spec["query"]
            assert (benchmark.accepted_paths(spec, True)
                    == benchmark.accepted_paths(spec, False))


def test_every_query_declares_expected_paths():
    assert benchmark.QUERIES, "query set must not be empty"
    for spec in benchmark.QUERIES:
        assert spec["expect"], spec["query"]
        assert spec["kind"] in ("identifier", "natural")


def test_first_match_reports_rank_and_path():
    results = [
        {"path": "other.py"},
        {"path": "pkg/register.py"},
        {"path": "pkg/register.py"},
    ]
    assert benchmark.first_match(results, {"pkg/register.py"}) == (2, "pkg/register.py")


def test_first_match_returns_none_when_absent():
    assert benchmark.first_match([{"path": "a.py"}], {"b.py"}) == (None, None)


def test_first_match_on_empty_results():
    assert benchmark.first_match([], {"a.py"}) == (None, None)


# --------------------------------------------------------------------------
# Eval harness — citations and groundedness
# --------------------------------------------------------------------------

def test_citation_markers_are_extracted():
    import eval as eval_mod

    text = "The backend pairs them [2], and login delegates to write_token [5][6]."
    assert eval_mod.cited_indices(text) == [2, 5, 6]


def test_no_markers_means_no_citations():
    import eval as eval_mod

    assert eval_mod.cited_indices("An answer with no citations at all.") == []


def test_groundedness_flags_an_invented_identifier():
    import eval as eval_mod

    context = "async def write_token(self, user): ..."
    text = "It calls `write_token` and then `get_user_token`."
    grounded, ungrounded = eval_mod.grounded_terms(text, context)

    assert "write_token" in grounded
    assert "get_user_token" in ungrounded, "hallucinated method should be caught"


def test_groundedness_checks_dotted_paths_component_wise():
    import eval as eval_mod

    context = "token = await strategy.write_token(user)"
    grounded, ungrounded = eval_mod.grounded_terms("See `strategy.write_token`.", context)
    assert not ungrounded
    assert {"strategy", "write_token"} <= set(grounded)


def test_generic_words_are_not_treated_as_evidence():
    import eval as eval_mod

    # `True` / `str` appearing in an answer says nothing about grounding.
    grounded, ungrounded = eval_mod.grounded_terms("Pass `True` and `str`.", "")
    assert grounded == [] and ungrounded == []


def test_prose_without_backticks_is_ignored():
    import eval as eval_mod

    grounded, ungrounded = eval_mod.grounded_terms("No code spans here at all.", "")
    assert grounded == [] and ungrounded == []


# --------------------------------------------------------------------------
# Prompt assembly
# --------------------------------------------------------------------------

def test_context_numbering_matches_the_source_list():
    chunks = [
        {"repo": "demo", "path": "a.py", "start_line": 10,
         "language": "python", "text": "x = 1"},
        {"repo": "demo", "path": "b.py", "start_line": 42,
         "language": "python", "text": "y = 2"},
    ]
    context = rag.build_context(chunks)

    # The [n] the model is told to cite must line up with the sources shown.
    assert "[1] a.py:10" in context
    assert "[2] b.py:42" in context

    sources = rag.to_sources(chunks)
    assert [s["n"] for s in sources] == [1, 2]
    assert sources[1]["path"] == "b.py" and sources[1]["start_line"] == 42


def test_context_includes_the_code():
    chunks = [{"repo": "demo", "path": "a.py", "start_line": 1, "language": "python",
               "text": "def hello():\n    return 1"}]
    assert "def hello():" in rag.build_context(chunks)
