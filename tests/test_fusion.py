"""
Reciprocal Rank Fusion.

RRF is four lines of arithmetic that decide the entire ranking, and a mistake in
it (an off-by-one on rank, the wrong k, summing scores instead of reciprocals)
produces output that still looks like a plausible ranking. So the numbers are
asserted against hand-computed values rather than just checking the order.
"""

import config
import retrieve


def test_scores_match_the_formula():
    # Single ranking: score(rank i, 0-based) == 1 / (k + i)
    fused = dict(retrieve.rrf([["a", "b", "c"]], k=60))
    assert fused["a"] == 1 / 60
    assert fused["b"] == 1 / 61
    assert fused["c"] == 1 / 62


def test_agreement_between_retrievers_beats_a_single_top_hit():
    # This is the entire point of fusion: "b" is second on both lists and must
    # outrank "a", which only one retriever liked.
    ranked = retrieve.rrf([["a", "b"], ["c", "b"]], k=60)
    order = [cid for cid, _ in ranked]
    assert order[0] == "b"

    scores = dict(ranked)
    assert scores["b"] == 1 / 61 + 1 / 61
    assert scores["a"] == 1 / 60


def test_results_are_ordered_best_first():
    ranked = retrieve.rrf([["x", "y", "z"], ["z", "y", "x"]])
    scores = [score for _, score in ranked]
    assert scores == sorted(scores, reverse=True)


def test_small_k_sharpens_the_top_of_the_ranking():
    # k damps how much rank 1 dominates rank 2; a smaller k widens the gap.
    def gap(k):
        s = dict(retrieve.rrf([["a", "b"]], k=k))
        return s["a"] - s["b"]

    assert gap(1) > gap(60)


def test_handles_empty_and_partial_input():
    assert retrieve.rrf([]) == []
    assert retrieve.rrf([[], []]) == []
    assert [cid for cid, _ in retrieve.rrf([["only"], []])] == ["only"]


def test_default_k_comes_from_config():
    assert dict(retrieve.rrf([["a"]]))["a"] == 1 / config.RRF_K


def test_every_input_id_survives_fusion():
    # Fusion must never silently drop a candidate — that would hide a chunk
    # from the reranker entirely.
    left, right = ["a", "b", "c"], ["c", "d", "e"]
    fused = {cid for cid, _ in retrieve.rrf([left, right])}
    assert fused == set(left) | set(right)
