"""
BM25 tokenization.

This is the highest-leverage thing to test in the whole retrieval path: index
and query go through the same function, so a change here silently degrades every
keyword search rather than raising anything. The behaviour it guarantees is that
an identifier is findable *both* by its exact name and by its word parts.
"""

import config


def test_keeps_whole_identifier_and_its_parts():
    tokens = config.tokenize("UserSessionManager")
    # exact-symbol search must still hit
    assert "usersessionmanager" in tokens
    # ...and so must "session manager"
    assert {"user", "session", "manager"} <= set(tokens)


def test_splits_snake_case():
    tokens = config.tokenize("JWT_SECRET")
    assert {"jwt", "secret"} <= set(tokens)


def test_handles_acronym_runs():
    # getHTTPResponse must not become get/h/t/t/p/response
    tokens = config.tokenize("getHTTPResponse")
    assert {"gethttpresponse", "get", "http", "response"} <= set(tokens)


def test_plain_word_is_not_duplicated():
    # A single-part token has no sub-parts worth adding; emitting it twice would
    # inflate its term frequency and skew BM25 toward ordinary prose.
    assert config.tokenize("login") == ["login"]


def test_lowercases_so_case_never_affects_matching():
    assert config.tokenize("JWTStrategy") == config.tokenize("jwtstrategy") + ["jwt", "strategy"] \
           or set(config.tokenize("jwtstrategy")) <= set(config.tokenize("JWTStrategy"))


def test_query_and_document_tokenize_identically():
    # The whole scheme depends on symmetry between index time and query time.
    text = "class UserSessionManager:  # handles JWT_SECRET"
    assert config.tokenize(text) == config.tokenize(text)
    assert set(config.tokenize("UserSessionManager")) <= set(config.tokenize(text))


def test_punctuation_and_empties_do_not_crash():
    assert config.tokenize("") == []
    assert config.tokenize("...///---") == []
    assert config.tokenize("a.b->c()") == ["a", "b", "c"]


def test_digits_are_kept():
    assert "oauth2" in config.tokenize("OAuth2")
