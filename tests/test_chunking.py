"""
Chunking, metadata, and file filtering.

`start_line` is the load-bearing field in this project: every citation the app
prints is `path:start_line`, and if it drifts the answers still *look* right
while pointing at the wrong code. Nothing else in the pipeline would notice, so
it gets the most direct test here — including the case that motivated computing
it from the splitter's offset rather than by searching for the chunk text.
"""

import config
import ingest


def make_source(n_functions=40):
    """A Python file with uniquely identifiable, line-aligned functions."""
    blocks = []
    for i in range(n_functions):
        blocks.append(
            f"def function_number_{i}(argument):\n"
            f"    \"\"\"Docstring for function {i}.\"\"\"\n"
            f"    value = argument * {i}\n"
            f"    return value\n"
        )
    return "\n".join(blocks)


def test_start_line_points_at_the_chunks_real_first_line():
    text = make_source()
    lines = text.split("\n")
    chunks = ingest.chunk_file(text, "pkg/module.py", "python", "demo")

    assert len(chunks) > 1, "test file should be big enough to split"
    for chunk in chunks:
        first_line = chunk["text"].split("\n")[0]
        file_line = lines[chunk["start_line"] - 1]
        # The chunk may begin mid-line, so the file's line must *contain* it.
        assert first_line in file_line, (
            f"chunk at {chunk['path']}:{chunk['start_line']} claims to start with "
            f"{first_line!r} but that line is {file_line!r}"
        )


def test_start_line_is_within_the_file():
    text = make_source()
    total = text.count("\n") + 1
    for chunk in ingest.chunk_file(text, "a.py", "python", "demo"):
        assert 1 <= chunk["start_line"] <= total


def test_repeated_code_gets_distinct_line_numbers():
    # The reason start_line comes from the splitter offset: searching for the
    # chunk text would return the FIRST occurrence for every duplicate block.
    block = (
        "def handler(request):\n"
        "    validate(request)\n"
        "    result = process(request)\n"
        "    return result\n"
    )
    filler = "".join(f"# padding line {i}\n" for i in range(60))
    text = block + filler + block + filler + block

    chunks = ingest.chunk_file(text, "dup.py", "python", "demo")
    handler_lines = sorted(
        c["start_line"] for c in chunks if c["text"].lstrip().startswith("def handler")
    )
    assert len(handler_lines) >= 2, "expected the duplicated block in several chunks"
    assert len(set(handler_lines)) == len(handler_lines), (
        f"duplicate blocks collapsed onto the same line number: {handler_lines}"
    )


def test_first_chunk_of_a_file_starts_at_line_one():
    text = make_source(5)
    chunks = ingest.chunk_file(text, "a.py", "python", "demo")
    assert chunks[0]["start_line"] == 1


def test_chunk_ids_are_unique_and_stable():
    text = make_source()
    first = ingest.chunk_file(text, "pkg/module.py", "python", "demo")
    second = ingest.chunk_file(text, "pkg/module.py", "python", "demo")

    ids = [c["chunk_id"] for c in first]
    assert len(ids) == len(set(ids)), "chunk ids collide — upsert would drop chunks"
    # Stability is what makes ingest resumable and upserts idempotent.
    assert ids == [c["chunk_id"] for c in second]
    assert ids[0].startswith("demo::pkg/module.py::")


def test_metadata_is_complete():
    chunk = ingest.chunk_file(make_source(3), "pkg/mod.py", "python", "demo")[0]
    assert set(chunk) >= {
        "chunk_id", "text", "repo", "path", "start_line", "language", "kind"
    }
    assert chunk["repo"] == "demo"
    assert chunk["path"] == "pkg/mod.py"
    assert chunk["kind"] == "source"


def test_trivial_chunks_are_dropped():
    assert ingest.chunk_file("\n\n   \n", "empty.py", "python", "demo") == []


def test_unsupported_language_falls_back_instead_of_raising():
    # yaml/json/toml have no Language enum entry; they must still chunk.
    text = "\n".join(f"key_{i}: value_{i}" for i in range(200))
    chunks = ingest.chunk_file(text, "conf.yaml", "yaml", "demo")
    assert chunks and all(c["start_line"] >= 1 for c in chunks)


# --------------------------------------------------------------------------
# File filtering
# --------------------------------------------------------------------------

class FakePath:
    """Minimal stand-in for pathlib.Path — classify_file only reads .name/.suffix."""

    def __init__(self, name):
        self.name = name
        self.suffix = ("." + name.rsplit(".", 1)[1]) if "." in name[1:] else ""


def test_source_extensions_are_recognised():
    assert ingest.classify_file(FakePath("service.py")) == "python"
    assert ingest.classify_file(FakePath("app.tsx")) == "ts"
    assert ingest.classify_file(FakePath("main.go")) == "go"


def test_docs_and_special_filenames_are_recognised():
    assert ingest.classify_file(FakePath("README.md")) == "markdown"
    assert ingest.classify_file(FakePath("Dockerfile")) == "dockerfile"


def test_noise_is_rejected():
    for name in ("package-lock.json", "yarn.lock", "bundle.min.js",
                 "app.js.map", "photo.png", "binary.exe"):
        assert ingest.classify_file(FakePath(name)) is None, f"{name} should be skipped"


# --------------------------------------------------------------------------
# Content classification
# --------------------------------------------------------------------------

def test_chunk_kind_classification():
    cases = {
        "fastapi_users/manager.py": "source",
        "src/requests/sessions.py": "source",
        "tests/test_manager.py": "tests",
        "pkg/conftest.py": "tests",
        "examples/beanie/app/users.py": "examples",
        "docs/configuration/oauth.md": "docs",
    }
    for path, expected in cases.items():
        language = "markdown" if path.endswith(".md") else "python"
        assert config.chunk_kind(path, language) == expected, path


def test_top_level_markdown_counts_as_docs():
    assert config.chunk_kind("README.md", "markdown") == "docs"
