"""
Single source of truth for every tunable in the Hybrid RAG pipeline.

Everything else (ingest / retrieve / rag / app / benchmark) imports from here, so
changing a chunk size or a top-k happens in exactly one place.

Nothing in this module is expensive to import: no models are loaded and no
network calls are made. It only reads `.env` and defines constants.
"""

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

# Quiet ChromaDB's telemetry reporter. Must be set BEFORE chromadb is imported
# anywhere, which is why it lives in the module every other module imports.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

load_dotenv()

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).parent
REPOS_DIR = ROOT / "repos"          # cloned repositories
CHROMA_DIR = str(ROOT / "chroma_db")  # ChromaDB persistent store (needs a str)
BM25_DIR = ROOT / "bm25_index"      # pickled BM25 index + aligned chunk store
MODELS_DIR = ROOT / "models"        # locally downloaded models (see download_model.py)

COLLECTION_NAME = "code_chunks"

# --------------------------------------------------------------------------
# Demo repository (Phase 1 works this one end-to-end before generalizing)
# --------------------------------------------------------------------------
DEFAULT_REPO_URL = "https://github.com/fastapi-users/fastapi-users"

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
# Embeddings: ChromaDB's built-in all-MiniLM-L6-v2 (384-dim, ONNX). Local + free,
# no torch required. The same function must be used at ingest and query time or
# the vectors aren't comparable.
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# A local copy of the cross-encoder, if `python download_model.py` has been run.
# Preferred over the Hub id because the Hub client has no resume on this
# connection — see download_model.py for why that matters.
CROSS_ENCODER_LOCAL = MODELS_DIR / "ms-marco-MiniLM-L-6-v2"


def cross_encoder_source():
    """Local model directory when present, otherwise the Hub id."""
    if (CROSS_ENCODER_LOCAL / "config.json").exists():
        return str(CROSS_ENCODER_LOCAL)
    return CROSS_ENCODER_MODEL

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = 0

# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
VECTOR_TOP_K = 20   # candidates pulled from the vector index
BM25_TOP_K = 20     # candidates pulled from the keyword index
RRF_K = 60          # Reciprocal Rank Fusion damping constant
FUSED_TOP_K = 20    # how many fused candidates get sent to the cross-encoder
FINAL_TOP_K = 6     # how many chunks reach the LLM

# --------------------------------------------------------------------------
# File filtering
# --------------------------------------------------------------------------
# extension -> language label. The label doubles as the key we hand to
# RecursiveCharacterTextSplitter.from_language() (see ingest.LANGUAGE_SPLITTERS).
EXTENSION_LANGUAGE = {
    ".py": "python",
    ".js": "js",
    ".jsx": "js",
    ".ts": "ts",
    ".tsx": "ts",
    ".java": "java",
    ".go": "go",
    ".cs": "csharp",
    ".rb": "ruby",
    ".rs": "rust",
    ".md": "markdown",
    ".rst": "rst",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".tf": "hcl",
}

# Extensionless / specially-named files we still want.
INCLUDE_FILENAMES = {"dockerfile", "makefile", "procfile"}
INCLUDE_PREFIXES = ("readme",)  # README, README.md, README.rst, ...

# Directories never walked into.
SKIP_DIRS = {
    ".git", ".github", ".hg", ".svn",
    "node_modules", "venv", ".venv", "env", "__pycache__",
    "dist", "build", ".next", ".nuxt", "out", "target",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
    "site-packages", "vendor", "coverage", "htmlcov",
    ".idea", ".vscode",
}

# Exact filenames that are noise (lockfiles etc.).
SKIP_FILENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "cargo.lock", "composer.lock", "gemfile.lock", "go.sum",
}

# Substrings that mark a file as generated/minified.
SKIP_SUBSTRINGS = (".min.js", ".min.css", ".map", ".lock", ".bundle.js")

MAX_FILE_BYTES = 1_000_000  # ~1 MB — anything larger is data, not source
MIN_CHUNK_CHARS = 30        # drop whitespace-only / trivial fragments


# --------------------------------------------------------------------------
# Chunk kind
# --------------------------------------------------------------------------
# Repos with a real doc site are mostly prose *about* the code, and that prose
# out-competes the code on both retrievers: it's written in the same natural
# language as the query, and it repeats an identifier more often than the one
# line that defines it. Often that's the right answer — a guide explains the
# flow better than the source. Sometimes it isn't: "show me the implementation"
# means the implementation.
#
# So tag every chunk and let the caller choose, rather than guessing centrally.
CHUNK_KINDS = ("source", "docs", "tests", "examples")

_TEST_MARKERS = ("test_", "_test.", "conftest")


def chunk_kind(path, language):
    """Classify a repo-relative path as source / docs / tests / examples."""
    parts = path.lower().split("/")
    name = parts[-1]

    if any(p in ("tests", "test", "__tests__", "spec") for p in parts[:-1]):
        return "tests"
    if any(marker in name for marker in _TEST_MARKERS):
        return "tests"
    if any(p in ("examples", "example", "samples", "demo") for p in parts[:-1]):
        return "examples"
    if any(p in ("docs", "doc", "documentation", "website") for p in parts[:-1]):
        return "docs"
    if language in ("markdown", "rst"):
        return "docs"
    return "source"


# --------------------------------------------------------------------------
# BM25 tokenization
# --------------------------------------------------------------------------
# Lives here (not in ingest.py) because the index and the query MUST be
# tokenized identically — one definition, imported by both sides.
_WORD = re.compile(r"[A-Za-z0-9]+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")


def tokenize(text):
    """
    Tokenize code for BM25.

    Plain word-splitting is useless on code: a search for `UserSessionManager`
    would never match a chunk that talks about "the session manager", and a
    search for "session manager" would never match the class. So each raw token
    is kept AND split into its camelCase / snake_case / PascalCase parts:

        UserSessionManager -> usersessionmanager, user, session, manager
        JWT_SECRET         -> jwt, secret
        getHTTPResponse    -> gethttpresponse, get, http, response

    Keeping the whole token preserves exact-identifier precision; the subtokens
    add the recall that makes BM25 useful on natural-language queries too.
    """
    tokens = []
    for raw in _WORD.findall(text):
        lowered = raw.lower()
        tokens.append(lowered)
        parts = _CAMEL.findall(raw)
        if len(parts) > 1:
            tokens.extend(p.lower() for p in parts)
    return tokens


def has_openai_key():
    """True when a real OpenAI key is configured (placeholder doesn't count)."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return bool(key) and key != "sk-your-key-here"


def use_utf8_stdout():
    """Make emoji / box-drawing output safe in the Windows console."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
