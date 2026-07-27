"""
Phase 1/2 — Repository ingestion.

Turns a Git repository into the two indexes the hybrid retriever searches:

    repo URL or local path
        -> clone into repos/<name>            (GitPython, skipped if present)
        -> walk + filter files                (source only: no lockfiles, no binaries)
        -> code-aware chunk per language      (RecursiveCharacterTextSplitter)
        -> attach metadata                    (repo / path / start_line / language)
        -> ChromaDB vector index + BM25 keyword index

Every chunk carries an exact `path:start_line`, which is what makes the answers
citable. Line numbers come from the splitter's `add_start_index` offset, not from
searching for the chunk text afterwards (which breaks on repeated code).

Usage:
    python ingest.py                                  # the default demo repo
    python ingest.py https://github.com/owner/name    # clone and ingest a repo
    python ingest.py C:\\path\\to\\local\\project       # ingest a local folder
    python ingest.py <url-a> <url-b> <path-c>         # several repos in one run
    python ingest.py --chunks-only                    # chunk + report, build no indexes
    python ingest.py --reset                          # wipe these repos' chunks first
"""

import sys
from collections import Counter
from pathlib import Path

import config  # sets ANONYMIZED_TELEMETRY before chromadb is imported anywhere

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter


# --------------------------------------------------------------------------
# Getting the repository onto disk
# --------------------------------------------------------------------------

def repo_name_from_url(url):
    """`https://github.com/owner/name.git` -> `name`."""
    return url.rstrip("/").removesuffix(".git").rsplit("/", 1)[-1]


def resolve_repo(source):
    """
    Return (local_path, repo_name) for a GitHub URL or a local folder path.

    Cloning is skipped when repos/<name> already exists, so re-running ingest
    after a crash costs nothing.
    """
    if source.startswith(("http://", "https://", "git@")):
        name = repo_name_from_url(source)
        dest = config.REPOS_DIR / name
        if dest.exists():
            print(f"Using existing clone: {dest}", flush=True)
        else:
            from git import Repo  # imported lazily so --chunks-only on a local path is fast

            config.REPOS_DIR.mkdir(exist_ok=True)
            print(f"Cloning {source} -> {dest} ...", flush=True)
            # depth=1: we only ever read the working tree, never the history.
            Repo.clone_from(source, dest, depth=1)
            print("Clone complete.", flush=True)
        return dest, name

    path = Path(source).expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(f"Not a directory and not a git URL: {source}")
    return path, path.name


# --------------------------------------------------------------------------
# File filtering
# --------------------------------------------------------------------------

def classify_file(path):
    """
    Return a language label for a file we want to index, or None to skip it.

    Skips by name (lockfiles), by pattern (minified/generated), and by extension
    (anything not in config.EXTENSION_LANGUAGE that isn't a known special file).
    """
    name = path.name.lower()

    if name in config.SKIP_FILENAMES:
        return None
    if any(s in name for s in config.SKIP_SUBSTRINGS):
        return None

    suffix = path.suffix.lower()
    if suffix in config.EXTENSION_LANGUAGE:
        return config.EXTENSION_LANGUAGE[suffix]

    # Extensionless / specially-named files worth keeping.
    stem = name.split(".")[0]
    if stem in config.INCLUDE_FILENAMES:
        return "dockerfile" if stem == "dockerfile" else "text"
    if name.startswith(config.INCLUDE_PREFIXES):
        return "markdown"

    return None


def iter_source_files(root):
    """Yield (path, language) for every indexable file under root."""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        # Prune anything living inside a skipped directory.
        if any(part in config.SKIP_DIRS for part in path.relative_to(root).parts[:-1]):
            continue
        language = classify_file(path)
        if language is None:
            continue
        try:
            if path.stat().st_size > config.MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path, language


def read_text(path):
    """Read a file as UTF-8, returning None for binary or undecodable content."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:8192]:  # NUL byte in the head == binary
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

# Language labels that langchain has real syntax separators for. Anything else
# (yaml, json, toml, hcl, text) falls back to the generic recursive splitter.
_ENUM_LANGUAGES = {lang.value for lang in Language}

_splitter_cache = {}


def get_splitter(language):
    """
    A splitter for this language, built once and reused.

    `from_language` gives separators that break on real syntax boundaries — for
    Python that's `\\nclass `, `\\ndef `, `\\n\\tdef ` — so a chunk tends to hold a
    whole function instead of half of two. Unsupported languages get the generic
    paragraph/line splitter, which is fine for config and data files.
    """
    if language not in _splitter_cache:
        kwargs = dict(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            add_start_index=True,  # exact byte offset -> exact line number
        )
        if language in _ENUM_LANGUAGES:
            splitter = RecursiveCharacterTextSplitter.from_language(Language(language), **kwargs)
        else:
            splitter = RecursiveCharacterTextSplitter(**kwargs)
        _splitter_cache[language] = splitter
    return _splitter_cache[language]


def chunk_file(text, rel_path, language, repo):
    """
    Split one file's text into chunk dicts with citation-ready metadata.

    `start_line` is derived by counting newlines before the chunk's offset, so
    `path:start_line` always points at the real first line of the chunk even
    when the same snippet appears several times in the file.
    """
    chunks = []
    for i, doc in enumerate(get_splitter(language).create_documents([text])):
        body = doc.page_content
        if len(body.strip()) < config.MIN_CHUNK_CHARS:
            continue
        offset = doc.metadata.get("start_index", 0)
        chunks.append({
            "chunk_id": f"{repo}::{rel_path}::{i}",
            "text": body,
            "repo": repo,
            "path": rel_path,
            "start_line": text.count("\n", 0, offset) + 1,
            "language": language,
            "kind": config.chunk_kind(rel_path, language),
        })
    return chunks


def collect_chunks(source=None, verbose=True):
    """
    Full Phase 1 pipeline: resolve the repo, walk it, chunk it.

    Returns (chunks, repo_name, stats). One bad file never kills the run — it is
    counted in stats["failed"] and skipped.
    """
    source = source or config.DEFAULT_REPO_URL
    root, repo = resolve_repo(source)

    files = list(iter_source_files(root))
    if verbose:
        print(f"\nScanning {root}\nFound {len(files)} indexable files.", flush=True)

    chunks = []
    by_language = Counter()
    files_by_language = Counter()
    by_kind = Counter()
    skipped = failed = 0

    for path, language in files:
        text = read_text(path)
        if text is None or not text.strip():
            skipped += 1
            continue
        rel_path = path.relative_to(root).as_posix()
        try:
            file_chunks = chunk_file(text, rel_path, language, repo)
        except Exception as e:
            failed += 1
            if verbose:
                print(f"  FAILED {rel_path}: {type(e).__name__}", flush=True)
            continue
        chunks.extend(file_chunks)
        by_language[language] += len(file_chunks)
        files_by_language[language] += 1
        for c in file_chunks:
            by_kind[c["kind"]] += 1

    stats = {
        "root": root,
        "files_scanned": len(files),
        "files_chunked": len(files) - skipped - failed,
        "skipped": skipped,
        "failed": failed,
        "chunks": len(chunks),
        "by_language": by_language,
        "files_by_language": files_by_language,
        "by_kind": by_kind,
    }
    return chunks, repo, stats


def print_report(repo, stats, chunks):
    """Human-readable summary of what ingestion produced."""
    print(f"\n{'=' * 68}")
    print(f"Repo: {repo}")
    print(f"{'=' * 68}")
    print(f"  files scanned : {stats['files_scanned']}")
    print(f"  files chunked : {stats['files_chunked']}")
    print(f"  skipped       : {stats['skipped']} (binary / empty / unreadable)")
    print(f"  failed        : {stats['failed']}")
    print(f"  total chunks  : {stats['chunks']}")

    print(f"\n  {'language':<12} {'files':>7} {'chunks':>8}")
    print(f"  {'-' * 12} {'-' * 7} {'-' * 8}")
    for language, count in stats["by_language"].most_common():
        print(f"  {language:<12} {stats['files_by_language'][language]:>7} {count:>8}")

    # Worth watching: on a repo with a real doc site, prose can be a third of the
    # index and will out-compete the code it describes on both retrievers.
    total = max(stats["chunks"], 1)
    print(f"\n  {'kind':<12} {'chunks':>8} {'share':>7}")
    print(f"  {'-' * 12} {'-' * 8} {'-' * 7}")
    for kind, count in stats["by_kind"].most_common():
        print(f"  {kind:<12} {count:>8} {count / total:>6.0%}")

    if chunks:
        # Show a middle chunk — the first file in a repo is usually __init__.py.
        sample = chunks[len(chunks) // 2]
        print(f"\n  Sample chunk")
        print(f"  {'-' * 60}")
        print(f"  chunk_id   : {sample['chunk_id']}")
        print(f"  citation   : {sample['path']}:{sample['start_line']}")
        print(f"  language   : {sample['language']}")
        print(f"  chars      : {len(sample['text'])}")
        print(f"  {'-' * 60}")
        for line in sample["text"].splitlines()[:12]:
            print(f"  | {line}")
        print(f"  | ...")


# --------------------------------------------------------------------------
# Index building (Phase 2)
# --------------------------------------------------------------------------
# Two indexes over the same chunks, each good at what the other is bad at:
#
#   ChromaDB  — MiniLM embeddings. Finds meaning: "where does login happen?"
#   BM25Okapi — token frequency. Finds exact symbols: `JWT_SECRET`.
#
# retrieve.py queries both and fuses the rankings.

CHROMA_BATCH = 200  # chunks embedded per upsert call


def get_collection(create=True):
    """Open (or create) the ChromaDB collection with the local MiniLM embedder."""
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    if create:
        return client.get_or_create_collection(config.COLLECTION_NAME, embedding_function=embed_fn)
    return client.get_collection(config.COLLECTION_NAME, embedding_function=embed_fn)


def bm25_path(repo):
    return config.BM25_DIR / f"{repo}.pkl"


def build_vector_index(chunks, repo, reset=False):
    """
    Upsert chunks into ChromaDB, embedding with the local MiniLM model.

    Resumable: chunk ids are deterministic, so a run that dies halfway can just
    be re-run — ids already in the collection are skipped, and `upsert` makes a
    re-embed harmless anyway.
    """
    collection = get_collection()

    if reset:
        existing = collection.get(where={"repo": repo}, include=[])["ids"]
        if existing:
            for i in range(0, len(existing), 500):
                collection.delete(ids=existing[i:i + 500])
            print(f"  reset: removed {len(existing)} existing chunks for '{repo}'", flush=True)
        todo = chunks
    else:
        have = set(collection.get(where={"repo": repo}, include=[])["ids"])
        todo = [c for c in chunks if c["chunk_id"] not in have]
        if have:
            print(f"  resume: {len(have)} chunks already embedded, {len(todo)} to go", flush=True)

    for start in range(0, len(todo), CHROMA_BATCH):
        batch = todo[start:start + CHROMA_BATCH]
        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "repo": c["repo"],
                    "path": c["path"],
                    "start_line": c["start_line"],
                    "language": c["language"],
                    "kind": c["kind"],
                }
                for c in batch
            ],
        )
        done = min(start + CHROMA_BATCH, len(todo))
        print(f"  embedding {done}/{len(todo)} ...", flush=True)

    print(f"  vector index: {collection.count()} chunks total in collection", flush=True)
    return collection.count()


def build_bm25_index(chunks, repo):
    """
    Build a BM25Okapi index over the chunks and pickle it with its chunk store.

    The file path is prepended to each document's token stream so that searching
    for `jwt` also favours chunks living in `strategy/jwt.py` — in code, the path
    is a strong relevance signal that the file's text alone may not carry.
    """
    import pickle

    from rank_bm25 import BM25Okapi

    corpus = [config.tokenize(c["path"] + " " + c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus)

    config.BM25_DIR.mkdir(exist_ok=True)
    payload = {"repo": repo, "bm25": bm25, "chunks": chunks}

    # Write to a temp file and rename, so an interrupted write can't leave a
    # half-pickled index behind that fails to load on the next run.
    target = bm25_path(repo)
    tmp = target.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(target)

    size_mb = target.stat().st_size / 1_000_000
    print(f"  BM25 index : {len(chunks)} chunks -> {target.name} ({size_mb:.1f} MB)", flush=True)
    return len(chunks)


def load_bm25(repo=None):
    """
    Load pickled BM25 indexes.

    Returns a list of {repo, bm25, chunks} payloads — one per ingested repo, or
    just the requested one. Each repo keeps its own BM25 corpus because BM25
    scores are corpus-relative and can't simply be concatenated.
    """
    import pickle

    if not config.BM25_DIR.exists():
        return []
    paths = [bm25_path(repo)] if repo else sorted(config.BM25_DIR.glob("*.pkl"))
    loaded = []
    for path in paths:
        if not path.exists():
            continue
        try:
            with open(path, "rb") as f:
                loaded.append(pickle.load(f))
        except Exception as e:
            print(f"warning: could not load {path.name}: {type(e).__name__}", flush=True)
    return loaded


def build_indexes(chunks, repo, reset=False):
    """Build both indexes over the same chunk list."""
    if not chunks:
        print("\nNo chunks to index.")
        return

    print(f"\nBuilding indexes for '{repo}' ({len(chunks)} chunks)...", flush=True)
    build_vector_index(chunks, repo, reset=reset)
    build_bm25_index(chunks, repo)


def main():
    config.use_utf8_stdout()

    args = sys.argv[1:]
    flags = {a for a in args if a.startswith("--")}
    sources = [a for a in args if not a.startswith("--")] or [config.DEFAULT_REPO_URL]

    # Each repo is ingested independently: its own chunk ids, its own BM25
    # corpus, its own metadata tag. A failure on one doesn't touch the others,
    # and re-running with a new URL adds to the index rather than replacing it.
    ingested, failed = [], []
    for n, source in enumerate(sources, 1):
        if len(sources) > 1:
            print(f"\n{'#' * 68}\n# [{n}/{len(sources)}] {source}\n{'#' * 68}", flush=True)
        try:
            chunks, repo, stats = collect_chunks(source)
        except Exception as e:
            print(f"FAILED to ingest {source}: {type(e).__name__}: {e}", flush=True)
            failed.append(source)
            continue

        print_report(repo, stats, chunks)

        if "--chunks-only" in flags:
            print("\n--chunks-only: stopping before index build.")
            continue

        build_indexes(chunks, repo, reset="--reset" in flags)
        ingested.append((repo, len(chunks)))

    if ingested:
        total = sum(n for _, n in ingested)
        names = ", ".join(f"{repo} ({n})" for repo, n in ingested)
        print(f"\nIndexed {total} chunks across {len(ingested)} repo(s): {names}")
        print('Try:  python retrieve.py "how does authentication work"')
    if failed:
        print(f"\n{len(failed)} source(s) failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
