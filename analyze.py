"""
Phase 6 — Repo auto-analysis.

Works out what a repository actually *is* — languages, frameworks, databases,
test tooling, deployment setup — so the assistant can show it in the sidebar
instead of making the user guess what they're querying.

Detection reads dependency manifests first (`pyproject.toml`, `package.json`,
`go.mod`, ...) because a declared dependency is a fact, then falls back to
scanning imports and file names for things that don't show up as packages
(Dockerfiles, k8s manifests, raw SQL). Everything is best-effort: an unreadable
or exotic manifest is skipped, never fatal.

Usage:
    python analyze.py                    # the default demo repo
    python analyze.py fastapi-users      # a repo already cloned into repos/
    python analyze.py C:\\path\\to\\project
"""

import json
import re
import sys
from collections import Counter

import config
import ingest

# --------------------------------------------------------------------------
# Signatures
# --------------------------------------------------------------------------
# dependency-name substring -> display label. Matched against the package names
# parsed out of whatever manifests the repo ships.
FRAMEWORK_DEPS = {
    # Python
    "fastapi": "FastAPI", "django": "Django", "flask": "Flask",
    "starlette": "Starlette", "pyramid": "Pyramid", "tornado": "Tornado",
    "sqlalchemy": "SQLAlchemy", "pydantic": "Pydantic", "celery": "Celery",
    "langchain": "LangChain", "transformers": "Transformers", "torch": "PyTorch",
    "streamlit": "Streamlit", "beanie": "Beanie (ODM)", "alembic": "Alembic",
    "httpx": "httpx", "uvicorn": "Uvicorn",
    # JS / TS
    "react": "React", "next": "Next.js", "vue": "Vue", "svelte": "Svelte",
    "angular": "Angular", "express": "Express", "nestjs": "NestJS",
    "tailwindcss": "Tailwind CSS", "prisma": "Prisma", "webpack": "webpack",
    "vite": "Vite",
    # Other ecosystems
    "gin-gonic": "Gin", "fiber": "Fiber", "echo": "Echo",
    "spring-boot": "Spring Boot", "rails": "Ruby on Rails", "actix": "Actix",
    "axum": "Axum", "tokio": "Tokio",
}

DATABASE_DEPS = {
    "psycopg": "PostgreSQL", "asyncpg": "PostgreSQL", "postgres": "PostgreSQL",
    "pg": "PostgreSQL", "mysqlclient": "MySQL", "pymysql": "MySQL",
    "mysql": "MySQL", "sqlite": "SQLite", "aiosqlite": "SQLite",
    "pymongo": "MongoDB", "motor": "MongoDB", "mongoose": "MongoDB",
    "mongodb": "MongoDB", "redis": "Redis", "elasticsearch": "Elasticsearch",
    "cassandra": "Cassandra", "neo4j": "Neo4j", "duckdb": "DuckDB",
    "chromadb": "ChromaDB", "qdrant": "Qdrant", "pinecone": "Pinecone",
}

TEST_DEPS = {
    "pytest": "pytest", "unittest2": "unittest", "nose": "nose", "tox": "tox",
    "jest": "Jest", "vitest": "Vitest", "mocha": "Mocha", "cypress": "Cypress",
    "playwright": "Playwright", "rspec": "RSpec", "junit": "JUnit",
}

# filename (lowercased) -> what shipping it tells us
FILE_SIGNALS = {
    "dockerfile": "Docker",
    "docker-compose.yml": "Docker Compose",
    "docker-compose.yaml": "Docker Compose",
    "makefile": "Make",
    "procfile": "Heroku/Procfile",
    ".pre-commit-config.yaml": "pre-commit",
    "netlify.toml": "Netlify",
    "vercel.json": "Vercel",
    "serverless.yml": "Serverless Framework",
}

MANIFESTS = (
    "pyproject.toml", "requirements.txt", "setup.py", "setup.cfg", "Pipfile",
    "package.json", "go.mod", "Cargo.toml", "Gemfile", "pom.xml",
    "build.gradle", "composer.json",
)

_PY_DEP = re.compile(r"^\s*[\"']?([A-Za-z0-9._-]+)", re.MULTILINE)
_GO_DEP = re.compile(r"^\s*([\w.\-/]+)\s+v", re.MULTILINE)
_TOML_DEP = re.compile(r"^\s*([A-Za-z0-9._-]+)\s*=", re.MULTILINE)


def _deps_from_manifest(name, text):
    """Best-effort package-name extraction from one manifest file."""
    lower = name.lower()
    names = set()

    if lower == "package.json":
        try:
            data = json.loads(text)
        except Exception:
            return names
        for key in ("dependencies", "devDependencies", "peerDependencies"):
            names.update(data.get(key, {}) or {})
        return names

    if lower == "go.mod":
        for match in _GO_DEP.findall(text):
            names.add(match.rsplit("/", 1)[-1])
            names.add(match)
        return names

    if lower in ("pyproject.toml", "cargo.toml", "pipfile"):
        # Covers both [tool.poetry.dependencies] style and PEP 621
        # dependencies = ["fastapi>=0.100", ...]
        names.update(_TOML_DEP.findall(text))
        for line in re.findall(r'"([A-Za-z0-9._\[\]-]+)\s*[><=~!]', text):
            names.add(line)
        names.update(re.findall(r'"([A-Za-z0-9._-]+)"\s*,', text))
        return names

    if lower in ("pom.xml", "build.gradle", "composer.json", "gemfile",
                 "setup.py", "setup.cfg", "requirements.txt"):
        names.update(_PY_DEP.findall(text))
        names.update(re.findall(r"[\"']([A-Za-z0-9._-]+)[\"']", text))
        return names

    return names


def _match(names, table):
    """Map a set of raw dependency names onto display labels via substring rules."""
    found = set()
    lowered = [n.lower() for n in names if n]
    for needle, label in table.items():
        for name in lowered:
            # exact, or a scoped/prefixed variant like @types/react or react-dom
            if name == needle or name.startswith(needle + "-") or name.endswith("/" + needle):
                found.add(label)
                break
    return found


def analyze(source=None):
    """
    Inspect a repository and return a dict describing it.

    `source` may be a repo name already cloned under repos/, a path, or a git
    URL (which is NOT cloned here — analysis only reads what's on disk).
    """
    source = source or config.DEFAULT_REPO_URL
    candidate = config.REPOS_DIR / source
    if candidate.is_dir():
        root, repo = candidate, source
    else:
        root, repo = ingest.resolve_repo(source)

    languages = Counter()
    kinds = Counter()
    dep_names = set()
    signals = set()
    manifests_seen = []
    total_bytes = 0
    file_count = 0

    for path, language in ingest.iter_source_files(root):
        file_count += 1
        languages[language] += 1
        rel = path.relative_to(root).as_posix()
        kinds[config.chunk_kind(rel, language)] += 1
        try:
            total_bytes += path.stat().st_size
        except OSError:
            pass

        name = path.name.lower()
        if name in FILE_SIGNALS:
            signals.add(FILE_SIGNALS[name])
        if path.name in MANIFESTS:
            manifests_seen.append(rel)
            text = ingest.read_text(path)
            if text:
                dep_names |= _deps_from_manifest(path.name, text)

    # Directory-level signals the file walk can't see (CI configs are skipped by
    # the ingest filter, and k8s manifests are just yaml).
    if (root / ".github" / "workflows").is_dir():
        signals.add("GitHub Actions")
    if (root / ".gitlab-ci.yml").exists():
        signals.add("GitLab CI")
    if any(root.rglob("*.tf")):
        signals.add("Terraform")
    if (root / "k8s").is_dir() or (root / "kubernetes").is_dir():
        signals.add("Kubernetes")

    return {
        "repo": repo,
        "root": root,
        "files": file_count,
        "size_mb": total_bytes / 1_000_000,
        "languages": dict(languages.most_common()),
        "kinds": dict(kinds.most_common()),
        "frameworks": sorted(_match(dep_names, FRAMEWORK_DEPS)),
        "databases": sorted(_match(dep_names, DATABASE_DEPS)),
        "testing": sorted(_match(dep_names, TEST_DEPS)),
        "signals": sorted(signals),
        "manifests": manifests_seen,
        "dependency_count": len(dep_names),
    }


def summary_line(info):
    """One-sentence description, e.g. 'A Python project using FastAPI, SQLAlchemy.'"""
    languages = list(info["languages"])
    primary = {"js": "JavaScript", "ts": "TypeScript", "csharp": "C#"}.get(
        languages[0], languages[0].title()
    ) if languages else "Unknown"
    parts = [f"A {primary} project"]
    if info["frameworks"]:
        parts.append("using " + ", ".join(info["frameworks"][:4]))
    if info["databases"]:
        parts.append("with " + ", ".join(info["databases"][:3]))
    return " ".join(parts) + "."


def main():
    config.use_utf8_stdout()
    source = sys.argv[1] if len(sys.argv) > 1 else None
    info = analyze(source)

    print(f"\n{'=' * 68}")
    print(f"  {info['repo']}")
    print(f"{'=' * 68}")
    print(f"  {summary_line(info)}\n")
    print(f"  indexable files : {info['files']}  ({info['size_mb']:.1f} MB)")
    print(f"  dependencies    : {info['dependency_count']} declared")

    def row(label, values):
        print(f"  {label:<16}: {', '.join(values) if values else '—'}")

    row("frameworks", info["frameworks"])
    row("databases", info["databases"])
    row("testing", info["testing"])
    row("infra / tooling", info["signals"])
    row("manifests", info["manifests"][:5])

    print(f"\n  {'language':<12} {'files':>7}")
    print(f"  {'-' * 12} {'-' * 7}")
    for language, count in info["languages"].items():
        print(f"  {language:<12} {count:>7}")

    print(f"\n  {'content':<12} {'files':>7}")
    print(f"  {'-' * 12} {'-' * 7}")
    for kind, count in info["kinds"].items():
        print(f"  {kind:<12} {count:>7}")
    print()


if __name__ == "__main__":
    main()
