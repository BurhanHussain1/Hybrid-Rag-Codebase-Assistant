"""
Resumable downloader for the cross-encoder reranker.

Why this exists: `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")` normally
downloads the model itself on first use. On a slow or flaky connection that
transfer can stall at zero bytes and hang the whole app with no progress output
and no resume — you just wait, kill it, and start over from nothing.

This script fetches the same files with explicit HTTP Range requests, so a
dropped connection resumes from where it stopped instead of restarting. Files
land in `models/ms-marco-MiniLM-L-6-v2/`, which retrieve.py prefers over the Hub
when it exists.

Run once (it is safe to re-run — completed files are skipped):
    python download_model.py
"""

import sys
import time
from pathlib import Path

import urllib.request
import urllib.error

import config

REPO = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BASE = f"https://huggingface.co/{REPO}/resolve/main"
DEST = config.MODELS_DIR / "ms-marco-MiniLM-L-6-v2"

# The weights file is the big one (~90 MB); the rest are tiny config/tokenizer
# files. `required=False` entries are nice-to-have — sentence-transformers works
# without them.
FILES = [
    ("config.json", True),
    ("model.safetensors", True),
    ("tokenizer.json", True),
    ("tokenizer_config.json", True),
    ("vocab.txt", True),
    ("special_tokens_map.json", False),
]

CHUNK = 1 << 16          # 64 KB per read
STALL_SECONDS = 45       # no bytes for this long -> treat the socket as dead
MAX_ATTEMPTS = 12


def remote_size(url):
    """Content-Length via a HEAD-ish GET, or None if the server won't say."""
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def download(filename, required):
    """
    Fetch one file, resuming from whatever is already on disk.

    Returns True on success. A missing optional file is not an error.
    """
    url = f"{BASE}/{filename}"
    target = DEST / filename
    partial = target.with_suffix(target.suffix + ".part")

    total = remote_size(url)
    if target.exists() and (total is None or target.stat().st_size == total):
        print(f"  {filename:<26} already complete")
        return True

    for attempt in range(1, MAX_ATTEMPTS + 1):
        have = partial.stat().st_size if partial.exists() else 0
        if total and have >= total:
            break

        headers = {"Range": f"bytes={have}-"} if have else {}
        request = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status == 200 and have:
                    # Server ignored the Range header — start over rather than
                    # append to a prefix that may not match.
                    have = 0
                    partial.unlink(missing_ok=True)

                mode = "ab" if have else "wb"
                last_progress = time.time()
                with open(partial, mode) as f:
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        f.write(block)
                        have += len(block)
                        now = time.time()
                        if now - last_progress > 0.5:
                            pct = f"{100 * have / total:5.1f}%" if total else "     "
                            print(f"\r  {filename:<26} {pct}  "
                                  f"{have / 1e6:7.1f} MB", end="", flush=True)
                            last_progress = now

            if total is None or have >= total:
                break

        except (urllib.error.URLError, TimeoutError, OSError) as e:
            got = partial.stat().st_size if partial.exists() else 0
            if required:
                print(f"\r  {filename:<26} attempt {attempt}/{MAX_ATTEMPTS} "
                      f"stopped at {got / 1e6:.1f} MB ({type(e).__name__}); resuming…",
                      flush=True)
            elif attempt >= 2:
                print(f"\r  {filename:<26} optional, skipping")
                return True
            time.sleep(min(2 * attempt, 15))
            continue

    if not partial.exists():
        if required:
            print(f"\r  {filename:<26} FAILED")
            return False
        print(f"\r  {filename:<26} optional, skipping")
        return True

    partial.replace(target)
    print(f"\r  {filename:<26} done      {target.stat().st_size / 1e6:7.1f} MB")
    return True


def main():
    config.use_utf8_stdout()
    DEST.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {REPO}\n  -> {DEST}\n")
    ok = True
    for filename, required in FILES:
        if not download(filename, required) and required:
            ok = False

    if ok:
        print(f"\nModel ready. retrieve.py will load it from {DEST} instead of the Hub.")
    else:
        print("\nSome required files are missing — re-run this script to resume.")
        sys.exit(1)


if __name__ == "__main__":
    main()
