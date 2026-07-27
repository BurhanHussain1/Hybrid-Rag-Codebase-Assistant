"""
Pytest configuration.

Living at the project root, this file makes pytest put the root on sys.path, so
the tests can `import config` / `import retrieve` the same way the app does —
without turning the flat layout into a package just to satisfy the test runner.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
