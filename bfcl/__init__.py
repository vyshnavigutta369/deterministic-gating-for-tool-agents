"""
BFCL (Berkeley Function-Calling Leaderboard) multi_turn harness.

vendor/ contains a minimal, unmodified copy of the real bfcl-eval package
(https://pypi.org/project/bfcl-eval/, official repo
https://github.com/ShishirPatil/gorilla) -- just the multi_turn execution
engine (execute_multi_turn_func_call), the real simulator classes
(GorillaFileSystem, TwitterAPI, etc.), and the BFCL_v4_multi_turn_base
dataset (200 real, human-curated tasks) -- not the full ~13MB package
(which also bundles unrelated categories, model handlers for many LLM
providers, and scripts we don't need). Vendored rather than pip-installed
because bfcl-eval's PyPI metadata requires Python >=3.10, but this project
runs 3.9; the actual code has no 3.10-only syntax and imports fine.

Nothing in vendor/ is modified from the original package.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
