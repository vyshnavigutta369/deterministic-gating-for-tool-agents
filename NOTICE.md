# Third-Party Components

This repository is a research companion to the paper *"Trajectory-Level
Grading for Tool-Batching Agents: A Provably-Safe Cost Mechanism and a
Calibration Case Study."* It vendors two external components, unmodified,
for reproducibility. Neither is our contribution; see their upstream
repositories for authoritative licensing terms.

## `env/` — CostBench

Source: Liu, J., Qian, C., Su, Z., Zong, Q., Huang, S., He, B., & Fung, Y. R.
(2026). *CostBench: Evaluating Multi-Turn Cost-Optimal Planning and
Adaptation in Dynamic Environments for LLM Tool-Use Agents.* ACL 2026.
https://github.com/JiayuJeff/CostBench

CostBench is our primary evaluation environment. We use its tool registry,
solver, and travel-planning domain as-is; our contribution (`brts_core.py`,
`run_brts_minimal.py`, `brts/`, `experiments/`) sits on top of it and does
not modify `env/` itself. See the upstream repository for its license and
for the full benchmark (search-database generation, config, etc.) beyond
what this paper's experiments exercise.

## `bfcl/vendor/` — bfcl-eval (Gorilla / Berkeley Function-Calling Leaderboard)

Source: https://github.com/ShishirPatil/gorilla,
package: https://pypi.org/project/bfcl-eval/

`bfcl/vendor/bfcl_eval/` is a minimal, unmodified subset of the official
`bfcl-eval` package — just the `multi_turn` execution engine and the
`BFCL_v4_multi_turn_base` dataset — vendored rather than pip-installed
because the package's PyPI metadata requires Python >=3.10 while this
project targets 3.9 (the vendored code itself has no 3.10-only syntax).
See `bfcl/__init__.py` for details and the upstream repository for its
license.
