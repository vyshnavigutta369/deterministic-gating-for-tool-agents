# Trajectory-Level Grading for Tool-Batching Agents

Code accompanying the paper *"Trajectory-Level Grading for Tool-Batching
Agents: A Provably-Safe Cost Mechanism and a Calibration Case Study"*
(submitted to the Workshop on Evaluation of Interactive Agents, NeurIPS 2026).

**What this paper studies.** When an LLM agent proposes several tool calls
at once to cut round-trip latency and cost, something has to judge whether
the batch is safe to execute and whether it was worth executing. We report
two results, evaluated trajectory-level on real API episodes:

1. A deterministic **gate** (Section 3.1) that verifies every proposed
   tool's preconditions against the pre-round state, plus two
   provably-safe, composable post-gate refinements — **redundancy
   elimination** and **cost-substitution** (Section 3.2) — that recover
   real cost savings without ever changing round count or increasing cost.
2. A **grader-calibration case study** (Section 5.3): a factorial ablation
   comparing the deterministic gate against two model-based baselines
   ("blind" batching and "self-judged" batching, matching prior work's
   actual mechanisms) across gpt-4o-mini, GPT-4o, and Claude Haiku,
   scaled incrementally (n=5 → n=20 → n=50 → n=100) to see which
   safety-divergence findings survive more data.

This repository is a **companion for reproducing those results**, not a
general-purpose tool-batching framework. It is organized so each script
maps to a specific table or figure in the paper (see below).

## Relationship to CostBench

[CostBench](https://github.com/JiayuJeff/CostBench) (Liu et al., 2026, ACL
2026) is our primary evaluation environment. `env/` is CostBench's harness,
vendored unmodified — we use its tool registry, ground-truth solver, and
travel-planning domain as-is. Our contribution sits on top of it:

- `brts_core.py`, `run_brts_minimal.py` — the gate (v1/v2/v3), the two
  post-gate cost mechanisms, and the four-condition safety-mode harness
  (`gated` / `blind` / `self_judged` / `gated_self_assessed`).
- `brts/` — Appendix A.1's deterministic, grader-free alternatives
  (scored-greedy negative control, precompute-then-batch,
  precompute-length-optimal) and the theory behind when they're optimal.
- `experiments/` — the sweep runner and every reproduction script.
- `bfcl/`, `logistics/` — the two secondary domains (Section 4, Appendix
  A.2.3): BFCL `parallel`/`multi_turn` and PDDL Logistics via `pyperplan`.

See `NOTICE.md` for the two vendored third-party components (`env/` and
`bfcl/vendor/`) and their upstream licenses.

## Setup

```bash
pip install -r requirements.txt

export OPENAI_API_KEY="your-key-here"       # GPT-4o, gpt-4o-mini
export ANTHROPIC_API_KEY="your-key-here"    # Claude Haiku
```

No other setup is required — the tool registry (`tools_ready_in_memory`)
is generated in memory per-seed; it does not depend on CostBench's offline
search-database build step.

## Repository structure

```
brts_core.py            The gate (validate_batch / v2 / v3) + cost-substitution
                         + redundancy elimination. Pure logic, no API calls.
run_brts_minimal.py      run_episode(): the LLM-driven harness — safety modes,
                         state caching, per-model call wrappers.
brts/                    Deterministic alternatives (Appendix A.1):
  scored.py                scored-greedy (negative control)
  precomputed.py            precompute-then-batch (cost-optimal)
  length_optimal.py         precompute-length-optimal + Theorem 1's algorithm
  synthetic.py               the no-bundle-tools synthetic domain generator
experiments/
  llm_sweep.py             run_llm_sweep_conditions(): N-condition sweep
                            runner with paired significance tests.
  bfcl_episode.py, bfcl_parallel_episode.py   BFCL parallel/multi_turn harness.
  synthetic_episode.py     Synthetic-domain harness.
  sweeps/                  One script per real reported result (see mapping
                            below). Each incurs real API cost when run.
  noise_checks/            Zero-cost synthetic tests, offline replays, and
                            small real-data verification/debugging scripts
                            from development (methodology transparency —
                            see Appendix A.4 for one documented example).
bfcl/                    BFCL adapter + vendored bfcl-eval subset (see NOTICE.md).
logistics/               PDDL domain/problem files (task_independent[_mixed]).
logistics_llm_harness.py, logistics_precompute.py   PDDL Logistics harness.
env/                     CostBench's harness, vendored (see NOTICE.md).
```

## Script → paper result mapping

| Paper item | Script(s) in `experiments/sweeps/` |
|---|---|
| Table 1 (gated vs. sequential, GPT-4o, n=20, 3 configs) | `gpt4o_20seed_location_transportation.py`, `gpt4o_20seed_location_accommodation.py`, `gpt4o_20seed_location_transportation_dining.py` |
| Table 2 (cost-substitution, 240 episodes, 3 models) | `gpt4o_mini_20seed_cost_substitution_comparison_all_pairings.py`, `gpt4o_20seed_cost_substitution_comparison_all_pairings.py`, `claude_haiku_20seed_cost_substitution_comparison_all_pairings.py` + `noise_checks/offline_replay_cost_substitution_invariants.py` |
| Table 3 (composition: redundancy elim. + cost-sub.) | `gpt4o_mini_5seed_redundancy_elimination_comparison.py`, `gpt4o_mini_5seed_redundancy_plus_cost_substitution_comparison.py` + `noise_checks/offline_replay_redundancy_elimination_invariants.py`, `noise_checks/offline_replay_composed_invariants.py` |
| Table 4 / Section 5.3 (4-condition factorial, gpt-4o-mini, n=50, 2 pairings) | `gpt4o_mini_50seed_location_transportation_safety_baselines_4cond.py`, `gpt4o_mini_50seed_location_accommodation_safety_baselines_4cond.py` (dry runs: `*_1seed_*_dryrun.py`) |
| GPT-4o safety signal, n=50 → n=100 | `gpt4o_50seed_location_transportation_safety_baselines_4cond.py`, `gpt4o_100seed_location_transportation_safety_baselines_4cond.py` |
| Claude Haiku replication, n=50 | `claude_haiku_50seed_location_transportation_safety_baselines_4cond.py` |
| Table 5 / Figure 1 (deterministic alternatives, seeds 1-5) | uses `brts/` directly — see `noise_checks/dryrun_*` for harness smoke tests |
| Table 6 / Figure 2 (gated-batched vs. sequential calls, 3 models) | `*_20seed_location_transportation.py`, `*_20seed_location_accommodation.py`, `*_20seed_location_transportation_dining.py` per model, plus `gpt4o_mini_5seed_location_*.py` |
| Table 7 (BFCL `parallel`/`multi_turn`) | `gpt4o_bfcl_parallel_20tasks.py`, `gpt4o_bfcl_multiturn_5tasks.py`, `claude_haiku_bfcl_parallel_20tasks.py`, `claude_haiku_bfcl_multiturn_5tasks.py` |
| Table 8 / Figure 3 (PDDL Logistics vs. proven optimum) | `gpt4o_logistics_independent_instances.py`, `gpt4o_logistics_mixed_extended_rounds.py`, `claude_haiku_logistics_independent_instances.py` |
| Appendix A.4 (temperature/recoverability fix) | `noise_checks/verify_empty_response_fix_temp0.py`, `noise_checks/llm_determinism_check.py` |

`noise_checks/` also contains the zero-cost verification stages
(`test_*.py`, `scan_*.py`) that preceded each real-money run above —
included for methodology transparency, not because they reproduce a
reported number on their own.

**Cost note.** Every script under `sweeps/` makes real, billed API calls.
Dry-run variants (`*_1seed_*_dryrun.py`) exist for the larger sweeps —
run those first. Per-episode dollar cost is computed and printed inline
using each script's configured per-token pricing.

## Citation

```
CostBench (our primary environment):
@article{liu2025costbench,
  title={CostBench: Evaluating Multi-Turn Cost-Optimal Planning and Adaptation in Dynamic Environments for LLM Tool-Use Agents},
  author={Liu, Jiayu and Qian, Cheng and Su, Zhaochen and Zong, Qing and Huang, Shijue and He, Bingxiang and Fung, Yi R},
  journal={arXiv preprint arXiv:2511.02734},
  year={2025}
}
```

Citation for this paper will be added on acceptance.
