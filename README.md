# Deterministic vs. Model-Based Graders for Tool-Using Agents

*How Much Exposure a Safety Comparison Needs* — accepted as a poster at the NeurIPS 2026 Workshop on Evaluating Interactive Agents (IAEval).

Code accompanying the paper: the gate, the two post-gate cost refinements, the four-condition harness, and the scripts behind every reported table.

---

## What this paper studies

An agent that uses tools decides on a call and then executes it. In most pipelines nothing intervenes between the two, and a call that runs before the inputs it depends on exist is not a wrong answer but a wrong action, already taken. Something has to decide, before execution, whether to let a batch run. We call whatever plays that role a **grader**.

We propose the **gate**: a deterministic check that a call's declared inputs are already in the state when the round begins. We compare it against the natural alternative — asking the model to vet its own batch — with unchecked execution as a control.

**The headline is not a measured safety difference.** Unsafe executions are rare, about one call in 1,100, and separating two graders at that rate takes thousands of calls per condition — more than this study or comparable ones ran. What separates the designs is what each guarantees and what each costs.

| | |
|---|---|
| Unsafe executions, ungated conditions | **7 / 7,675** executed calls (0.091%, ≈1 in 1,100) |
| Unsafe executions, gated conditions | **0 / 7,350** (3,961 gated + 3,389 gated+self-assessment) |
| Total executed calls | **15,025** |

Three contributions:

1. **A sizing rule for rare-event grader comparison.** With `r` the true unsafe-execution rate as a fraction of executed calls: ruling out a rate above `r` after zero events costs ≈ `3/r` calls; seeing at least one event with probability 0.8 costs ≈ `1.6/r`; telling `r` from `4r` at 80% power costs ≈ `4.5/r` per condition, `9/r` across two. At `r` = 0.1% that is roughly 3,000, 1,600 and 4,500. Our own largest arm reached 1,384 calls — about 20% power, short by a factor of 3.6 (Section 5.4).
2. **A composition result for verified pipelines.** After the gate approves a batch, two schema-preserving refinements — redundancy elimination and cost substitution — can neither add a round nor raise cost, proven and confirmed across 180 replayed episodes and three models (Section 5.2).
3. **A characterization of when grading is unnecessary.** Where a domain declares enough structure, the batch can be computed in advance rather than proposed. On CostBench the precomputed length-optimal schedule finishes in **2.00 rounds** everywhere, against 7.40 / 3.60 / 4.20 for gated batching, issuing **no LLM calls** (Section 5.5).

This repository is a **companion for reproducing those results**, not a general-purpose tool-batching framework. Each script maps to a specific table in the paper.

---

## Relationship to CostBench

[CostBench](https://github.com/JiayuJeff/CostBench) (Liu et al., ACL 2026) is our primary environment. `env/` is CostBench's harness, vendored unmodified — we use its tool registry, ground-truth solver and travel-planning domain as-is, modified only in that we allow batching at all. Our contribution sits on top:

- `brts_core.py`, `run_brts_minimal.py` — the gate, the two post-gate cost mechanisms, and the four-condition harness (`gated` / `blind` / `self_judged` / `gated_self_assessed`).
- `brts/` — the deterministic, grader-free alternatives of Section 5.5 and Appendix A.4 (scored-greedy negative control, precompute-then-batch, precompute-length-optimal) and the theory behind when they are optimal.
- `experiments/` — the sweep runner and every reproduction script.
- `bfcl/`, `logistics/` — the secondary domains of Section 4 and Appendix A.5.3: BFCL `parallel` / `multi_turn`, and PDDL Logistics via `pyperplan`.

See `NOTICE.md` for the two vendored third-party components and their upstream licenses.

---

## Setup

```bash
pip install -r requirements.txt

export OPENAI_API_KEY="your-key-here"       # GPT-4o, gpt-4o-mini
export ANTHROPIC_API_KEY="your-key-here"    # Claude Haiku 4.5
```

Targets Python 3.9. No other setup is required — the tool registry (`tools_ready_in_memory`) is generated in memory per seed and does not depend on CostBench's offline search-database build step.

All experiments are API-based; no local GPU training is involved.

---

## Repository structure

```
brts_core.py            The gate + cost substitution + redundancy elimination.
                         Pure logic, no API calls.
run_brts_minimal.py      run_episode(): the LLM-driven harness — the four
                         conditions, state caching, per-provider call wrappers.
brts/                    Deterministic alternatives (Section 5.5, Appendix A.4):
  scored.py                scored-greedy (deliberate negative control)
  precomputed.py           precompute-then-batch (cost-optimal)
  length_optimal.py        precompute-length-optimal + Theorem 1's algorithm
  synthetic.py             the no-bundle-tools synthetic domain generator
experiments/
  llm_sweep.py             run_llm_sweep_conditions(): N-condition sweep runner
                            with paired significance tests
  bfcl_episode.py, bfcl_parallel_episode.py   BFCL multi_turn / parallel harness
  synthetic_episode.py     synthetic-domain harness
  sweeps/                  one script per reported result (mapping below).
                            Each incurs real API cost when run.
  noise_checks/            zero-cost synthetic tests, offline replays, and the
                            small verification scripts from development
bfcl/                    BFCL adapter + vendored bfcl-eval subset (NOTICE.md)
logistics/               PDDL domain and problem files
logistics_llm_harness.py, logistics_precompute.py   PDDL Logistics harness
env/                     CostBench's harness, vendored (NOTICE.md)
```

---

## The four conditions

Each grader-comparison trial runs the same four conditions, differing only in what checks a proposed batch before it executes:

| Condition | What checks the batch |
|---|---|
| `gated` | the deterministic check runs and rejects unsafe calls — the reference |
| `blind` | nothing checks; the agent's proposal executes as-is |
| `self_judged` | the model is told no external system will check, and must judge its own calls |
| `gated_self_assessed` | the model self-assesses *and* the deterministic check runs |

In the ungated conditions the gate still runs silently so its would-be rejections can be counted. That count is the safety comparison. Self-judgment adds no second model call — every condition issues one LLM call per round, so any increase comes from extra rounds.

## Metrics

- **LLM calls** (primary) — prompts sent to the model, one per round regardless of how many tool calls that round executes.
- **Tool cost** (secondary) — CostBench's internal cost of tools executed. Dollar-prefixed because the benchmark writes it that way; it is **not** real API expenditure, which we do not report.
- **Goal-completion rate.**
- **Unsafe executions** — executed calls the gate would have rejected.

---

## Script → paper table mapping

| Paper item | Script(s) |
|---|---|
| **Table 2** — gated batching vs. sequential, GPT-4o, n = 20, three configurations | `sweeps/gpt4o_20seed_location_transportation.py`, `sweeps/gpt4o_20seed_location_accommodation.py`, `sweeps/gpt4o_20seed_location_transportation_dining.py` |
| **Table 3** — unsafe executions by condition, every Section 5.3 trial | `sweeps/gpt4o_mini_{5,20,50}seed_*_safety_baselines*.py`, `sweeps/gpt4o_{50,100}seed_location_transportation_safety_baselines_4cond.py`, `sweeps/claude_haiku_50seed_location_transportation_safety_baselines_4cond.py` (dry runs: `sweeps/*_1seed_*_dryrun.py`) |
| **Table 4 / Figure 1** — exposure requirements and the power curve | `<FILL: the Fisher's-exact simulation over 10,000 replicates is not currently in the repository — see note below>` |
| **Table 5** — mean rounds per episode, deterministic variants, seeds 1–5 | uses `brts/` directly (`scored.py`, `precomputed.py`, `length_optimal.py`); harness smoke tests in `noise_checks/dryrun_*` |
| **Table 6** — mean tool cost for the deterministic variants of Table 5 | same runs as Table 5 |
| **Table 7** — per-trial cost means behind Section 5.3 | the same safety-baseline sweeps as Table 3 |
| **Table 8** — cost substitution replayed offline, 180 episodes | `sweeps/gpt4o_mini_20seed_cost_substitution_comparison_all_pairings.py`, `sweeps/gpt4o_20seed_cost_substitution_comparison_all_pairings.py`, `sweeps/claude_haiku_20seed_cost_substitution_comparison_all_pairings.py`, plus `noise_checks/offline_replay_cost_substitution_invariants.py` |
| Redundancy elimination and the composition result (Section 5.2, Appendix A.2) | `sweeps/gpt4o_mini_5seed_redundancy_elimination_comparison.py`, `sweeps/gpt4o_mini_5seed_redundancy_plus_cost_substitution_comparison.py`, `noise_checks/offline_replay_redundancy_elimination_invariants.py`, `noise_checks/offline_replay_composed_invariants.py` |
| **Table 9** — mean LLM calls per episode, gated-batched vs. sequential | the three `*_20seed_location_*` scripts per model, plus `sweeps/gpt4o_mini_5seed_location_*.py` |
| **Table 10** — BFCL by category and mechanism | `sweeps/gpt4o_bfcl_parallel_20tasks.py`, `sweeps/gpt4o_bfcl_multiturn_5tasks.py`, `sweeps/claude_haiku_bfcl_parallel_20tasks.py`, `sweeps/claude_haiku_bfcl_multiturn_5tasks.py` |
| **Table 11** — batched round counts against the `pyperplan` optimum | `sweeps/gpt4o_logistics_independent_instances.py`, `sweeps/gpt4o_logistics_mixed_extended_rounds.py`, `sweeps/claude_haiku_logistics_independent_instances.py` |
| Synthetic no-bundle-tool domain (Section 4, Appendix A.5.3) | `sweeps/gpt4o_synthetic_5seed.py`, `sweeps/claude_haiku_synthetic_5seed.py`, `brts/synthetic.py` |
| Appendix A.8 — temperature and the empty-response failure mode | `noise_checks/verify_empty_response_fix_temp0.py`, `noise_checks/llm_determinism_check.py` |

`noise_checks/` also holds the zero-cost verification stages (`test_*.py`, `scan_*.py`) that preceded each billed run — included for methodology transparency, not because they reproduce a reported number on their own.

**Cost note.** Every script under `sweeps/` makes real, billed API calls. Dry-run variants (`*_1seed_*_dryrun.py`) exist for the larger sweeps; run those first. Per-episode cost is computed and printed inline from each script's configured per-token pricing.

---

## Statistics

Conditions with zero unsafe executions are reported with an exact one-sided Clopper–Pearson 95% upper bound; where events occurred, the two-sided exact 95% interval. The original GPT-4o n = 20 sweep retained only summary statistics, so those use unpaired Welch's t-tests; later runs kept per-case data and report the paired t-test with a Wilcoxon signed-rank robustness check, both in the released result files. **No multiple-comparison correction is applied** — the cost differences have small uncorrected p-values, but the borderline call-count comparisons should not be read as independently confirmed.

---

## Caveats carried from the paper

- **Model breadth and uneven exposure.** Three models, unequally: 3,772 ungated calls on GPT-4o, 2,999 on gpt-4o-mini, 904 on Claude Haiku. The two nulls rest on exposures too small to establish a zero rate.
- **A quantified power limit.** The gpt-4o-mini bounds at n = 50 are about the size of the rate first observed, so they cannot tell zero risk from that rate.
- **The reference solver is also used by the methods it evaluates.** CostBench's ground-truth solver defines our cost reference *and* computes the precompute sequences, favouring those variants by construction. Theorem 1 is solver-independent, PDDL Logistics uses third-party `pyperplan`, and the grader comparison never invokes the solver.
- **What an unsafe execution stands for.** CostBench tracks output types, not values, so we measure the dependency violation rather than the downstream harm it would cause in a system with real side effects.
- **A narrow grader.** The gate is a precondition check. It asks whether a call's inputs are ready, not whether the task was done well.

## Implementation notes

- **PDDL Logistics** needed two fixes before its results could be trusted: `pyperplan`'s operator names include literal parentheses that model proposals initially omitted, causing complete rejection across an entire domain; and the model cycled between load and unload with no memory of having visited a state. Both are fixed here (Appendix A.7).
- **Temperature.** OpenAI calls run at temperature 0. Claude Haiku's main runs used Anthropic's default; pinning it to 0 *raised* the failure rate, because at temperature 0 an empty-response round leaves the next prompt byte-for-byte identical, so the same wrong conclusion repeats with no mechanism to escape. Corrective feedback for empty-response rounds fixes this (Appendix A.8). Those numbers come from the recoverability sweep and are not comparable to Table 9.

---

## Citation

```bibtex
@inproceedings{gutta2026deterministic,
  title     = {Deterministic vs. Model-Based Graders for Tool-Using Agents:
               How Much Exposure a Safety Comparison Needs},
  author    = {Gutta, Vyshnavi},
  booktitle = {NeurIPS 2026 Workshop on Evaluating Interactive Agents (IAEval)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=KVwuJRAd1u}
}
```

Primary environment:

```bibtex
@article{liu2025costbench,
  title={CostBench: Evaluating Multi-Turn Cost-Optimal Planning and Adaptation in Dynamic Environments for LLM Tool-Use Agents},
  author={Liu, Jiayu and Qian, Cheng and Su, Zhaochen and Zong, Qing and Huang, Shijue and He, Bingxiang and Fung, Yi R},
  journal={arXiv preprint arXiv:2511.02734},
  year={2025}
}
```

## License

MIT (see `LICENSE`), applying to the code in this repository **excluding** `env/` and `bfcl/vendor/`, which are third-party components vendored unmodified. CostBench, BFCL and PDDL Logistics / `pyperplan` remain under their own licenses and are not relicensed here; see `NOTICE.md`.
