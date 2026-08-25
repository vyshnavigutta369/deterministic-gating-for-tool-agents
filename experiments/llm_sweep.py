"""
Unified free-choice LLM batching-vs-sequential sweep runner.

This single run_llm_sweep() replaces what used to be ~8 separate,
near-duplicate scripts (one per model/category-combo/seed-range
combination): each only differed in task/categories/label/model/seed-range/
pricing, while reimplementing the same batched-vs-sequential loop, the same
per-seed table printing, and the same summary-stats logic. Consolidated here
so entry-point scripts (experiments/sweeps/*.py) are just a few lines each.

Always tracks real OpenAI token usage -> real dollar cost (via
run_episode's total_prompt_tokens/total_completion_tokens, converted using
whatever per-model price is passed in), and always explicitly flags any
goal_reached=False episode rather than silently folding it into an average
-- this was added after gpt-4o-mini's location+transportation run showed a
hallucinated tool name repeated for 27 rounds with no recovery, which
skewed a naive average badly enough that it had to be caught and reported
by hand. Making this the only code path means it can't be missed again.
"""
import os
import sys
import statistics
from scipy import stats as scipy_stats
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, regardless of cwd
from run_brts_minimal import run_episode


def _stats(values):
    return {
        "mean": statistics.mean(values), "min": min(values), "max": max(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _paired_test(a_vals, b_vals):
    """Paired significance (a vs b run on the SAME seed, so a paired test is
    the correct one, not an independent-samples test). Paired t-test assumes
    normal differences; Wilcoxon signed-rank is the distribution-free
    fallback -- returned alongside rather than instead of, since n is often
    small (5-20 seeds) and it's better to show both than silently pick the
    one that looks best. Returns (mean_diff, t_stat, t_p, w_p)."""
    mean_diff = statistics.mean(a - b for a, b in zip(a_vals, b_vals))
    t_stat, t_p = scipy_stats.ttest_rel(a_vals, b_vals)
    try:
        w_stat, w_p = scipy_stats.wilcoxon(a_vals, b_vals)
    except ValueError:
        # all paired differences are exactly zero, or n too small -- not computable
        w_p = float("nan")
    return mean_diff, t_stat, t_p, w_p


def _print_significance(seed_results, has_failures):
    """Paired significance (batched vs sequential) for both LLM calls and
    tool cost. Needs at least 2 seeds to form a paired sample at all."""
    n = len(seed_results)
    if n < 2:
        return
    print(f"\n=== STATISTICAL SIGNIFICANCE (batched vs sequential, paired across {n} seeds) ===")
    if has_failures:
        print("NOTE: this run had goal_reached=False episode(s) -- their (possibly max-rounds-capped) "
              "calls/cost values are still included below, same as the 'ALL EPISODES' stats above. "
              "A failure can pull calls/cost far from what a successful episode would show, which can "
              "swing a p-value in either direction -- interpret accordingly, don't treat it as clean.")
    for metric_label, batch_key, seq_key in [("LLM calls", "batched_calls", "sequential_calls"),
                                              ("tool cost", "batched_cost", "sequential_cost")]:
        batched_vals = [r[batch_key] for r in seed_results]
        seq_vals = [r[seq_key] for r in seed_results]
        mean_diff, t_stat, t_p, w_p = _paired_test(batched_vals, seq_vals)
        print(f"{metric_label}: mean(batched-sequential)={mean_diff:+.2f} | "
              f"paired t-test p={t_p:.4g} (t={t_stat:.3f}) | Wilcoxon signed-rank p={w_p:.4g}")


def run_llm_sweep(task: str, categories, label: str, model: str = "gpt-4o", seeds=range(1, 6),
                   input_price_per_million: float = 2.50, output_price_per_million: float = 10.00):
    """
    Run batched vs sequential episodes for every seed in `seeds`, using
    run_episode from run_brts_minimal.py unchanged. Prints a per-seed table,
    an explicit failures list, real API dollar cost (total / per-episode /
    per-call), and mean/min/max/std stats for calls and tool cost -- computed
    over all episodes, and separately over successful-only episodes whenever
    any failure exists.

    input_price_per_million / output_price_per_million: this model's
    published per-1M-token pricing (default is GPT-4o's $2.50/$10.00; pass
    e.g. 0.15/0.60 for gpt-4o-mini).
    """
    input_price_per_token = input_price_per_million / 1_000_000
    output_price_per_token = output_price_per_million / 1_000_000

    def api_dollar_cost(prompt_tokens: int, completion_tokens: int) -> float:
        return prompt_tokens * input_price_per_token + completion_tokens * output_price_per_token

    is_claude = model.startswith("claude")
    required_env_var = "ANTHROPIC_API_KEY" if is_claude else "OPENAI_API_KEY"
    provider = "Anthropic" if is_claude else "OpenAI"
    print(f"call_llm_stub now makes REAL {provider} API calls (model={model}) and will incur cost.")
    print(f"Requires {required_env_var} set in your environment (your own key -- see "
          f"{'_get_anthropic_client' if is_claude else '_get_client'}).")
    print(f"This run makes TWO full episodes (batched + sequential) PER SEED, "
          f"seeds {seeds.start}-{seeds.stop - 1}, categories={categories}.\n")
    if not os.environ.get(required_env_var):
        print(f"{required_env_var} not set -- aborting before making any real calls.")
        sys.exit(1)

    seed_results = []
    for seed in seeds:
        print(f"=== BATCHED (allow_batch=True), seed={seed} ===")
        batched = run_episode(task=task, categories=categories, seed=seed, allow_batch=True, model=model)
        batched_api_cost = api_dollar_cost(batched["total_prompt_tokens"], batched["total_completion_tokens"])
        print(f"Batched summary: rounds={batched['rounds']}, llm_calls={batched['llm_calls']}, "
              f"total_tool_cost={batched['total_tool_cost']:.2f}, real_api_dollar_cost=${batched_api_cost:.4f}, "
              f"goal_reached={batched['goal_reached']}\n")

        print(f"=== SEQUENTIAL (allow_batch=False), same seed={seed} ===")
        sequential = run_episode(task=task, categories=categories, seed=seed, allow_batch=False, model=model)
        sequential_api_cost = api_dollar_cost(sequential["total_prompt_tokens"], sequential["total_completion_tokens"])
        print(f"Sequential summary: rounds={sequential['rounds']}, llm_calls={sequential['llm_calls']}, "
              f"total_tool_cost={sequential['total_tool_cost']:.2f}, real_api_dollar_cost=${sequential_api_cost:.4f}, "
              f"goal_reached={sequential['goal_reached']}\n")

        seed_results.append({
            "seed": seed,
            "batched_calls": batched["llm_calls"], "batched_cost": batched["total_tool_cost"],
            "batched_api_cost": batched_api_cost, "batched_ok": batched["goal_reached"],
            "sequential_calls": sequential["llm_calls"], "sequential_cost": sequential["total_tool_cost"],
            "sequential_api_cost": sequential_api_cost, "sequential_ok": sequential["goal_reached"],
        })

    print(f"=== SUMMARY (seeds {seeds.start}-{seeds.stop - 1}, {label}, model={model}) ===")
    print(f"{'seed':>4} {'b_calls':>7} {'b_cost':>9} {'b_apicost':>10} {'b_ok':>5} | "
          f"{'s_calls':>7} {'s_cost':>9} {'s_apicost':>10} {'s_ok':>5}")
    for r in seed_results:
        print(f"{r['seed']:>4} {r['batched_calls']:>7} {r['batched_cost']:>9.2f} "
              f"{r['batched_api_cost']:>10.4f} {str(r['batched_ok']):>5} | "
              f"{r['sequential_calls']:>7} {r['sequential_cost']:>9.2f} "
              f"{r['sequential_api_cost']:>10.4f} {str(r['sequential_ok']):>5}")

    # Explicit failure flagging -- never silently averaged away.
    failures = []
    for r in seed_results:
        if not r["batched_ok"]:
            failures.append(f"seed={r['seed']} BATCHED FAILED (goal_reached=False)")
        if not r["sequential_ok"]:
            failures.append(f"seed={r['seed']} SEQUENTIAL FAILED (goal_reached=False)")

    print("\n=== FAILURES ===")
    if failures:
        print(f"{len(failures)} episode(s) did NOT reach the goal:")
        for f in failures:
            print(f"  - {f}")
        print("These are EXCLUDED from the 'successful only' stats below, and called out "
              "separately from the 'all episodes' stats -- do not treat the naive average as "
              "representative of real task performance when failures exist.")
    else:
        print("None -- all episodes reached the goal.")

    total_api_cost = sum(r["batched_api_cost"] + r["sequential_api_cost"] for r in seed_results)
    total_episodes = len(seed_results) * 2
    total_llm_calls = sum(r["batched_calls"] + r["sequential_calls"] for r in seed_results)
    print(f"\n=== REAL API DOLLAR COST ({model}: "
          f"${input_price_per_million:.2f}/1M input, ${output_price_per_million:.2f}/1M output) ===")
    print(f"Total real API dollar cost across all {total_episodes} episodes: ${total_api_cost:.4f}")
    print(f"Cost per episode: ${total_api_cost / total_episodes:.4f}")
    print(f"Cost per LLM call: ${total_api_cost / total_llm_calls:.6f}")

    print("\n=== STATS: ALL EPISODES (includes any failures above) ===")
    for cond_label, calls_key, cost_key in [("batched", "batched_calls", "batched_cost"),
                                             ("sequential", "sequential_calls", "sequential_cost")]:
        calls_stats = _stats([r[calls_key] for r in seed_results])
        cost_stats = _stats([r[cost_key] for r in seed_results])
        print(f"{cond_label}: calls mean={calls_stats['mean']:.2f} min={calls_stats['min']} "
              f"max={calls_stats['max']} std={calls_stats['std']:.2f} | "
              f"cost mean={cost_stats['mean']:.2f} min={cost_stats['min']:.2f} "
              f"max={cost_stats['max']:.2f} std={cost_stats['std']:.2f}")

    if failures:
        print("\n=== STATS: SUCCESSFUL EPISODES ONLY (failures excluded) ===")
        for cond_label, calls_key, cost_key, ok_key in [
            ("batched", "batched_calls", "batched_cost", "batched_ok"),
            ("sequential", "sequential_calls", "sequential_cost", "sequential_ok"),
        ]:
            ok_rows = [r for r in seed_results if r[ok_key]]
            if not ok_rows:
                print(f"{cond_label}: no successful episodes")
                continue
            calls_stats = _stats([r[calls_key] for r in ok_rows])
            cost_stats = _stats([r[cost_key] for r in ok_rows])
            print(f"{cond_label} ({len(ok_rows)}/{len(seed_results)} succeeded): "
                  f"calls mean={calls_stats['mean']:.2f} min={calls_stats['min']} "
                  f"max={calls_stats['max']} std={calls_stats['std']:.2f} | "
                  f"cost mean={cost_stats['mean']:.2f} min={cost_stats['min']:.2f} "
                  f"max={cost_stats['max']:.2f} std={cost_stats['std']:.2f}")

    _print_significance(seed_results, has_failures=bool(failures))

    return seed_results


def run_llm_sweep_conditions(task: str, categories, label: str, conditions: dict, reference: str,
                              model: str = "gpt-4o", seeds=range(1, 6),
                              input_price_per_million: float = 2.50, output_price_per_million: float = 10.00):
    """
    Generalization of run_llm_sweep to an arbitrary set of NAMED conditions
    instead of exactly batched/sequential -- built for comparing the real
    safety-gated batching condition against baselines that skip validate_batch
    as an execution gate (run_episode's safety_mode="blind"/"self_judged",
    mimicking prior work -- W&D and MCP-Bench respectively). Reuses
    run_episode unchanged, the same per-seed table / failure-flagging /
    cost-accounting structure as run_llm_sweep, and _paired_test for
    significance -- computed for each non-reference condition against
    `reference`, since that's the comparison that actually answers "does
    skipping the safety gate change calls/cost, and at what real safety
    cost" rather than an all-pairs comparison nobody asked for.

    `conditions`: {condition_name: kwargs passed straight to run_episode
    (e.g. allow_batch, safety_mode)}.
    `reference`: key into `conditions` to treat as the baseline other
    conditions are compared against (e.g. "gated").
    """
    input_price_per_token = input_price_per_million / 1_000_000
    output_price_per_token = output_price_per_million / 1_000_000

    def api_dollar_cost(prompt_tokens: int, completion_tokens: int) -> float:
        return prompt_tokens * input_price_per_token + completion_tokens * output_price_per_token

    is_claude = model.startswith("claude")
    required_env_var = "ANTHROPIC_API_KEY" if is_claude else "OPENAI_API_KEY"
    provider = "Anthropic" if is_claude else "OpenAI"
    print(f"call_llm_stub now makes REAL {provider} API calls (model={model}) and will incur cost.")
    print(f"This run makes {len(conditions)} full episodes PER SEED, seeds {seeds.start}-{seeds.stop - 1}, "
          f"conditions={list(conditions)}, categories={categories}.\n")
    if not os.environ.get(required_env_var):
        print(f"{required_env_var} not set -- aborting before making any real calls.")
        sys.exit(1)
    assert reference in conditions, f"reference={reference!r} must be a key in conditions={list(conditions)}"

    seed_results = []
    for seed in seeds:
        row = {"seed": seed}
        for name, kwargs in conditions.items():
            print(f"=== {name.upper()}, seed={seed} ===")
            r = run_episode(task=task, categories=categories, seed=seed, model=model, **kwargs)
            cost = api_dollar_cost(r["total_prompt_tokens"], r["total_completion_tokens"])
            print(f"{name} summary: rounds={r['rounds']}, llm_calls={r['llm_calls']}, "
                  f"total_tool_cost={r['total_tool_cost']:.2f}, real_api_dollar_cost=${cost:.4f}, "
                  f"goal_reached={r['goal_reached']}, would_have_been_rejected="
                  f"{r['total_would_have_been_rejected']}/{r['total_executed_calls']} executed calls "
                  f"({r['unsafe_execution_rate']:.1%}), v2_extra_accepted={r['total_v2_extra_accepted']}, "
                  f"total_rejected={r['total_rejected_calls']}\n")
            row[f"{name}_calls"] = r["llm_calls"]
            row[f"{name}_cost"] = r["total_tool_cost"]
            row[f"{name}_api_cost"] = cost
            row[f"{name}_ok"] = r["goal_reached"]
            row[f"{name}_unsafe_count"] = r["total_would_have_been_rejected"]
            row[f"{name}_executed_count"] = r["total_executed_calls"]
            row[f"{name}_v2_extra_accepted"] = r["total_v2_extra_accepted"]
            row[f"{name}_rejected_count"] = r["total_rejected_calls"]
            row[f"{name}_cost_substitutions"] = r["total_cost_substitutions"]
            row[f"{name}_cost_saved_by_substitution"] = r["total_cost_saved_by_substitution"]
            row[f"{name}_rounds"] = r["rounds"]
            row[f"{name}_redundant_dropped"] = r["total_redundant_calls_dropped"]
            row[f"{name}_cost_saved_by_redundancy_elimination"] = r["total_cost_saved_by_redundancy_elimination"]
        seed_results.append(row)

    print(f"=== SUMMARY (seeds {seeds.start}-{seeds.stop - 1}, {label}, model={model}) ===")
    header = f"{'seed':>4}"
    for name in conditions:
        header += f" {name + '_calls':>16} {name + '_cost':>14} {name + '_ok':>8} {name + '_unsafe':>12}"
    print(header)
    for row in seed_results:
        line = f"{row['seed']:>4}"
        for name in conditions:
            line += (f" {row[f'{name}_calls']:>16} {row[f'{name}_cost']:>14.2f} "
                      f"{str(row[f'{name}_ok']):>8} {row[f'{name}_unsafe_count']:>12}")
        print(line)

    # Explicit failure flagging -- never silently averaged away.
    failures = []
    for row in seed_results:
        for name in conditions:
            if not row[f"{name}_ok"]:
                failures.append(f"seed={row['seed']} {name.upper()} FAILED (goal_reached=False)")
    print("\n=== FAILURES ===")
    if failures:
        print(f"{len(failures)} episode(s) did NOT reach the goal:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("None -- all episodes reached the goal.")

    total_api_cost = sum(row[f"{name}_api_cost"] for row in seed_results for name in conditions)
    total_episodes = len(seed_results) * len(conditions)
    total_llm_calls = sum(row[f"{name}_calls"] for row in seed_results for name in conditions)
    print(f"\n=== REAL API DOLLAR COST ({model}: "
          f"${input_price_per_million:.2f}/1M input, ${output_price_per_million:.2f}/1M output) ===")
    print(f"Total real API dollar cost across all {total_episodes} episodes: ${total_api_cost:.4f}")
    print(f"Cost per episode: ${total_api_cost / total_episodes:.4f}")
    print(f"Cost per LLM call: ${total_api_cost / total_llm_calls:.6f}")

    print("\n=== STATS PER CONDITION (calls / cost / real unsafe-execution rate) ===")
    for name in conditions:
        calls_stats = _stats([row[f"{name}_calls"] for row in seed_results])
        cost_stats = _stats([row[f"{name}_cost"] for row in seed_results])
        total_unsafe = sum(row[f"{name}_unsafe_count"] for row in seed_results)
        total_executed = sum(row[f"{name}_executed_count"] for row in seed_results)
        unsafe_rate = total_unsafe / total_executed if total_executed else 0.0
        total_v2_extra = sum(row[f"{name}_v2_extra_accepted"] for row in seed_results)
        total_rejected = sum(row[f"{name}_rejected_count"] for row in seed_results)
        total_cost_subs = sum(row[f"{name}_cost_substitutions"] for row in seed_results)
        total_cost_saved = sum(row[f"{name}_cost_saved_by_substitution"] for row in seed_results)
        total_redundant_dropped = sum(row[f"{name}_redundant_dropped"] for row in seed_results)
        total_redundancy_saved = sum(row[f"{name}_cost_saved_by_redundancy_elimination"] for row in seed_results)
        v2_extra_str = f" | V2 EXTRA-ACCEPTED (beyond v1): {total_v2_extra} calls" if total_v2_extra else ""
        cost_sub_str = f" | COST-SUBSTITUTIONS: {total_cost_subs} calls, ${total_cost_saved:.2f} saved" \
            if total_cost_subs else ""
        redundancy_str = f" | REDUNDANCY DROPPED: {total_redundant_dropped} calls, " \
                          f"${total_redundancy_saved:.2f} saved" if total_redundant_dropped else ""
        print(f"{name}: calls mean={calls_stats['mean']:.2f} min={calls_stats['min']} "
              f"max={calls_stats['max']} std={calls_stats['std']:.2f} | "
              f"cost mean={cost_stats['mean']:.2f} min={cost_stats['min']:.2f} "
              f"max={cost_stats['max']:.2f} std={cost_stats['std']:.2f} | "
              f"UNSAFE EXECUTION RATE: {total_unsafe}/{total_executed} executed calls ({unsafe_rate:.1%}) "
              f"would have been rejected by the real safety gate | TOTAL REJECTED BY THIS GATE: "
              f"{total_rejected} calls{v2_extra_str}{cost_sub_str}{redundancy_str}")

    print(f"\n=== PAIRED SIGNIFICANCE vs reference condition '{reference}' (across {len(seed_results)} seeds) ===")
    for name in conditions:
        if name == reference:
            continue
        for metric_label, suffix in [("LLM calls", "_calls"), ("tool cost", "_cost")]:
            a_vals = [row[f"{name}{suffix}"] for row in seed_results]
            b_vals = [row[f"{reference}{suffix}"] for row in seed_results]
            mean_diff, t_stat, t_p, w_p = _paired_test(a_vals, b_vals)
            print(f"{name} vs {reference} -- {metric_label}: mean({name}-{reference})={mean_diff:+.2f} | "
                  f"paired t-test p={t_p:.4g} (t={t_stat:.3f}) | Wilcoxon signed-rank p={w_p:.4g}")

    return seed_results
