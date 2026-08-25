"""
Minimal, faithful BRTS harness for CostBench -- built standalone rather than
patching env/run.py in place, because that file's single-tool-call code path
is deeply intertwined with ~100+ lines of blocking-event logic (ban_tool,
cost_change, seeded reproducibility) that all assume exactly one tool_call.
Editing it safely is real, separate work; this harness avoids that risk by
using CostBench's real Tool/GroundtruthSolver objects and cost model directly,
scoped to the static (no-blocker) case, which is what Section 4.1 of the
design doc says to test first anyway.

HONEST REQUIREMENT: this needs a real LLM API key to actually run end-to-end.
None is available in this sandbox (network restricted, and it should be your
own personal key, not a shared/company one -- see the earlier conversation
about why). Everything up to the actual API call is real and has been run;
the API call itself has NOT been executed anywhere in this project yet.
"""
import os
import sys
import json
import time
from typing import Dict, List, FrozenSet, Set

from openai import OpenAI, RateLimitError, APIConnectionError
from anthropic import Anthropic, RateLimitError as AnthropicRateLimitError, APIConnectionError as AnthropicAPIConnectionError

sys.path.insert(0, '.')
from env.domains.travel.tool_registry import tools_ready_in_memory
from env.utils.solver import GroundtruthSolver
from brts_core import (validate_batch, validate_batch_v2, validate_batch_v3, apply_batch, apply_batch_blind,
                        apply_cost_substitution, apply_redundancy_elimination, get_currently_executable,
                        is_goal_reached)

_client = None
_anthropic_client = None


def _clean_api_key(raw: str) -> str:
    """API keys are contiguous tokens with no legitimate whitespace -- a
    terminal that line-wraps a long pasted key can splice in a literal
    newline/spaces at the wrap point, which then reaches the SDK as an
    invalid HTTP header value. Any whitespace found is corruption, not
    signal, so strip all of it rather than just leading/trailing."""
    return "".join(raw.split())


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable not set. "
                "Run: export OPENAI_API_KEY='your-key-here' in your terminal first."
            )
        _client = OpenAI(api_key=_clean_api_key(api_key))
    return _client


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Run: export ANTHROPIC_API_KEY='your-key-here' in your terminal first."
            )
        _anthropic_client = Anthropic(api_key=_clean_api_key(api_key))
    return _anthropic_client


def build_tool_description_block(tool_names: List[str], tools: Dict) -> str:
    lines = []
    for name in sorted(tool_names):
        t = tools[name]
        lines.append(f"- {name}: cost={t.cost}, inputs={list(t.input_types)}, output={t.output_type}")
    return "\n".join(lines)


def build_empty_response_feedback(visible_tools: List[str]) -> str:
    """Corrective feedback for when a round proposed zero tool calls even
    though at least one was genuinely callable -- mirrors the cycle-detection
    feedback in logistics_llm_harness.py (same failure family: a model that
    gets zero information about why/that its last proposal stalled just
    repeats it, and at temperature=0 that repetition is deterministic and
    therefore permanent for the rest of the episode)."""
    return (
        "WARNING: last round you proposed zero tool calls (an empty "
        "\"tool_calls\": [] list), but the goal has not yet been reached and "
        "no progress was made. The currently available information listed "
        "below genuinely DOES satisfy the inputs of at least one callable "
        f"tool right now: {sorted(visible_tools)}. Do not repeat an empty "
        "response -- reconsider and propose at least one of the tools "
        "listed below as 'Currently callable tools'."
    )


def build_full_rejection_feedback(rejected_names: List[str], state: FrozenSet[str], tools: Dict,
                                   visible_tools: List[str]) -> str:
    """Corrective feedback for when a round proposed at least one tool call
    but EVERY proposed tool was rejected (zero execution, no state change)
    -- the other real failure mode found in this project's own logs (see
    the CostBench-family log analysis: 37 real stuck episodes, 20 of which
    never recovered, all traced to this exact gap -- with no feedback, the
    next round's prompt was byte-identical to this one, and at
    temperature=0 that reliably reproduced the exact same wrong proposal
    forever). Symmetric to build_empty_response_feedback above, and reuses
    the same missing-input reasoning style validate_batch_v2's diagnostics
    already use for gate v2's v2_extra_reasoning.

    Two distinct reasons a name can be rejected, explained differently,
    since "which input is missing" doesn't apply to a tool that was never
    real in the first place:
    - It's a REAL tool, but its inputs aren't satisfied by the current
      state yet -- name exactly which input(s) are missing.
    - It's not a real tool at all (a hallucinated/nonexistent name -- the
      real, repeatedly-observed Location_Finish_from_Step3 case this was
      built to fix) -- say so explicitly and list the real available tool
      names, since there is no missing precondition to describe."""
    lines = []
    for name in rejected_names:
        tool = tools.get(name)
        if tool is None:
            lines.append(f"  - {name!r} is NOT a real tool -- no tool with that exact name exists.")
        else:
            missing = sorted(set(tool.input_types) - state)
            lines.append(f"  - {name!r} is a real tool, but its input(s) {missing} are not yet available.")
    return (
        "WARNING: last round you proposed tool call(s), but EVERY one was REJECTED -- "
        "zero tools executed, no progress was made:\n" + "\n".join(lines) +
        f"\nThe real, currently callable tools are: {sorted(visible_tools)}. "
        "Do not repeat the exact same rejected proposal -- if a name doesn't exist, use one of "
        "the real tool names listed above instead (do not guess a similar-sounding variant); if "
        "a real tool's input is missing, propose a different tool whose inputs ARE already "
        "satisfied, or propose the tool that actually produces the missing input first."
    )


def build_universal_rejection_feedback(rejected_structured: List[dict], valid_names: List[str],
                                        visible_tools: List[str]) -> str:
    """Corrective feedback for gate v3 (validate_batch_v3): fires whenever
    a round had ANY rejection at all -- partial or full -- not only the
    full-rejection case build_full_rejection_feedback handles for v1/v2.
    Partial rejections previously got NO feedback under any prior gate
    version, even after the full-rejection fix: e.g. the real, repeated
    replay case (gpt-4o-mini, location+transportation, seed=1) proposes
    ['Location_Finish_from_Step3', 'Transportation_Refine_to_Step2'] at
    round 3 -- only the first is rejected, the second executes -- so
    build_full_rejection_feedback's `not valid` check never fires there at
    all, and the model gets no signal until the FULLY-rejected round 4.
    v3's reasoning is generated directly by the gate itself (see
    validate_batch_v3's docstring for why that's a structural difference,
    not just an earlier trigger point), so this function only needs to
    format it, not re-derive it.

    `rejected_structured` is exactly what validate_batch_v3 returns as its
    second element -- a list of {"name", "reason", "missing_inputs"} dicts."""
    lines = []
    for r in rejected_structured:
        if r["reason"] == "nonexistent_tool":
            lines.append(f"  - {r['name']!r} is NOT a real tool -- no tool with that exact name exists.")
        else:
            lines.append(f"  - {r['name']!r} is a real tool, but its input(s) {r['missing_inputs']} "
                          f"are not yet available.")
    progress_note = (
        f"The following DID execute successfully this round: {valid_names}.\n"
        if valid_names else
        "None of your proposed tools executed this round -- zero progress was made.\n"
    )
    return (
        "NOTE: some of your proposed tool call(s) this round were REJECTED:\n" + "\n".join(lines) + "\n"
        + progress_note +
        f"The real, currently callable tools are: {sorted(visible_tools)}. "
        "Do not repeat a rejected proposal -- if a name doesn't exist, use one of the real tool "
        "names listed above instead (do not guess a similar-sounding variant); if a real tool's "
        "input is missing, propose the tool that actually produces that input first, or propose a "
        "different tool whose inputs are already satisfied."
    )


SELF_JUDGED_BATCH_RULE = (
    "You MAY propose multiple tool calls this round. There is NO external system checking "
    "whether your batch is safe to execute together -- you are solely responsible for judging "
    "this yourself. Before finalizing your answer, self-assess each candidate tool: would it "
    "still produce a correct, valid result if executed in this same batch, using ONLY what is "
    "listed as currently available below (i.e. it does NOT depend on any other proposed tool's "
    "output this round)? Include a tool in your batch ONLY if you are genuinely confident it "
    "passes this test -- do not guess or include a tool just because it seems related."
)

CHAIN_AWARE_BATCH_RULE = (
    "You MAY propose multiple tool calls this round, INCLUDING a tool that depends on another "
    "tool's output from THIS SAME batch -- you do NOT need to wait for a separate round to "
    "propose a dependent step. The system will automatically resolve the correct execution "
    "order for you (running each prerequisite before whatever depends on it), so propose the "
    "full set of tools needed to make real progress this round, even if some of them depend on "
    "others you are also proposing right now."
)


def build_prompt(state: FrozenSet[str], visible_tools: List[str], tools: Dict, task: str, allow_batch: bool,
                  allow_batch_cost_aware: bool = False, last_round_feedback: str = None,
                  self_judged: bool = False, chain_aware: bool = False) -> str:
    tool_block = build_tool_description_block(visible_tools, tools)
    if self_judged:
        # Mimics MCP-Bench's approach: the model is asked to judge batch
        # safety itself, with no external validate_batch gate -- see
        # run_episode's safety_mode="self_judged".
        batch_rule = SELF_JUDGED_BATCH_RULE
    elif chain_aware:
        # Explicitly PERMITS (encourages, even) proposing a same-round
        # dependency chain, unlike the default batch_rule below which
        # explicitly forbids it -- see run_episode's chain_aware_prompt.
        batch_rule = CHAIN_AWARE_BATCH_RULE
    else:
        batch_rule = (
            "You MAY propose multiple tool calls this round, but ONLY tools whose required inputs "
            "are already satisfied by what is listed as currently available below -- never a tool "
            "that needs another proposed tool's output."
            if allow_batch else
            "You may propose exactly ONE tool call this round."
        )
        if allow_batch and allow_batch_cost_aware:
            batch_rule += (
                "\nWhen batching, still prefer the combination of tools with the lowest total cost "
                "that accomplishes the same progress -- do not default to bundled/composite tools "
                "just because they reduce the number of calls."
            )
    feedback_block = f"\n{last_round_feedback}\n" if last_round_feedback else ""
    return f"""You are planning a {task} task. Minimize total cost.
Currently available information: {sorted(state)}
Currently callable tools:
{tool_block}
{feedback_block}
{batch_rule}
Respond ONLY with JSON: {{"tool_calls": ["tool_name_1", "tool_name_2", ...]}}
"""


def call_llm_stub(prompt: str, model: str = "gpt-4o", max_retries: int = 8):
    """Real implementation -- makes an actual OpenAI API call. Requires
    OPENAI_API_KEY to be set in your environment (see _get_client above).
    Retries with backoff on rate limits, since a multi-seed sweep can exceed
    a lower-tier account's tokens-per-minute cap well before the sweep ends.

    Returns (tool_calls, usage) where usage is {"prompt_tokens": int,
    "completion_tokens": int} from the real API response's usage field --
    this is what the response already reports, just not previously captured."""
    client = _get_client()
    response = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            break
        except (RateLimitError, APIConnectionError) as e:
            # Symmetric fix to call_llm_stub_claude's -- a transient connection
            # error is exactly as retryable as a rate limit, and only rate
            # limits were covered here before.
            if attempt == max_retries - 1:
                raise
            wait = min(5 * (2 ** attempt), 60)
            reason = "rate limited" if isinstance(e, RateLimitError) else "connection error"
            print(f"  {reason}, waiting {wait}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(wait)
    content = response.choices[0].message.content
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }
    try:
        parsed = json.loads(content)
        return parsed.get("tool_calls", []), usage
    except (json.JSONDecodeError, AttributeError) as e:
        raise ValueError(f"Model did not return valid JSON with a 'tool_calls' field. Got: {content!r}") from e


def _extract_json_object(text: str) -> dict:
    """Claude has no native JSON-mode flag (unlike response_format={"type":
    "json_object"} on the OpenAI call above) -- it's told via the prompt
    alone to return raw JSON, but in practice still sometimes wraps its
    answer in a ```json ... ``` fence and/or appends a prose explanation
    after the JSON block despite being told not to (observed for real:
    claude-haiku-4-5 returned a fenced tool_calls object followed by several
    paragraphs justifying the choice). json.loads chokes on either the
    leading fence or the trailing prose, so instead find the first '{' and
    use JSONDecoder.raw_decode, which parses one JSON value and simply
    stops there -- ignoring anything before the '{' or after the object's
    matching '}' -- regardless of what surrounds it."""
    start = text.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found in response", text, 0)
    obj, _ = json.JSONDecoder().raw_decode(text, start)
    return obj


def call_llm_stub_claude(prompt: str, model: str = "claude-haiku-4-5-20251001", max_retries: int = 8):
    """Claude-compatible variant of call_llm_stub, using the real Anthropic
    API (client.messages.create) instead of OpenAI's. Requires
    ANTHROPIC_API_KEY to be set in your environment (see _get_anthropic_client
    above). temperature=0 to match the OpenAI path exactly. Anthropic's API
    has no equivalent of OpenAI's response_format JSON mode, so the same
    build_prompt JSON instruction is relied on alone, with _extract_json_object
    handling whatever surrounds it (fences, trailing prose).

    Returns (tool_calls, usage) in the same shape as call_llm_stub, so
    run_episode can treat both models identically."""
    client = _get_anthropic_client()
    response = None
    for attempt in range(max_retries):
        try:
            # max_tokens=4096 (was 1000): a real n=20 CostBench run showed a
            # large-tool-count round where Claude's chain-of-thought (it
            # enumerates every candidate tool's cost/inputs before deciding)
            # ran past 1000 tokens and got cut off before ever reaching the
            # {"tool_calls": [...]} JSON -- no amount of post-hoc parsing can
            # recover a response that was truncated before the JSON existed,
            # so the fix has to be a larger budget, not smarter extraction.
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except (AnthropicRateLimitError, AnthropicAPIConnectionError) as e:
            # A real n=20 x 3-pairing sweep crashed the ENTIRE run on a transient
            # DNS/connection blip ("nodename nor servname provided, or not
            # known") -- only rate limits were retried before, so one flaky
            # network moment lost all progress on a 180-episode paid run.
            # Connection errors are exactly the kind of transient failure
            # retrying is for; treat them the same as rate limits.
            if attempt == max_retries - 1:
                raise
            wait = min(5 * (2 ** attempt), 60)
            reason = "rate limited" if isinstance(e, AnthropicRateLimitError) else "connection error"
            print(f"  {reason}, waiting {wait}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(wait)
    content = response.content[0].text
    usage = {
        "prompt_tokens": response.usage.input_tokens,
        "completion_tokens": response.usage.output_tokens,
    }
    try:
        parsed = _extract_json_object(content)
        return parsed.get("tool_calls", []), usage
    except (json.JSONDecodeError, AttributeError) as e:
        raise ValueError(f"Model did not return valid JSON with a 'tool_calls' field. Got: {content!r}") from e


def run_episode(task: str, categories: List[str], seed: int, allow_batch: bool, max_rounds: int = 30,
                 allow_batch_cost_aware: bool = False, model: str = "gpt-4o", safety_mode: str = "gated",
                 gate_version: str = "v1", chain_aware_prompt: bool = False,
                 full_rejection_feedback: bool = True, cost_substitution: bool = False,
                 redundancy_elimination: bool = False):
    """
    redundancy_elimination (default False): an ADDITIONAL capability,
    independent of gate_version and cost_substitution -- when
    True, apply_redundancy_elimination runs on `valid` every round, right
    after the gate has decided what's safe to execute, and BEFORE
    cost_substitution (dedup first, then substitute the survivor if a
    cheaper equal-output alternative exists elsewhere). Never rejects
    anything the gate accepted for being unsafe; only drops a slot that is
    genuinely redundant WITHIN this same round -- 2+ accepted tools
    targeting the identical output_type, keeping only the cheapest. Found
    live (real log, gpt-4o-mini, location+transportation, seed=2, round=7):
    the model proposed 4 different "finish transportation" tools in one
    batch, all producing TravelTransportation, all 4 executed and charged
    ($309.61 in pure waste, since the state only needed the type once).
    Real-log scan (experiments/noise_checks/scan_redundancy_in_real_logs.py)
    confirmed genuine, non-trivial recurrence: 34/2451 real rounds (1.39%)
    across 240 real episodes, $2273.74 total real waste, concentrated in
    gpt-4o-mini (2.24% of its rounds) and location+transportation (2.75%);
    zero occurrences for GPT-4o. Since apply_batch only cares about the SET
    of output_types in a batch, dropping every same-output duplicate except
    the cheapest can never remove an output_type the batch would otherwise
    have produced -- state and round count are unaffected by construction,
    same invariant discipline as cost_substitution; only total_tool_cost
    can go down. Only meaningful when safety_mode is "gated"/
    "gated_self_assessed"; a no-op otherwise.

    full_rejection_feedback (default True -- this is a correctness fix, not
    an experimental variant, so it's on by default): whether a round that
    proposed at least one tool call but had EVERY one rejected (zero
    execution) generates corrective feedback for the next round
    (build_full_rejection_feedback) instead of silently leaving
    last_round_feedback=None. Set False only to reproduce the OLD
    (pre-fix) behavior for an explicit with/without comparison -- real
    project logs showed 20 of 37 real stuck episodes never recovered under
    the old behavior, because a fully-rejected round left the next
    prompt byte-identical to the one that just failed.

    chain_aware_prompt (default False, so all existing behavior is
    unchanged unless explicitly opted into): when True, build_prompt uses
    CHAIN_AWARE_BATCH_RULE instead of the default batch_rule -- explicitly
    PERMITS proposing a tool that depends on another proposed tool's output
    in the same batch (the default prompt explicitly FORBIDS this), telling
    the model the system resolves execution order automatically. Exists to
    actually exercise gate v2's chain-resolution mechanism: under the
    default prompt, the model is told never to propose a same-round
    dependency in the first place, so v1 and v2 end up agreeing on every
    real proposal (confirmed empirically: zero divergence across 60 live
    episodes). This flag removes that self-imposed restriction so v1 (which
    still can't resolve chains) and v2 (which can) actually diverge on real
    model output. Orthogonal to gate_version and safety_mode -- combine
    freely (though it's only meaningful under safety_mode in ("gated",
    "gated_self_assessed"), since only those use an execution gate at all).

    cost_substitution (default False): an ADDITIONAL capability, independent
    of gate_version -- when True, apply_cost_substitution runs on `valid`
    every round, right after the gate has already decided what's safe to
    execute, right before apply_batch. It never rejects or reorders
    anything; it only swaps an accepted slot for a cheaper visible tool
    producing the identical output_type, when one exists. Since apply_batch
    only cares about output_type, this is guaranteed by construction to
    never change round count or state -- only total_tool_cost can go down.
    Only meaningful when safety_mode is "gated"/"gated_self_assessed" (the
    two modes that have a real gate-accepted `valid` list to substitute
    into); a no-op otherwise. Combines freely with any gate_version.

    gate_version ("v1" default, or "v2"): which gate function backs the two
    GATED safety_modes ("gated" and "gated_self_assessed"). Does not affect
    "blind"/"self_judged" (they're ungated by definition regardless of
    gate_version) or the would_have_been_rejected diagnostic, which always
    measures against v1 specifically (the original, conservative gate) as
    a fixed reference point.
      - "v1": validate_batch, unchanged -- only accepts a proposed tool if
        its inputs are satisfied by the pre-round state ALONE.
      - "v2": validate_batch_v2 -- additionally accepts a tool whose inputs
        are satisfied by another accepted tool earlier in a resolved order
        WITHIN the same round (a real, same-round dependency chain), via
        topological sort. See validate_batch_v2's docstring in
        brts_core.py for the full resolution algorithm.
    When gate_version="v2", this also tracks total_v2_extra_accepted: how
    many executed calls, across the episode, v2 accepted that v1 (checked
    against the same pre-round proposal/state, computed as a side-by-side
    diagnostic every round regardless of gate_version) would NOT have --
    i.e. real evidence the chain-resolution mechanism is doing genuine
    work, not just passing through unchanged.

    safety_mode:
      - "gated" (default, unchanged behavior): the plain free-choice batching
        prompt; validate_batch is the real execution gate -- only genuinely
        input-satisfied proposals ever execute.
      - "blind": mimics W&D's method -- the SAME free-choice batching prompt
        as "gated", but validate_batch is NOT used to gate execution; every
        proposed real tool name executes via apply_batch_blind regardless of
        whether its inputs are actually satisfied yet.
      - "self_judged": mimics MCP-Bench's approach -- the model itself is
        prompted (via build_prompt's self_judged=True) to self-assess batch
        safety; whatever it returns is trusted and executed the same
        ungated way as "blind".
      - "gated_self_assessed": combines both -- the SAME self-assessment
        prompt as "self_judged" (encouraging a better first-try proposal),
        but validate_batch is still the real execution gate, same as
        "gated". Tests whether self-assessment improves the gated
        condition's efficiency (fewer proposals the gate has to reject)
        even though the external gate already makes execution equally safe
        either way.
    For "blind"/"self_judged" (the two ungated modes), validate_batch is
    still run every round as a pure DIAGNOSTIC (never gates execution) to
    measure how often the baseline actually executed something the real
    safety gate would have rejected -- returned as
    total_would_have_been_rejected/total_executed_calls. For the two gated
    modes this is always 0 by construction, since nothing ungated ever runs.
    """
    data = tools_ready_in_memory(refinement_level=5, min_atomic_cost=19, max_atomic_cost=21,
                                  noise_std=0.1, random_seed=seed)
    solver = GroundtruthSolver(data['tools'])
    tools = solver.tools

    from env.domains.travel.enums import ENUM_MAPPINGS
    from env.core.data_types import get_final_type
    initial = {"TimeInfo"}
    target_types = set()
    for cat in categories:
        for enum_class in ENUM_MAPPINGS[cat]["search"]:
            initial.add(enum_class.__name__)
        target_types.add(get_final_type(cat))
    state = frozenset(initial)

    rounds = 0
    total_llm_calls = 0
    total_tool_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_would_have_been_rejected = 0
    total_executed_calls = 0
    total_v2_extra_accepted = 0
    total_rejected_calls = 0
    total_cost_substitutions = 0
    total_cost_saved_by_substitution = 0.0
    total_redundant_calls_dropped = 0
    total_cost_saved_by_redundancy_elimination = 0.0
    round_log = []
    last_round_feedback = None
    while not is_goal_reached(state, target_types) and rounds < max_rounds:
        # Exclude tools whose output_type is already in state -- calling them
        # again can never help, and leaving them visible is what caused the
        # model to loop re-proposing an already-satisfied step instead of
        # progressing to the next unmet dependency.
        visible = [n for n in get_currently_executable(state, tools) if tools[n].output_type not in state]
        if not visible:
            break
        prompt = build_prompt(state, visible, tools, task, allow_batch,
                               allow_batch_cost_aware=allow_batch_cost_aware,
                               last_round_feedback=last_round_feedback,
                               self_judged=(safety_mode in ("self_judged", "gated_self_assessed")),
                               chain_aware=chain_aware_prompt)
        stub = call_llm_stub_claude if model.startswith("claude") else call_llm_stub
        proposed, usage = stub(prompt, model=model)
        total_llm_calls += 1
        total_prompt_tokens += usage["prompt_tokens"]
        total_completion_tokens += usage["completion_tokens"]

        # Computed unconditionally, regardless of safety_mode -- this is what
        # the REAL safety gate would do to this exact proposal. For "gated"
        # it's also what actually executes; for "blind"/"self_judged" it's
        # diagnostic-only and never gates.
        diag_valid, diag_rejected = validate_batch(proposed, state, tools)

        v2_extra_accepted_this_round = []
        v2_extra_reasoning = {}
        rejected_structured = None  # only populated for gate_version="v3" -- feeds build_universal_rejection_feedback directly
        if safety_mode in ("gated", "gated_self_assessed"):
            if gate_version in ("v2", "v3"):
                if gate_version == "v2":
                    valid, rejected = validate_batch_v2(proposed, state, tools)
                else:
                    valid, rejected_structured = validate_batch_v3(proposed, state, tools)
                    rejected = [r["name"] for r in rejected_structured]  # plain names, for uniform logging/round_log
                # What did this gate accept that v1 (diag_valid, computed above
                # against this exact same proposal/state) would NOT have? Real
                # evidence the chain-resolution mechanism is doing work, not a
                # no-op. `state` here is still the PRE-round value -- apply_batch
                # (a few lines below) hasn't reassigned it yet.
                v2_extra_accepted_this_round = [n for n in valid if n not in diag_valid]
                v2_extra_reasoning = {}
                for name in v2_extra_accepted_this_round:
                    missing_from_state = sorted(set(tools[name].input_types) - state)
                    providers = {t: tools[t].output_type for t in valid
                                 if tools[t].output_type in missing_from_state}
                    v2_extra_reasoning[name] = f"needed {missing_from_state}, provided this round by {providers}"
            else:
                valid = diag_valid
                rejected = diag_rejected
            if not allow_batch and len(valid) > 1:
                valid = valid[:1]  # enforce CostBench's real original constraint faithfully
            gate_valid = valid  # PRE-substitution/dedup names -- what the gate itself accepted, for feedback text only
            redundancy_dropped_this_round = []
            if redundancy_elimination:
                valid, redundancy_dropped_this_round = apply_redundancy_elimination(valid, tools)
            cost_substitutions_this_round = []
            if cost_substitution:
                valid, cost_substitutions_this_round = apply_cost_substitution(valid, state, tools, visible)
            state = apply_batch(state, tools, valid)
            total_tool_cost += sum(tools[n].cost for n in valid)
            would_have_been_rejected_this_round = []  # always empty by construction -- nothing ungated ever ran
        else:  # "blind" or "self_judged" -- no external gate, apply_batch_blind executes ungated
            executed_names = proposed
            if not allow_batch and len(executed_names) > 1:
                executed_names = executed_names[:1]
            state, valid, cost = apply_batch_blind(state, tools, executed_names)
            total_tool_cost += cost
            rejected = [n for n in proposed if n not in valid]  # hallucinated names / sequential-truncated only
            # Of what we ACTUALLY executed blindly, which would the real gate
            # have refused (real tool, but inputs not yet satisfied)?
            would_have_been_rejected_this_round = [n for n in valid if n in diag_rejected]
            cost_substitutions_this_round = []  # cost_substitution only applies to a real gate's accepted `valid`
            redundancy_dropped_this_round = []  # redundancy_elimination only applies to a real gate's accepted `valid`
            gate_valid = valid  # no substitution possible in this branch; kept for symmetry with the gated branch

        total_would_have_been_rejected += len(would_have_been_rejected_this_round)
        total_executed_calls += len(valid)
        total_v2_extra_accepted += len(v2_extra_accepted_this_round)
        total_rejected_calls += len(rejected)
        total_cost_substitutions += len(cost_substitutions_this_round)
        total_cost_saved_by_substitution += sum(s["cost_saved"] for s in cost_substitutions_this_round)
        total_redundant_calls_dropped += len(redundancy_dropped_this_round)
        total_cost_saved_by_redundancy_elimination += sum(d["cost_saved"] for d in redundancy_dropped_this_round)
        round_log.append({
            "round": rounds, "proposed": proposed, "valid": valid, "rejected": rejected,
            "would_have_been_rejected": would_have_been_rejected_this_round,
            "v2_extra_accepted": v2_extra_accepted_this_round,
            "v2_extra_reasoning": v2_extra_reasoning,
            "cost_substitutions": cost_substitutions_this_round,
            "redundancy_dropped": redundancy_dropped_this_round,
            "prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"],
        })
        # `visible` is recomputed next iteration from (possibly unchanged) state,
        # so this fires again next round too if the model repeats the same
        # mistake -- same re-fire-every-round behavior as the logistics
        # harness's cycle-detection feedback. The four outcomes below are
        # mutually exclusive by construction (a plain if/elif/elif/else,
        # checked in this exact order): empty proposal; gate v3 with ANY
        # rejection (partial or full -- broader than the v1/v2 case below,
        # since v3's reasoning is generated directly by the gate and doesn't
        # need to wait for a full rejection to have something to say); a
        # full rejection under v1/v2 (unchanged from before, gated behind
        # full_rejection_feedback); or (the final implicit else) nothing
        # needs explaining -- at least one tool executed and (for v1/v2)
        # nothing was rejected, or execution wasn't gated at all.
        # `state` is safe to reuse for the v1/v2 full-rejection case even
        # though it was already reassigned above (apply_batch/apply_batch_blind
        # with an empty `valid` batch is a no-op, so it still equals the
        # pre-round state) -- build_universal_rejection_feedback needs no
        # state at all, since validate_batch_v3 already computed its
        # reasoning against the correct pre-round-plus-chain state itself.
        # Uses `gate_valid` (the gate's own pre-substitution acceptance),
        # NOT `valid` -- cost_substitution is bookkeeping about which
        # concrete tool gets charged, not information the model needs to
        # reconstruct what happened; feeding it substituted names back into
        # the prompt would make cost_substitution's on/off state leak into
        # subsequent model behavior, breaking the "invisible to the agent"
        # property the whole mechanism is supposed to have.
        if not proposed:
            last_round_feedback = build_empty_response_feedback(visible)
        elif gate_version == "v3" and rejected:
            last_round_feedback = build_universal_rejection_feedback(rejected_structured, gate_valid, visible)
        elif not valid and full_rejection_feedback:
            last_round_feedback = build_full_rejection_feedback(rejected, state, tools, visible)
        else:
            last_round_feedback = None
        unsafe_tag = f" [UNSAFE: would-have-been-rejected={would_have_been_rejected_this_round}]" \
            if would_have_been_rejected_this_round else ""
        v2_tag = f" [V2 CHAIN: accepted beyond v1={v2_extra_accepted_this_round} | {v2_extra_reasoning}]" \
            if v2_extra_accepted_this_round else ""
        cs_tag = f" [COST-SUB: {cost_substitutions_this_round}]" if cost_substitutions_this_round else ""
        print(f"  round {rounds}: proposed={proposed} valid={valid} rejected={rejected}{unsafe_tag}{v2_tag}{cs_tag} "
              f"(prompt_tokens={usage['prompt_tokens']}, completion_tokens={usage['completion_tokens']})")
        rounds += 1

    return {
        "rounds": rounds, "llm_calls": total_llm_calls, "total_tool_cost": total_tool_cost,
        "total_prompt_tokens": total_prompt_tokens, "total_completion_tokens": total_completion_tokens,
        "total_would_have_been_rejected": total_would_have_been_rejected,
        "total_executed_calls": total_executed_calls,
        "total_cost_substitutions": total_cost_substitutions,
        "total_cost_saved_by_substitution": total_cost_saved_by_substitution,
        "total_redundant_calls_dropped": total_redundant_calls_dropped,
        "total_cost_saved_by_redundancy_elimination": total_cost_saved_by_redundancy_elimination,
        "unsafe_execution_rate": (total_would_have_been_rejected / total_executed_calls) if total_executed_calls else 0.0,
        "total_v2_extra_accepted": total_v2_extra_accepted,
        "total_rejected_calls": total_rejected_calls,
        "goal_reached": is_goal_reached(state, target_types),
        "round_log": round_log,
    }


if __name__ == "__main__":
    print("call_llm_stub now makes REAL OpenAI API calls and will incur cost.")
    print("Requires OPENAI_API_KEY set in your environment (your own key -- see _get_client).")
    print("This run makes THREE full episodes (sequential, batched, batched cost-aware) PER SEED, seeds 1-10.\n")
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)

    CONDITIONS = [
        ("sequential", dict(allow_batch=False, allow_batch_cost_aware=False)),
        ("batched", dict(allow_batch=True, allow_batch_cost_aware=False)),
        ("batched_cost_aware", dict(allow_batch=True, allow_batch_cost_aware=True)),
    ]

    seed_results = []
    for seed in range(1, 11):
        row = {"seed": seed}
        for label, kwargs in CONDITIONS:
            print(f"=== {label.upper()}, seed={seed} ===")
            result = run_episode(task="location", categories=["location", "transportation"],
                                  seed=seed, **kwargs)
            print(f"{label} summary: rounds={result['rounds']}, llm_calls={result['llm_calls']}, "
                  f"total_tool_cost={result['total_tool_cost']:.2f}, goal_reached={result['goal_reached']}\n")
            row[f"{label}_calls"] = result["llm_calls"]
            row[f"{label}_cost"] = result["total_tool_cost"]
            row[f"{label}_ok"] = result["goal_reached"]
        seed_results.append(row)

    print("=== SUMMARY (seeds 1-10): LLM calls / total tool cost per condition ===")
    print(f"{'seed':>4} {'seq_calls':>9} {'seq_cost':>9} {'batch_calls':>11} {'batch_cost':>10} "
          f"{'ca_calls':>8} {'ca_cost':>9}")
    for r in seed_results:
        print(f"{r['seed']:>4} {r['sequential_calls']:>9} {r['sequential_cost']:>9.2f} "
              f"{r['batched_calls']:>11} {r['batched_cost']:>10.2f} "
              f"{r['batched_cost_aware_calls']:>8} {r['batched_cost_aware_cost']:>9.2f}")
