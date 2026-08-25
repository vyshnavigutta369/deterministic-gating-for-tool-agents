"""claude-haiku-4-5, location+transportation, seeds 1-5. See experiments/llm_sweep.py.

Uses call_llm_stub_claude (run_brts_minimal.py) via run_episode's model-name
dispatch (any model starting with "claude" routes to the Anthropic client
instead of OpenAI's). Requires ANTHROPIC_API_KEY, not OPENAI_API_KEY.

Pricing verified 2026-07-27 (web search against current Anthropic pricing
pages): claude-haiku-4-5 is $1.00/1M input tokens, $5.00/1M output tokens --
different from GPT-4o-mini's $0.15/$0.60, so do not reuse those defaults.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from experiments.llm_sweep import run_llm_sweep

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)
    run_llm_sweep(task="location", categories=["location", "transportation"],
                  label="location+transportation", model="claude-haiku-4-5-20251001",
                  input_price_per_million=1.00, output_price_per_million=5.00)
