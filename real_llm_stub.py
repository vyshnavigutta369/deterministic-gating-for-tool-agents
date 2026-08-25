"""
Fully wired version of call_llm_stub for run_brts_minimal.py.

TO USE THIS FOR REAL:
1. Revoke the key you pasted in chat (platform.openai.com -> API Keys) --
   it's exposed and must not be reused, regardless of anything else.
2. Generate a NEW key there.
3. On your OWN machine (not in any chat), run:
       export OPENAI_API_KEY="your-new-key-here"
       pip install openai
4. Copy the function below into run_brts_minimal.py, replacing the
   NotImplementedError stub version of call_llm_stub.
5. Run: python3 run_brts_minimal.py

This code is real and complete. It cannot run in this sandbox (no network
path to OpenAI's API here) -- it will only work on your own machine, with
your own key, set as an environment variable.
"""
import os
import json
from openai import OpenAI

_client = None

def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable not set. "
                "Run: export OPENAI_API_KEY='your-key-here' in your terminal first."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def call_llm_stub(prompt: str, model: str = "gpt-4o") -> list:
    """Real implementation -- makes an actual API call. Requires
    OPENAI_API_KEY to be set in your environment (see module docstring)."""
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = response.choices[0].message.content
    try:
        parsed = json.loads(content)
        return parsed.get("tool_calls", [])
    except (json.JSONDecodeError, AttributeError) as e:
        raise ValueError(f"Model did not return valid JSON with a 'tool_calls' field. Got: {content!r}") from e
