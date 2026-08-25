"""
DELIBERATE MINIMAL STAND-IN -- not a vendored copy of the real file.

The real bfcl_eval.constants.model_config pulls in ~40 provider-specific
model handler modules (Claude, Gemini, Mistral, Cohere, etc.), most needing
SDKs not installed in this project, just to supply MODEL_CONFIG_MAPPING --
of which ast_checker.py's convert_func_name() only ever reads ONE field:
underscore_to_dot.

That field exists because BFCL's OFFICIAL evaluation calls each model's
NATIVE function-calling API, where some providers (historically OpenAI,
Mistral, Google) reject dots in function names -- so BFCL escapes '.' to
'_' before sending the function doc, then converts back for comparison.
This harness never uses any native function-calling API: it asks the model
for raw JSON text (see experiments/bfcl_parallel_episode.py), where a dot in
a JSON key is not restricted at all. So the correct value for THIS design is
simply "no conversion needed" -- False -- regardless of model name.

This does not change any actual comparison/type-checking logic in
ast_checker.py (which is vendored unmodified) -- it only supplies the one
config value that code path reads, without dragging in unrelated provider
SDKs this project doesn't use.
"""


class _NoConversionConfig:
    underscore_to_dot = False


class _ModelConfigMapping:
    def __getitem__(self, key):
        return _NoConversionConfig()


MODEL_CONFIG_MAPPING = _ModelConfigMapping()
