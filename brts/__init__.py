"""
Deterministic (zero-LLM-call) batch-selection mechanisms for the BRTS harness.

- scored: greedy per-round scoring (RIBQ-style cluster/score/select), with a
  documented shared-prerequisite safeguard and a round_tax knob.
- precomputed: precompute each category's cost-optimal path once (Dijkstra),
  then execute in lockstep with shared-state deduplication.
- length_optimal: same lockstep execution, but each category's precomputed
  path minimizes step count first, cost second.
- synthetic: a no-bundle-tools synthetic domain generator for ablation tests.
"""
from .scored import scored_batch_selection
from .precomputed import execute_precomputed_sequences, precompute_then_batch_selection
from .length_optimal import precompute_length_optimal_selection
from .synthetic import generate_synthetic_domain

__all__ = [
    "scored_batch_selection",
    "execute_precomputed_sequences",
    "precompute_then_batch_selection",
    "precompute_length_optimal_selection",
    "generate_synthetic_domain",
]
