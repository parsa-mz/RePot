"""RePoT: one-shot PoT + verified-prefix suffix repair.

- ``RePoTAgent``         (alg.py)  — the headline algorithm.
- ``AdaptiveRePoTAgent`` (adaptive.py) — rule-based dispatcher between fresh
  PoT-retry and suffix repair based on the verified-prefix fraction.
- ``VEXExecutor``        (executor.py) — the chunked verified-execution loop
  that the prefix-repair path delegates to.
"""
