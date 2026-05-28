# Datasets

> **Canonical dataset**: [`pmazaher/repot-bench`](https://huggingface.co/datasets/pmazaher/repot-bench) on HuggingFace.
> The JSONLs in this directory are the frozen paper-version mirror — use them for offline access or exact paper-version reproducibility. The HF copy is the canonical version (`datasets.load_dataset("pmazaher/repot-bench", "<config>")`).

This directory contains the datasets released alongside the paper.

## `problems/puzzlezoo_775.jsonl` — PuzzleZoo-775

The main puzzle suite, 775 verified problems stratified across four classical
planning environments × controllable complexity.

| Environment       | Complexities | # problems |
|-------------------|--------------|------------|
| Tower of Hanoi    | 8            | 200        |
| Checker Jumping   | 9            | 225        |
| River Crossing    | 4            | 100        |
| Blocksworld       | 10           | 250        |

Schema (one JSON object per line):

| Field                      | Type | Notes                                        |
|----------------------------|------|----------------------------------------------|
| `problem_id`               | str  | Primary key.                                 |
| `environment`              | str  | `tower_of_hanoi` \| `checker_jumping` \| `river_crossing` \| `blocksworld` |
| `complexity`               | int  | Env-specific (e.g. # disks, # blocks).       |
| `initial_state`            | dict | Env-specific representation.                 |
| `goal_state`               | dict | Goal state or predicate.                     |
| `oracle_solution`          | list | Ground-truth action sequence.                |
| `min_steps` / `max_steps`  | int  | Plan length bounds.                          |
| `natural_language_prompt`  | str  | Exact prompt the model sees.                 |
| `metadata`                 | dict | Generator metadata (seed, branching stats).  |

## `problems/planbench_blocksworld.jsonl` — PlanBench Blocksworld adapter

378 instances (3–12 blocks) adapted from the PlanBench Blocksworld split into
the same `Problem` schema used by PuzzleZoo. The original PDDL files come from
the PlanBench repository:

> Valmeekam et al., *PlanBench: An Extensible Benchmark for Evaluating Large
> Language Models on Planning and Reasoning about Change*, NeurIPS 2023.
> <https://arxiv.org/abs/2206.10498>

The original PDDL instances are not redistributed here; this JSONL is a
schema adapter that lets you run the same harness against PlanBench.

## `derail/derail_550.jsonl` — Derail-550

550 mid-rollout injection cases built on top of PuzzleZoo. For each case we
ran the oracle plan to ~1/3 of the way through and injected one wrong action;
methods are scored on whether they recover from the post-injection state to
the goal.

These are **method-agnostic case definitions** — each row describes the
checkpoint state and the injected error. Run `repot derail` to evaluate any
recovery method against these cases.

Schema:

| Field              | Type | Notes                                                |
|--------------------|------|------------------------------------------------------|
| `case_id`          | str  | Primary key (`derail_<sha1[:12]>`).                  |
| `problem_id`       | str  | FK to `puzzlezoo_775.jsonl`.                         |
| `environment`      | str  | Mirrors source problem.                              |
| `complexity`       | int  | Mirrors source problem.                              |
| `injection_step`   | int  | Index along the oracle plan (~1/3 of plan length).   |
| `injection_type`   | str  | Categorical label for the kind of wrong action.      |
| `last_valid_state` | dict | Trusted checkpoint state (step `injection_step-1`).  |
| `injected_action`  | dict | The wrong action applied at `injection_step`.        |
| `injected_state`   | dict | Post-injection state passed to recovery methods.     |

Per-method results are not part of this dataset; reproduce them by running
`repot derail` against the case file.

## License

Datasets are released under **CC-BY-4.0**. Source code is **Apache-2.0** —
see `LICENSE` at the repo root.
