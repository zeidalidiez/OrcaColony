# Capability tasks

This directory holds frozen behavioral tasks for practical model research.
Systems studies under `research/` answer whether an execution method is correct
or resource-feasible. A capability task answers whether training changed one
declared behavior.

Each task must provide:

- an exact task and oracle;
- legal and reproducible data generation;
- a public behavioral-validation split;
- a separately keyed final holdout that stays outside Git until checkpoint
  selection;
- a sample-level evaluator with frozen thresholds and guardrails;
- an initialization baseline before volunteer training;
- contributor-credit intake and private-first Hugging Face destinations.

The final holdout lock may be committed before training, but its examples and
generation key must not be committed or uploaded until the release evaluation.

The first task is [`record-patch-v1`](record-patch-v1/TASK.md).
