"""CLI entry point: python -m ber.pipeline ingest|block|featurize|train|predict|submit|all.

Contract:
In: config + cached artifacts of the previous stage. Out: the artifacts of the requested stage.
Every stage supports --subworld F (closed sub-world, CLAUDE.md §6).

Owner: Faiz (R1 Lead / Eval / Decision)
"""
