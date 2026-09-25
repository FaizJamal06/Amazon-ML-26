"""Exact macro F0.5 per S1 entity, mirroring the official rule (empty/empty = 1, empty vs non-empty = 0).

Contract:
In: predictions {s1_id: set} + ground truth {s1_id: set}. Out: macro F0.5 overall / per country / per script.

Owner: Faiz (R1 Lead / Eval / Decision)
"""
