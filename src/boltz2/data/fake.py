"""Synthetic complex generator — build this FIRST.

``fake_complex()`` fabricates a batch dict with the same keys the real parser
will eventually emit, so every module can be shape/grad tested without real
biology. Provide a deterministic (fixed-seed) variant for the overfit-one test.
"""
