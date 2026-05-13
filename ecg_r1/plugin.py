"""Inference-time no-op plugin for ECG-R1 checkpoints.

The training checkpoint references ``ecg_r1/plugin.py`` in ``external_plugins``.
Rollout imports it during argument parsing, but the reward plugins defined in
the original training tree are not needed for serving.
"""

