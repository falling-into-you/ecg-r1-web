from __future__ import annotations

import config

from .mock import MockProvider
from .swift_rollout import SwiftRolloutProvider


def get_provider():
    backend = config.INFERENCE_BACKEND
    if backend == "mock":
        return MockProvider()
    if backend == "swift_rollout":
        return SwiftRolloutProvider(
            infer_url=config.SWIFT_ROLLOUT_URL,
            health_url=config.SWIFT_ROLLOUT_HEALTH_URL,
            timeout_s=config.INFERENCE_TIMEOUT_S,
        )
    raise ValueError(f"Unsupported INFERENCE_BACKEND={backend!r}")
