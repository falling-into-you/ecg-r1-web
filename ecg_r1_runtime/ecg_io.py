from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch


def load_ecg(ecg_path: str, ecg_seq_length: Optional[int] = 5000, root_ecg_dir: Optional[str] = None) -> torch.Tensor:
    """Load a WFDB ECG record and return a channel-first tensor."""
    import wfdb

    if isinstance(ecg_path, torch.Tensor):
        return ecg_path

    path = ecg_path
    if root_ecg_dir and not os.path.isabs(path):
        path = os.path.join(root_ecg_dir, path)

    try:
        ecg_data = wfdb.rdsamp(path)[0]
    except Exception as exc:
        raise ValueError(f"Failed to load ECG from {path}: {exc}") from exc

    ecg_data[np.isnan(ecg_data)] = 0
    ecg_data[np.isinf(ecg_data)] = 0
    ecg_tensor = torch.from_numpy(np.transpose(ecg_data, (1, 0)).astype(np.float32))

    channels, length = ecg_tensor.shape
    if ecg_seq_length is None:
        return ecg_tensor

    if length < ecg_seq_length:
        padded = torch.zeros((channels, ecg_seq_length), dtype=ecg_tensor.dtype)
        padded[:, :length] = ecg_tensor
        return padded

    if length > ecg_seq_length:
        return ecg_tensor[:, :ecg_seq_length]

    return ecg_tensor
