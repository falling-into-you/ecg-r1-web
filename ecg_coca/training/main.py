import logging

import torch

from ecg_coca.open_clip.factory import create_model_and_transforms, get_model_config


def _pt_load(file_path, map_location=None):
    return torch.load(file_path, map_location=map_location, weights_only=False)


def get_ecg_encoder(model_name, checkpoint_path, device):
    """Load the ECG encoder used by ECG-R1 without importing training code."""
    model_config = get_model_config(model_name)
    logging.info("ecg encoder name: %s", model_name)

    model, _, preprocess_val = create_model_and_transforms(
        model_name,
        "",
        precision="amp",
        device=device,
        jit=False,
        output_dict=True,
    )
    model.to_empty(device=device)
    model = model.ecg
    checkpoint = _pt_load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint["state_dict"]
    ecg_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module.ecg"):
            ecg_state_dict[key[len("module.ecg."):]] = value
    model.load_state_dict(ecg_state_dict)

    logging.info("loaded ECG checkpoint %s", checkpoint_path)
    model.lock()
    return model, preprocess_val, model_config
