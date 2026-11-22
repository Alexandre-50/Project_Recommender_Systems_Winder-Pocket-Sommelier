"""Load a trained model and score users (used by evaluation and by the API)."""
import json
from pathlib import Path

import torch

from sommelier.data.loaders import load_processed
from sommelier.models.hgib import HGIBKGAT


def load_model(model_dir, ds=None):
    """Rebuild the model saved by train.py. Returns (model, dataset, config)."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    if ds is None:
        ds = load_processed(config["data_dir"])
    dim = config["dim"]
    model = HGIBKGAT(ds.ckg, dim=dim, layer_dims=(dim, dim // 2, dim // 4), beta=config["beta"],
                     use_gre=config["use_gre"],
                     use_questionnaire=config["use_questionnaire"], use_hgib=config["use_hgib"])
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.eval()
    return model, ds, config


def embeddings(model, edges=None):
    """Final node representations (one forward on the whole graph, no gradient)."""
    model.eval()
    with torch.no_grad():
        return model.forward(edges)[-1][0]
