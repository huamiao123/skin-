"""Check the durable best checkpoints on validation only."""
import gc
import json
from pathlib import Path

import torch

from decoder import build, HERE
from train import reference, interior_masks, validate, atomic_json


def main():
    reference.setup()
    data, _ = reference.load_data()
    val = reference.cache_new(len(data.ids["val"]))
    results = {}
    interiors = None
    for i, arm in enumerate(("f3", "uper64", "f4")):
        folder = HERE/arm
        chosen = json.loads((folder/"complete.json").read_text())["best"]
        model = build(arm)
        state = torch.load(folder/"best.pth", map_location="cpu")
        assert chosen["epoch"] == state["epoch"]
        model.load_state_dict(state["model"], strict=True)
        del state
        reference.fill_cache(model, data, "val", val)
        if interiors is None:
            interiors = interior_masks(val["target"])
        scores, rows = validate(model, val, data.ids["val"], interiors)
        err = abs(chosen["val_macro_dice"] - scores["val_macro_dice"])
        assert err < 1e-6, (arm, err)
        results[arm] = dict(epoch=chosen["epoch"], replay_val_dice=scores["val_macro_dice"],
                            absolute_difference=err, n=len(rows))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(HERE/"checkpoint_replay.json", results)
    print(json.dumps(results), flush=True)


if __name__=="__main__":
    main()
