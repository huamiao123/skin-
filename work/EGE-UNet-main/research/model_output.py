from __future__ import annotations

from typing import Any

import torch


def unpack_segmentation_output(output: Any) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Return ``(final_probability, deep_supervision_probabilities)``.

    The adapter understands the legacy EGE tuple, the explicit research wrapper,
    and the named legacy dictionaries. It never applies sigmoid: research_v1
    keeps the existing probability-output contract.
    """
    if isinstance(output, dict) and "legacy_output" in output:
        output = output["legacy_output"]

    if isinstance(output, dict):
        final = output.get("final_output", output.get("final"))
        deep = output.get("deep_supervision", ())
    elif isinstance(output, tuple) and len(output) == 2:
        deep, final = output
    elif torch.is_tensor(output):
        final, deep = output, ()
    else:
        raise TypeError(f"unsupported segmentation output type: {type(output)!r}")

    if not torch.is_tensor(final):
        raise TypeError("final segmentation output must be a tensor")
    if torch.is_tensor(deep):
        deep = (deep,)
    elif not isinstance(deep, (tuple, list)):
        raise TypeError("deep supervision output must be a tensor sequence")
    return final, tuple(deep)


def assert_probability_output(probability: torch.Tensor) -> None:
    if not probability.is_floating_point():
        raise TypeError("segmentation probability must be floating point")
    if not torch.isfinite(probability).all():
        raise ValueError("segmentation probability contains non-finite values")
    if torch.any(probability < 0) or torch.any(probability > 1):
        raise ValueError("research_v1 expects probabilities in [0, 1], not logits")

