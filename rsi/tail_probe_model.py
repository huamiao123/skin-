"""Independent student decoder for the preregistered seed17 tail experiment.

Construct the supplied ``ControlledModel`` from the strict, audited B model
state before constructing this class. No initializer or pretrained downloader
runs here: every component is copied from that supplied model. The reference
CNN (encoder, prefix and tail), transformer and auxiliary head remain frozen.
The student tail has independent parameter and buffer storage.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .models import ControlledModel, state_hash


GROUPS = ("F-Mean", "F-RSI", "U-Mean", "U-RSI", "U-NoMessage")
EXPECTED_TRAINABLE = {"F-Mean": 131584, "F-RSI": 131584,
                      "U-Mean": 311329, "U-RSI": 311329,
                      "U-NoMessage": 179745}


def interpolate_update_logits(z_ref: Tensor, z_on: Tensor, alpha: float) -> Tensor:
    """Scale the complete logit update, retaining exact alpha=0/1 endpoints.

    This is separate from feature/message shrinkage. It must be called on the
    canonical logits before restoring original image coordinates. The sole
    scalar alpha is fixed across images; it is never inferred from labels.
    """
    if z_ref.shape != z_on.shape or z_ref.ndim != 4 or z_ref.shape[1] != 1:
        raise ValueError("complete update requires matching logits[B,1,H,W]")
    if z_ref.device != z_on.device:
        raise ValueError("reference and student logits must share a device")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be one finite scalar in [0,1]")
    with torch.autocast(device_type=z_ref.device.type, enabled=False):
        if alpha == 0:
            return z_ref.float()
        if alpha == 1:
            return z_on.float()
        return z_ref.float() + alpha * (z_on.float() - z_ref.float())


class TailProbeModel(nn.Module):
    """B reference plus a distinct student tail; group selects trainability."""

    def __init__(self, base_model: ControlledModel, group: str) -> None:
        super().__init__()
        if not isinstance(base_model, ControlledModel):
            raise TypeError("base_model must be the strict-loaded ControlledModel")
        self.input_size = base_model.input_size
        # Copying, rather than taking ownership, also protects the caller's B.
        self.cnn = copy.deepcopy(base_model.cnn)
        self.cnn._permanently_frozen = True
        self.transformer = copy.deepcopy(base_model.transformer)
        self.aux_head = copy.deepcopy(base_model.aux_head)
        self.message = copy.deepcopy(base_model.message)
        self.student_tail = copy.deepcopy(base_model.tail)
        self._group = ""
        self.set_group(group)

    @property
    def group(self) -> str:
        return self._group

    @property
    def reference_tail(self) -> nn.Module:
        return self.cnn.tail

    @property
    def tail(self) -> nn.Module:
        """The trainable/frozen student tail, never the reference tail."""
        return self.student_tail

    def _components(self) -> dict[str, nn.Module]:
        return {"cnn": self.cnn, "transformer": self.transformer,
                "aux_head": self.aux_head, "message": self.message,
                "student_tail": self.student_tail}

    def _component_trainable(self, name: str) -> bool:
        return ((name == "message" and self.group != "U-NoMessage")
                or (name == "student_tail" and self.group.startswith("U-")))

    def set_group(self, group: str) -> "TailProbeModel":
        if group not in GROUPS:
            raise ValueError(f"Unknown tail probe group: {group!r}; expected {GROUPS}")
        self._group = group
        for name, module in self._components().items():
            module.requires_grad_(self._component_trainable(name))
            for parameter in module.parameters():
                parameter.grad = None
        self.train(self.training)
        count = sum(p.numel() for p in self.trainable_parameters())
        if count != EXPECTED_TRAINABLE[group]:
            raise RuntimeError(f"{group} trainable count {count} != {EXPECTED_TRAINABLE[group]}")
        return self

    def train(self, mode: bool = True) -> "TailProbeModel":
        super().train(mode)
        if self._group:
            for name, module in self._components().items():
                module.train(mode and self._component_trainable(name))
        return self

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def context_token_valid(self, pixel_valid: Tensor | None, batch_size: int) -> Tensor | None:
        # Reuse the exact pooling and letterbox treatment of the frozen source.
        return ControlledModel.context_token_valid(self, pixel_valid, batch_size)

    def _features(self, rgb: Tensor, *, need_context: bool) -> tuple[Tensor, Tensor, Tensor | None]:
        with torch.no_grad():
            f8, c4 = self.cnn.features(rgb)
            t16 = self.transformer(rgb) if need_context else None
        return f8, c4, t16

    def _student(self, f8: Tensor, c4: Tensor, t16: Tensor | None,
                 pixel_valid: Tensor | None) -> tuple[Tensor, Tensor | None]:
        if self.group == "U-NoMessage":
            message = None
            updated = f8
        else:
            if t16 is None:
                raise RuntimeError("message groups require frozen transformer features")
            message = self.message(f8, t16, self.context_token_valid(pixel_valid, f8.shape[0]))
            updated = f8 + message
        # No no_grad here: frozen F tails still backpropagate to the message.
        return self.student_tail(updated, c4), message

    def forward(self, rgb: Tensor, pixel_valid: Tensor | None = None) -> Tensor:
        """Student deployment: one tail decode, no teacher or labels as input."""
        f8, c4, t16 = self._features(rgb, need_context=self.group != "U-NoMessage")
        return self._student(f8, c4, t16, pixel_valid)[0]

    def forward_details(self, rgb: Tensor, pixel_valid: Tensor | None = None) -> dict[str, Any]:
        f8, c4, t16 = self._features(rgb, need_context=self.group != "U-NoMessage")
        with torch.no_grad():
            z_ref = self.reference_tail(f8, c4)
        z_on, message = self._student(f8, c4, t16, pixel_valid)
        return {"z0": z_ref, "z1": z_on, "message": message,
                "f8": f8, "c4": c4, "t16": t16}

    def forward_pair(self, rgb: Tensor, pixel_valid: Tensor | None = None) -> tuple[Tensor, Tensor]:
        out = self.forward_details(rgb, pixel_valid)
        return out["z0"], out["z1"]

    def forward_reference(self, rgb: Tensor) -> Tensor:
        with torch.no_grad():
            f8, c4 = self.cnn.features(rgb)
            return self.reference_tail(f8, c4)

    def forward_cnn(self, rgb: Tensor) -> Tensor:
        """Explicit alias for the fixed B teacher, not the current off output."""
        return self.forward_reference(rgb)

    def forward_off(self, rgb: Tensor, pixel_valid: Tensor | None = None) -> Tensor:
        """Current student tail off diagnostic; never an RSI teacher."""
        f8, c4, _ = self._features(rgb, need_context=False)
        return self.student_tail(f8, c4)

    def teacher_state_hash(self) -> str:
        return state_hash(self.cnn)

    def anchor_state_hash(self) -> str:
        return self.teacher_state_hash()

    def frozen_state_hash(self) -> str:
        hashes = {name: state_hash(module) for name, module in self._components().items()
                  if not self._component_trainable(name)}
        return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()

    def independent_tail_storage(self) -> bool:
        reference = list(self.reference_tail.parameters()) + list(self.reference_tail.buffers())
        student = list(self.student_tail.parameters()) + list(self.student_tail.buffers())
        reference_storages = {tensor.untyped_storage().data_ptr() for tensor in reference if tensor.numel()}
        return all(tensor.untyped_storage().data_ptr() not in reference_storages
                   for tensor in student if tensor.numel())

    def load_legacy_state_dict(self, state: Mapping[str, Tensor], *, strict: bool = True):
        """Strictly reproduce a completed legacy F state with the same teacher.

        The complete legacy ControlledModel state is required. Only message
        changes are permitted relative to this object's common B initialization.
        U groups cannot use this entry point to initialize from a legacy D run.
        """
        if not strict:
            raise ValueError("legacy reuse requires strict=True")
        if not self.group.startswith("F-"):
            raise ValueError("legacy D state reuse is restricted to F groups")
        current = self.state_dict()
        expected = {name for name in current if not name.startswith("student_tail.")}
        if set(state) != expected:
            raise ValueError("legacy state keys differ from the complete ControlledModel state")
        for name in sorted(expected):
            if name.startswith(("cnn.", "transformer.", "aux_head.")):
                if not torch.equal(state[name].detach().cpu(), current[name].detach().cpu()):
                    raise ValueError(f"legacy frozen state differs from common B: {name}")
        translated = dict(state)
        translated.update({"student_tail." + name: state["cnn.tail." + name]
                           for name in self.student_tail.state_dict()})
        result = self.load_state_dict(translated, strict=True)
        self.train(self.training)
        return result

    def architecture_audit(self) -> dict[str, Any]:
        return {"group": self.group, "input_size": self.input_size,
                "parameters": {name: sum(p.numel() for p in module.parameters())
                               for name, module in self._components().items()},
                "trainable_parameters": sum(p.numel() for p in self.trainable_parameters()),
                "trainable_parameter_names": self.trainable_parameter_names(),
                "student_reference_storage_independent": self.independent_tail_storage(),
                "teacher": "fixed common B CNN including its original decoder tail",
                "deployment_tail_decodes": 1,
                "deployment_transformer_forwards": int(self.group != "U-NoMessage")}
