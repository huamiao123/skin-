# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import namedtuple
from typing import Optional, List, Union

import torch
from transformers import PretrainedConfig, PreTrainedModel
from .mamba_vision import *
from timm.models import create_model, load_checkpoint


class MambaVisionConfig(PretrainedConfig):

    def __init__(
        self,
        args: Optional[dict] = None,
        **kwargs,
    ):
        self.args = args
        super().__init__(**kwargs)


class MambaVisionModel(PreTrainedModel):
    """Pretrained Hugging Face model for MambaVision.

    This class inherits from PreTrainedModel, which provides
    HuggingFace's functionality for loading and saving models.
    """

    config_class = MambaVisionConfig

    def __init__(self, config):
        super().__init__(config)
        MambaVisionArgs = namedtuple("MambaVisionArgs", config.args.keys())
        args = MambaVisionArgs(**config.args)
        self.config = config
        self.model = create_model(args.model)

    def forward(self, x: torch.Tensor):
        return self.model.forward(x)
