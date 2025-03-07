from __future__ import annotations

import gc
from typing import Optional

import numpy as np
import torch

from ...impl.image_utils import as_3d
from ...impl.pytorch.types import PyTorchInpaintModel
from ...impl.pytorch.utils import (
    np2tensor,
    safe_cuda_cache_empty,
    tensor2np,
    to_pytorch_execution_options,
)
from ...node_base import NodeBase
from ...node_factory import NodeFactory
from ...properties import expression
from ...properties.inputs import ImageInput
from ...properties.inputs.pytorch_inputs import InpaintModelInput
from ...properties.outputs import ImageOutput
from ...utils.exec_options import ExecutionOptions, get_execution_options
from ...utils.utils import get_h_w_c
from . import category as PyTorchCategory


@NodeFactory.register("chainner:pytorch:inpaint")
class InpaintNode(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = "Inpaint an image with given mask."
        self.inputs = [
            ImageInput(channels=3),
            ImageInput(label="Mask", channels=1),
            InpaintModelInput(),
        ]
        self.outputs = [
            ImageOutput(
                image_type=expression.Image(
                    width="Input0.width & Input1.width",
                    height="Input0.height & Input1.height",
                ),
                channels=3,
            ).with_never_reason(
                "The given image and mask must have the same resolution."
            )
        ]
        self.category = PyTorchCategory
        self.name = "Inpaint"
        self.icon = "PyTorch"
        self.sub = "Processing"

    @staticmethod
    def ceil_modulo(x: int, mod: int) -> int:
        if x % mod == 0:
            return x
        return (x // mod + 1) * mod

    def pad_img_to_modulo(
        self,
        img: np.ndarray,
        mod: int,
        square: bool = False,
        min_size: Optional[int] = None,
    ):
        img = as_3d(img)
        h, w, _ = get_h_w_c(img)
        out_h = self.ceil_modulo(h, mod)
        out_w = self.ceil_modulo(w, mod)

        if min_size is not None:
            assert min_size % mod == 0
            out_w = max(min_size, out_w)
            out_h = max(min_size, out_h)

        if square:
            max_size = max(out_h, out_w)
            out_h = max_size
            out_w = max_size

        return np.pad(img, ((0, out_h - h), (0, out_w - w), (0, 0)), mode="symmetric")

    def inpaint(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        model: PyTorchInpaintModel,
        options: ExecutionOptions,
    ):
        with torch.no_grad():
            # TODO: use bfloat16 if RTX
            use_fp16 = options.fp16 and model.supports_fp16
            device = torch.device(options.full_device)
            model = model.to(device)
            model = model.half() if use_fp16 else model.float()

            orig_height, orig_width, _ = get_h_w_c(img)

            img = self.pad_img_to_modulo(
                img, model.pad_mod, model.pad_to_square, model.min_size
            )
            mask = self.pad_img_to_modulo(
                mask, model.pad_mod, model.pad_to_square, model.min_size
            )

            img_tensor = np2tensor(img, change_range=True)
            mask_tensor = np2tensor(mask, change_range=True)

            d_img = None
            d_mask = None
            try:
                d_img = img_tensor.to(device)
                d_img = d_img.half() if use_fp16 else d_img.float()

                d_mask = mask_tensor.to(device)
                d_mask = (d_mask > 0.5) * 1
                d_mask = d_mask.half() if use_fp16 else d_mask.float()

                result = model(d_img, d_mask)
                result = tensor2np(
                    result.detach().cpu().detach(),
                    change_range=False,
                    imtype=np.float32,
                )

                del d_img
                del d_mask

                return result[0:orig_height, 0:orig_width]
            except RuntimeError:
                # Collect garbage (clear VRAM)
                if d_img is not None:
                    try:
                        d_img.detach().cpu()
                    except:
                        pass
                    del d_img
                if d_mask is not None:
                    try:
                        d_mask.detach().cpu()
                    except:
                        pass
                    del d_mask
                gc.collect()
                safe_cuda_cache_empty()

                raise

    def run(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        model: PyTorchInpaintModel,
    ) -> np.ndarray:
        """Inpaint an image"""

        assert (
            img.shape[:2] == mask.shape[:2]
        ), "Input image and mask must have the same resolution"

        exec_options = to_pytorch_execution_options(get_execution_options())

        return self.inpaint(img, mask, model, exec_options)
