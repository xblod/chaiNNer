from __future__ import annotations

import numpy as np

from ...impl.color_transfer import OverflowMethod, TransferColorSpace, color_transfer
from ...node_base import NodeBase
from ...node_factory import NodeFactory
from ...properties.inputs import BoolInput, EnumInput, ImageInput
from ...properties.outputs import ImageOutput
from ...utils.utils import get_h_w_c
from . import category as ImageFilterCategory


@NodeFactory.register("chainner:image:color_transfer")
class ColorTransferNode(NodeBase):
    """
    Transfers colors from one image to another

    This code was adapted from Adrian Rosebrock's color_transfer script,
    found at: https://github.com/jrosebr1/color_transfer (© 2014, MIT license).
    """

    def __init__(self):
        super().__init__()
        self.description = """Transfers colors from reference image.
            Different combinations of settings may perform better for
            different images. Try multiple setting combinations to find
            best results."""
        self.inputs = [
            ImageInput("Image", channels=[1, 3, 4]),
            ImageInput("Reference Image", channels=[3, 4]),
            EnumInput(
                TransferColorSpace,
                label="Colorspace",
                option_labels={TransferColorSpace.LAB: "L*a*b*"},
            ),
            EnumInput(OverflowMethod),
            BoolInput("Reciprocal Scaling Factor", default=True),
        ]
        self.outputs = [ImageOutput("Image", image_type="Input0")]
        self.category = ImageFilterCategory
        self.name = "Color Transfer"
        self.icon = "MdInput"
        self.sub = "Correction"

    def run(
        self,
        img: np.ndarray,
        ref_img: np.ndarray,
        colorspace: TransferColorSpace,
        overflow_method: OverflowMethod,
        reciprocal_scale: bool,
    ) -> np.ndarray:
        """
        Transfers the color distribution from source image to target image.
        """

        _, _, img_c = get_h_w_c(img)

        # Preserve alpha
        alpha = None
        if img_c == 4:
            alpha = img[:, :, 3]

        transfer = color_transfer(
            img, ref_img, colorspace, overflow_method, reciprocal_scale
        )

        if alpha is not None:
            transfer = np.dstack((transfer, alpha))

        return transfer
