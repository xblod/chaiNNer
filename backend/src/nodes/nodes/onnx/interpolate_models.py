from __future__ import annotations

from copy import deepcopy
from typing import List, Tuple

import numpy as np
import onnx
from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
from onnx import numpy_helper as onph
from sanic.log import logger

from ...impl.onnx.model import OnnxModel, load_onnx_model
from ...impl.onnx.utils import safely_optimize_onnx_model
from ...impl.upscale.auto_split_tiles import NO_TILING
from ...node_base import NodeBase
from ...node_factory import NodeFactory
from ...properties.inputs import OnnxModelInput, SliderInput
from ...properties.outputs import NumberOutput, OnnxModelOutput
from . import category as ONNXCategory
from .upscale_image import OnnxImageUpscaleNode


@NodeFactory.register("chainner:onnx:interpolate_models")
class OnnxInterpolateModelsNode(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = "Interpolate two ONNX models of the same type together. \
            Note: models must share a common 'pretrained model' ancestor \
            in order to be interpolatable."
        self.inputs = [
            OnnxModelInput("Model A"),
            OnnxModelInput("Model B"),
            SliderInput(
                "Weights",
                controls_step=5,
                slider_step=1,
                maximum=100,
                default=50,
                unit="%",
                note_expression="`Model A ${100 - value}% ― Model B ${value}%`",
                ends=("A", "B"),
            ),
        ]
        self.outputs = [
            OnnxModelOutput(),
            NumberOutput("Amount A", "100 - Input2"),
            NumberOutput("Amount B", "Input2"),
        ]

        self.category = ONNXCategory
        self.name = "Interpolate Models"
        self.icon = "BsTornado"
        self.sub = "Utility"

    @staticmethod
    def perform_interp(
        model_a_weights: RepeatedCompositeFieldContainer,
        model_b_weights: RepeatedCompositeFieldContainer,
        amount: float,
    ) -> List[onnx.TensorProto]:
        amount_b = amount / 100
        amount_a = 1 - amount_b

        interp_weights_list = []
        for weight_a, weight_b in zip(model_a_weights, model_b_weights):
            weight_name = weight_b.name
            weight_array_a = onph.to_array(weight_a)
            weight_array_b = onph.to_array(weight_b)

            assert (
                weight_array_a.shape == weight_array_b.shape
            ), "Weights must have same size and shape"

            weight_array_interp = (
                weight_array_a * amount_a + weight_array_b * amount_b
            ).astype(weight_array_a.dtype)
            weight_interp = onph.from_array(weight_array_interp, weight_name)
            interp_weights_list.append(weight_interp)

        return interp_weights_list

    def check_will_upscale(self, model: OnnxModel):
        fake_img = np.ones((3, 3, 3), dtype=np.float32, order="F")
        result = OnnxImageUpscaleNode().run(fake_img, model, NO_TILING)

        mean_color = np.mean(result)
        del result
        return mean_color > 0.5

    def run(
        self,
        a: OnnxModel,
        b: OnnxModel,
        amount: int,
    ) -> Tuple[OnnxModel, int, int]:
        if amount == 0:
            return a, 100, 0
        elif amount == 100:
            return b, 0, 100

        # Just to be sure there is no mismatch from opt/un-opt models
        model_proto_a = onnx.load_from_string(a.bytes)
        model_proto_a = safely_optimize_onnx_model(model_proto_a)
        model_a_weights = model_proto_a.graph.initializer

        model_proto_b = onnx.load_from_string(b.bytes)
        model_proto_b = safely_optimize_onnx_model(model_proto_b)
        model_b_weights = model_proto_b.graph.initializer

        assert len(model_a_weights) == len(
            model_b_weights
        ), "Models must have same number of weights"

        logger.debug(f"Interpolating models...")
        interp_weights_list = self.perform_interp(
            model_a_weights, model_b_weights, amount
        )

        model_proto_interp = deepcopy(model_proto_b)
        for _ in range(len(model_proto_interp.graph.initializer)):  # type: ignore
            # Assigning a new value or assigning to field index do not seem to work
            model_proto_interp.graph.initializer.pop()  # type: ignore
        model_proto_interp.graph.initializer.extend(interp_weights_list)  # type: ignore
        model_interp = model_proto_interp.SerializeToString()  # type: ignore

        model = load_onnx_model(model_interp)
        if not self.check_will_upscale(model):
            raise ValueError(
                "These models are not compatible and not able to be interpolated together"
            )

        return model, 100 - amount, amount
