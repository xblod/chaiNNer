from __future__ import annotations

from typing import Any

from ...impl.pytorch.architecture.Swin2SR import Swin2SR
from ...impl.pytorch.architecture.SwinIR import SwinIR
from ...impl.pytorch.types import PyTorchSRModel
from ...node_base import NodeBase
from ...node_factory import NodeFactory
from ...properties.inputs import OnnxFpDropdown, SrModelInput
from ...properties.outputs import NcnnModelOutput, TextOutput
from . import category as PyTorchCategory
from .convert_to_onnx import ConvertTorchToONNXNode

try:
    from ..onnx.convert_to_ncnn import FP_MODE_32, ConvertOnnxToNcnnNode
except:
    ConvertOnnxToNcnnNode = None


@NodeFactory.register("chainner:pytorch:convert_to_ncnn")
class ConvertTorchToNCNNNode(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = """Convert a PyTorch model to NCNN. Internally, this node uses ONNX as an intermediate format."""
        self.inputs = [
            SrModelInput("PyTorch Model"),
            OnnxFpDropdown(),
        ]
        self.outputs = [
            NcnnModelOutput(label="NCNN Model"),
            TextOutput("FP Mode", "FpMode::toString(Input1)"),
        ]

        self.category = PyTorchCategory
        self.name = "Convert To NCNN"
        self.icon = "NCNN"
        self.sub = "Utility"

    def run(self, model: PyTorchSRModel, is_fp16: int) -> Any:
        if ConvertOnnxToNcnnNode is None:
            raise ModuleNotFoundError(
                "Converting to NCNN is done through ONNX as an intermediate format (PyTorch -> ONNX -> NCNN), \
                and therefore requires the ONNX dependency to be installed. Please install ONNX through the dependency \
                manager to use this node."
            )

        assert not isinstance(
            model, SwinIR
        ), "SwinIR is not supported for NCNN conversion at this time."

        assert not isinstance(
            model, Swin2SR
        ), "Swin2SR is not supported for NCNN conversion at this time."

        # Intermediate conversion to ONNX is always fp32
        onnx_model = ConvertTorchToONNXNode().run(model, FP_MODE_32)[0]
        ncnn_model, fp_mode = ConvertOnnxToNcnnNode().run(onnx_model, is_fp16)

        return ncnn_model, fp_mode
