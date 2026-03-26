from __future__ import annotations

import sys
from typing import Protocol, Sequence

import torch
from PIL import Image


class Encoder(Protocol):
    model_name: str

    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        ...

    def encode_text(self, text: str) -> torch.Tensor:
        ...


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / tensor.norm(dim=-1, keepdim=True)


class HuggingFaceEncoder:
    def __init__(self, model_name: str, stream = None) -> None:
        self.model_name = model_name
        self._stream = stream or sys.stderr
        model_cls, processor_cls = self._resolve_classes(model_name)

        print(
            "Loading model (the first run may download files from HuggingFace)...",
            file=self._stream,
            flush=True,
        )
        self._model = model_cls.from_pretrained(model_name)
        self._processor = processor_cls.from_pretrained(model_name)
        self._model.eval()
        print("Model loaded", file=self._stream, flush=True)

    @staticmethod
    def _resolve_classes(model_name: str):
        lowered = model_name.lower()
        if "chinese-clip" in lowered:
            from transformers import ChineseCLIPModel, ChineseCLIPProcessor

            return ChineseCLIPModel, ChineseCLIPProcessor

        from transformers import CLIPModel, CLIPProcessor

        return CLIPModel, CLIPProcessor

    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = self._processor(images=list(images), return_tensors="pt")
        with torch.no_grad():
            output = self._model.get_image_features(pixel_values=inputs["pixel_values"])
        if not isinstance(output, torch.Tensor):
            output = output.pooler_output
        return _normalize(output)

    def encode_text(self, text: str) -> torch.Tensor:
        inputs = self._processor(text=text, return_tensors="pt", padding=True)
        with torch.no_grad():
            output = self._model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        if not isinstance(output, torch.Tensor):
            output = output.pooler_output
        return _normalize(output)


def create_encoder(model_name: str, stream = None) -> Encoder:
    return HuggingFaceEncoder(model_name=model_name, stream=stream)
