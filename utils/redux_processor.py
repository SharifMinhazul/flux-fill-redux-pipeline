import json
import logging
import os
from PIL import Image
import safetensors
import torch

from utils.misc.clip_vision import load

class ReduxProcessor:
    """Handles Redux model processing."""

    def __init__(self, clip_path: str):
        self.clip_vision = load(clip_path)

    def encode(self, image, crop):
        crop_image = True
        if crop != "center":
            crop_image = False
        output = self.clip_vision.encode_image(image, crop=crop_image)
        return (output,)
    
    def apply_redux(self, image: torch.Tensor) -> torch.Tensor:
        """Apply Redux model to image."""
        with torch.no_grad():
            return self.encode(image, crop=True)
    