import json
import logging
import os
from PIL import Image
import safetensors
import torch

from utils.misc.sd import load_style_model, CLIPType, load_clip
from utils.misc.clip_vision import load
from utils.misc import node_helpers

class ReduxProcessor:
    """Handles Redux model processing."""

    def __init__(self, vae,
                 clip_vision_path: str = "models/clip_vision/sigclip_vision_patch14_384.safetensors",
                 style_model_path: str = "models/style_models/flux1-redux-dev.safetensors",
                 clip_path1: str = "models/clip/clip_l.safetensors",
                 clip_path2: str = "models/clip/t5xxl_fp16.safetensors"):
        # CLIP Vision model
        self.clip_vision = load(clip_vision_path)

        # Style model
        self.style_model = load_style_model(style_model_path)

        # Dual CLIP model
        self.clip = load_clip(ckpt_paths=[clip_path1, clip_path2], clip_type=CLIPType.FLUX)

        # VAE model
        self.vae = vae

    def encode(self, image, crop):
        '''Encode image using CLIP Vision model.'''
        crop_image = True
        if crop != "center":
            crop_image = False
        output = self.clip_vision.encode_image(image, crop=crop_image)
        return output
    
    def apply_stylemodel(self, clip_vision_output, conditioning, strength, strength_type):
        '''Apply style model to image.'''
        cond = self.style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        if strength_type == "multiply":
            cond *= strength

        c = []
        for t in conditioning:
            n = [torch.cat((t[0], cond), dim=1), t[1].copy()]
            c.append(n)
        return c

    
    def inpaint_model_conditioning(self, positive, negative, pixels, mask, noise_mask=True):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

        orig_pixels = pixels
        pixels = orig_pixels.clone()
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
            mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]

        m = (1.0 - mask.round()).squeeze(1)
        for i in range(3):
            pixels[:,:,:,i] -= 0.5
            pixels[:,:,:,i] *= m
            pixels[:,:,:,i] += 0.5
        concat_latent = self.vae.encode(pixels)
        orig_latent = self.vae.encode(orig_pixels)

        out_latent = {}

        out_latent["samples"] = orig_latent
        if noise_mask:
            out_latent["noise_mask"] = mask

        out = []
        for conditioning in [positive, negative]:
            c = node_helpers.conditioning_set_values(conditioning, {"concat_latent_image": concat_latent,
                                                                    "concat_mask": mask})
            out.append(c)
        return (out[0], out[1], out_latent)
    
    def apply_redux(self, style_image: torch.Tensor, merged_image: torch.Tensor,
                    merged_mask: torch.Tensor, text_prompt: str = "",
                    negative_text_prompt: str = "",
                    text_guidance: int = 3.5) -> torch.Tensor:
        """Apply Redux model to image."""
        # Encode image
        clip_vision_output = self.encode(style_image, crop="center")

        # Encode text prompt
        tokens = self.clip.tokenize(text_prompt)
        text_conditioning = self.clip.encode_from_tokens_scheduled(tokens)

        # Encode negative text prompt
        tokens = self.clip.tokenize(negative_text_prompt)
        negative_text_conditioning = self.clip.encode_from_tokens_scheduled(tokens)

        # Apply guidance
        text_conditioning = node_helpers.conditioning_set_values(text_conditioning, {"guidance": text_guidance})

        conditioning = self.apply_stylemodel(clip_vision_output, text_conditioning, 10.0, "multiply")

        return self.inpaint_model_conditioning(conditioning, negative_text_conditioning, merged_image, merged_mask)

    