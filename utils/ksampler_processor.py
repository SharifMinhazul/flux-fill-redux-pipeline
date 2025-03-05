import torch
from PIL import Image

import utils.misc.sample
import utils.misc.utility
from utils.misc.sd import load_diffusion_model
from utils.misc import latent_preview

class KSamplerProcessor:
    """Handles KSampler model processing."""
    def __init__(self, vae, weight_dtype, unet_path: str = "models/unet/flux1-fill-dev.safetensors"):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        self.model = load_diffusion_model(unet_path, model_options=model_options)
        self.vae = vae
    
    def common_ksampler(self, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
        latent_image = latent["samples"]
        latent_image = utils.misc.sample.fix_empty_latent_channels(self.model, latent_image)

        if disable_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = utils.misc.sample.prepare_noise(latent_image, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        callback = latent_preview.prepare_callback(self.model, steps)
        disable_pbar = not utils.misc.utility.PROGRESS_BAR_ENABLED
        samples = utils.misc.sample.sample(self.model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                    denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                    force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
        out = latent.copy()
        out["samples"] = samples
        return out

    def apply_ksampler(self, seed, steps, cfg, sampler_name, scheduler, positive,
                       negative, latent, denoise=1.0) -> Image:
        """Pass everything through KSampler with Flux Fill Dev."""
        with torch.no_grad():
            samples = self.common_ksampler(seed, steps, cfg, sampler_name, scheduler,
                                       positive, negative, latent, denoise)

        # Decode latent samples
        with torch.no_grad():
            images = self.vae.decode(samples["samples"])    
        if len(images.shape) == 5: #Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2],
                                    images.shape[-1])
        return images