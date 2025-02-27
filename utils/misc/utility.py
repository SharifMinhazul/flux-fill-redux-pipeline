import torch
import numpy as np
from PIL import Image
from typing import Union, List

# Utility functions from mtb nodes: https://github.com/melMass/comfy_mtb
def pil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)

    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def np2tensor(img_np: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor:
    if isinstance(img_np, list):
        return torch.cat([np2tensor(img) for img in img_np], dim=0)

    return torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0)


def tensor2np(tensor: torch.Tensor):
    if len(tensor.shape) == 3:  # Single image
        return np.clip(255.0 * tensor.cpu().numpy(), 0, 255).astype(np.uint8)
    else:  # Batch of images
        return [np.clip(255.0 * t.cpu().numpy(), 0, 255).astype(np.uint8) for t in tensor]
    
def tensor2pil(image: torch.Tensor) -> List[Image.Image]:
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image[i]))
        return out

    return [
        Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )
    ]

def bislerp(samples, width, height):
    def slerp(b1, b2, r):
        '''slerps batches b1, b2 according to ratio r, batches should be flat e.g. NxC'''
        
        c = b1.shape[-1]

        #norms
        b1_norms = torch.norm(b1, dim=-1, keepdim=True)
        b2_norms = torch.norm(b2, dim=-1, keepdim=True)

        #normalize
        b1_normalized = b1 / b1_norms
        b2_normalized = b2 / b2_norms

        #zero when norms are zero
        b1_normalized[b1_norms.expand(-1,c) == 0.0] = 0.0
        b2_normalized[b2_norms.expand(-1,c) == 0.0] = 0.0

        #slerp
        dot = (b1_normalized*b2_normalized).sum(1)
        omega = torch.acos(dot)
        so = torch.sin(omega)

        #technically not mathematically correct, but more pleasing?
        res = (torch.sin((1.0-r.squeeze(1))*omega)/so).unsqueeze(1)*b1_normalized + (torch.sin(r.squeeze(1)*omega)/so).unsqueeze(1) * b2_normalized
        res *= (b1_norms * (1.0-r) + b2_norms * r).expand(-1,c)

        #edge cases for same or polar opposites
        res[dot > 1 - 1e-5] = b1[dot > 1 - 1e-5] 
        res[dot < 1e-5 - 1] = (b1 * (1.0-r) + b2 * r)[dot < 1e-5 - 1]
        return res
    
    def generate_bilinear_data(length_old, length_new, device):
        coords_1 = torch.arange(length_old, dtype=torch.float32, device=device).reshape((1,1,1,-1))
        coords_1 = torch.nn.functional.interpolate(coords_1, size=(1, length_new), mode="bilinear")
        ratios = coords_1 - coords_1.floor()
        coords_1 = coords_1.to(torch.int64)
        
        coords_2 = torch.arange(length_old, dtype=torch.float32, device=device).reshape((1,1,1,-1)) + 1
        coords_2[:,:,:,-1] -= 1
        coords_2 = torch.nn.functional.interpolate(coords_2, size=(1, length_new), mode="bilinear")
        coords_2 = coords_2.to(torch.int64)
        return ratios, coords_1, coords_2

    orig_dtype = samples.dtype
    samples = samples.float()
    n,c,h,w = samples.shape
    h_new, w_new = (height, width)
    
    #linear w
    ratios, coords_1, coords_2 = generate_bilinear_data(w, w_new, samples.device)
    coords_1 = coords_1.expand((n, c, h, -1))
    coords_2 = coords_2.expand((n, c, h, -1))
    ratios = ratios.expand((n, 1, h, -1))

    pass_1 = samples.gather(-1,coords_1).movedim(1, -1).reshape((-1,c))
    pass_2 = samples.gather(-1,coords_2).movedim(1, -1).reshape((-1,c))
    ratios = ratios.movedim(1, -1).reshape((-1,1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h, w_new, c).movedim(-1, 1)

    #linear h
    ratios, coords_1, coords_2 = generate_bilinear_data(h, h_new, samples.device)
    coords_1 = coords_1.reshape((1,1,-1,1)).expand((n, c, -1, w_new))
    coords_2 = coords_2.reshape((1,1,-1,1)).expand((n, c, -1, w_new))
    ratios = ratios.reshape((1,1,-1,1)).expand((n, 1, -1, w_new))

    pass_1 = result.gather(-2,coords_1).movedim(1, -1).reshape((-1,c))
    pass_2 = result.gather(-2,coords_2).movedim(1, -1).reshape((-1,c))
    ratios = ratios.movedim(1, -1).reshape((-1,1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h_new, w_new, c).movedim(-1, 1)
    return result.to(orig_dtype)

def lanczos(samples, width, height):
    images = [Image.fromarray(np.clip(255. * image.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8)) for image in samples]
    images = [image.resize((width, height), resample=Image.Resampling.LANCZOS) for image in images]
    images = [torch.from_numpy(np.array(image).astype(np.float32) / 255.0).movedim(-1, 0) for image in images]
    result = torch.stack(images)
    return result.to(samples.device, samples.dtype)