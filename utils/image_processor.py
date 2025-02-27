import math
from PIL import Image
from fastapi import UploadFile
import numpy as np
import torch
import torchvision.transforms.functional as F
from scipy.ndimage import gaussian_filter, grey_dilation, binary_fill_holes, binary_closing

from utils.misc.utility import pil2tensor, tensor2pil, lanczos

MAX_RESOLUTION=16384

def rescale(samples, width, height, algorithm: str):
    if algorithm == "bislerp":  # convert for compatibility with old workflows
        algorithm = "bicubic"
    algorithm = getattr(Image, algorithm.upper())  # i.e. Image.BICUBIC
    samples_pil: Image.Image = F.to_pil_image(samples[0].cpu()).resize((width, height), algorithm)
    samples = F.to_tensor(samples_pil).unsqueeze(0)
    return samples

class ImageProcessor:
    """Handles image processing steps."""
    
    @staticmethod
    def load_image(upload_file: UploadFile | str, mode: str = "RGB") -> Image:
        if isinstance(upload_file, str):
            return Image.open(upload_file).convert(mode)
        """Load an image from an upload file."""
        return Image.open(upload_file.file).convert(mode)
    
    @staticmethod
    def resize(image, width, height, method="stretch", interpolation="nearest", condition="always", multiple_of=0, keep_proportion=False):
        if isinstance(image, Image.Image):
            image = pil2tensor(image)
 
        _, oh, ow, _ = image.shape
        x = y = x2 = y2 = 0
        pad_left = pad_right = pad_top = pad_bottom = 0

        if keep_proportion:
            method = "keep proportion"

        if multiple_of > 1:
            width = width - (width % multiple_of)
            height = height - (height % multiple_of)

        if method == 'keep proportion' or method == 'pad':
            if width == 0 and oh < height:
                width = MAX_RESOLUTION
            elif width == 0 and oh >= height:
                width = ow

            if height == 0 and ow < width:
                height = MAX_RESOLUTION
            elif height == 0 and ow >= width:
                height = oh

            ratio = min(width / ow, height / oh)
            new_width = round(ow*ratio)
            new_height = round(oh*ratio)

            if method == 'pad':
                pad_left = (width - new_width) // 2
                pad_right = width - new_width - pad_left
                pad_top = (height - new_height) // 2
                pad_bottom = height - new_height - pad_top

            width = new_width
            height = new_height
        elif method.startswith('fill'):
            width = width if width > 0 else ow
            height = height if height > 0 else oh

            ratio = max(width / ow, height / oh)
            new_width = round(ow*ratio)
            new_height = round(oh*ratio)
            x = (new_width - width) // 2
            y = (new_height - height) // 2
            x2 = x + width
            y2 = y + height
            if x2 > new_width:
                x -= (x2 - new_width)
            if x < 0:
                x = 0
            if y2 > new_height:
                y -= (y2 - new_height)
            if y < 0:
                y = 0
            width = new_width
            height = new_height
        else:
            width = width if width > 0 else ow
            height = height if height > 0 else oh

        if "always" in condition \
            or ("downscale if bigger" == condition and (oh > height or ow > width)) or ("upscale if smaller" == condition and (oh < height or ow < width)) \
            or ("bigger area" in condition and (oh * ow > height * width)) or ("smaller area" in condition and (oh * ow < height * width)):

            outputs = image.permute(0,3,1,2)

            if interpolation == "lanczos":
                outputs = lanczos(outputs, width, height)
            else:
                outputs = F.interpolate(outputs, size=(height, width), mode=interpolation)

            if method == 'pad':
                if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                    outputs = F.pad(outputs, (pad_left, pad_right, pad_top, pad_bottom), value=0)

            outputs = outputs.permute(0,2,3,1)

            if method.startswith('fill'):
                if x > 0 or y > 0 or x2 > 0 or y2 > 0:
                    outputs = outputs[:, y:y2, x:x2, :]
        else:
            outputs = image

        if multiple_of > 1 and (outputs.shape[2] % multiple_of != 0 or outputs.shape[1] % multiple_of != 0):
            width = outputs.shape[2]
            height = outputs.shape[1]
            x = (width % multiple_of) // 2
            y = (height % multiple_of) // 2
            x2 = width - ((width % multiple_of) - x)
            y2 = height - ((height % multiple_of) - y)
            outputs = outputs[:, y:y2, x:x2, :]
        
        outputs = torch.clamp(outputs, 0, 1)

        return(outputs, outputs.shape[2], outputs.shape[1],)

    @staticmethod
    def save_image(image, path: str):
        if isinstance(image, torch.Tensor):
            image = F.to_pil_image(image[0].cpu())
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image.save(path)

    @staticmethod
    def get_image_name_without_extension(image: Image) -> str:
        return ".".join(image.filename.split(".")[:-1])

    @staticmethod
    def grow_and_blur_mask(mask, blur_pixels):
        if blur_pixels > 0.001:
            sigma = blur_pixels / 4
            growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
            out = []
            for m in growmask:
                mask_np = m.numpy()
                kernel_size = math.ceil(sigma * 1.5 + 1)
                kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                dilated_mask = grey_dilation(mask_np, footprint=kernel)
                output = dilated_mask.astype(np.float32) * 255
                output = torch.from_numpy(output)
                out.append(output)
            mask = torch.stack(out, dim=0)
            mask = torch.clamp(mask, 0.0, 1.0)

            mask_np = mask.numpy()
            filtered_mask = gaussian_filter(mask_np, sigma=sigma)
            mask = torch.from_numpy(filtered_mask)
            mask = torch.clamp(mask, 0.0, 1.0)
        
        return mask

    @staticmethod
    def adjust_to_aspect_ratio(x_min, x_max, y_min, y_max, width, height, target_width, target_height):
        x_min_key, x_max_key, y_min_key, y_max_key = x_min, x_max, y_min, y_max

        # Calculate the current width and height
        current_width = x_max - x_min + 1
        current_height = y_max - y_min + 1

        # Calculate aspect ratios
        aspect_ratio = target_width / target_height
        current_aspect_ratio = current_width / current_height

        if current_aspect_ratio < aspect_ratio:
            # Adjust width to match target aspect ratio
            new_width = int(current_height * aspect_ratio)
            extend_x = (new_width - current_width)
            x_min = max(x_min - extend_x//2, 0)
            x_max = min(x_max + extend_x//2, width - 1)
        else:
            # Adjust height to match target aspect ratio
            new_height = int(current_width / aspect_ratio)
            extend_y = (new_height - current_height)
            y_min = max(y_min - extend_y//2, 0)
            y_max = min(y_max + extend_y//2, height - 1)

        return int(x_min), int(x_max), int(y_min), int(y_max)

    @staticmethod
    def adjust_to_preferred(x_min, x_max, y_min, y_max, width, height, preferred_x_start, preferred_x_end, preferred_y_start, preferred_y_end):
        # Ensure the area is within preferred bounds as much as possible
        if preferred_x_start <= x_min and preferred_x_end >= x_max and preferred_y_start <= y_min and preferred_y_end >= y_max:
            return x_min, x_max, y_min, y_max

        # Shift x_min and x_max to fit within preferred bounds if possible
        if x_max - x_min + 1 <= preferred_x_end - preferred_x_start + 1:
            if x_min < preferred_x_start:
                x_shift = preferred_x_start - x_min
                x_min += x_shift
                x_max += x_shift
            elif x_max > preferred_x_end:
                x_shift = x_max - preferred_x_end
                x_min -= x_shift
                x_max -= x_shift

        # Shift y_min and y_max to fit within preferred bounds if possible
        if y_max - y_min + 1 <= preferred_y_end - preferred_y_start + 1:
            if y_min < preferred_y_start:
                y_shift = preferred_y_start - y_min
                y_min += y_shift
                y_max += y_shift
            elif y_max > preferred_y_end:
                y_shift = y_max - preferred_y_end
                y_min -= y_shift
                y_max -= y_shift

        return int(x_min), int(x_max), int(y_min), int(y_max)

    @staticmethod
    def apply_padding(min_val, max_val, max_boundary, padding):
        # Calculate the midpoint and the original range size
        original_range_size = max_val - min_val + 1
        midpoint = (min_val + max_val) // 2

        # Determine the smallest multiple of padding that is >= original_range_size
        if original_range_size % padding == 0:
            new_range_size = original_range_size
        else:
            new_range_size = (original_range_size // padding + 1) * padding

        # Calculate the new min and max values centered on the midpoint
        new_min_val = max(midpoint - new_range_size // 2, 0)
        new_max_val = new_min_val + new_range_size - 1

        # Ensure the new max doesn't exceed the boundary
        if new_max_val >= max_boundary:
            new_max_val = max_boundary - 1
            new_min_val = max(new_max_val - new_range_size + 1, 0)

        # Ensure the range still ends on a multiple of padding
        # Adjust if the calculated range isn't feasible within the given constraints
        if (new_max_val - new_min_val + 1) != new_range_size:
            new_min_val = max(new_max_val - new_range_size + 1, 0)

        return new_min_val, new_max_val

    @staticmethod
    def inpaint_crop(image, mask, context_expand_pixels, context_expand_factor, fill_mask_holes, blur_mask_pixels, invert_mask, blend_pixels, mode, rescale_algorithm, force_width, force_height, rescale_factor, padding, min_width, min_height, max_width, max_height, optional_context_mask=None):
        if image.shape[0] > 1:
            assert mode == "forced size", "Mode must be 'forced size' when input is a batch of images"
        assert image.shape[0] == mask.shape[0], "Batch size of images and masks must be the same"
        if optional_context_mask is not None:
            assert optional_context_mask.shape[0] == image.shape[0], "Batch size of optional_context_masks must be the same as images or None"

        result_stitch = {'x': [], 'y': [], 'original_image': [], 'cropped_mask_blend': [], 'rescale_x': [], 'rescale_y': [], 'start_x': [], 'start_y': [], 'initial_width': [], 'initial_height': []}
        results_image = []
        results_mask = []

        batch_size = image.shape[0]
        for b in range(batch_size):
            one_image = image[b].unsqueeze(0)
            one_mask = mask[b].unsqueeze(0)
            one_optional_context_mask = None
            if optional_context_mask is not None:
                one_optional_context_mask = optional_context_mask[b].unsqueeze(0)

            stitch, cropped_image, cropped_mask = IsADirectoryErrornpaint_crop_single_image(one_image, one_mask, context_expand_pixels, context_expand_factor, fill_mask_holes, blur_mask_pixels, invert_mask, blend_pixels, mode, rescale_algorithm, force_width, force_height, rescale_factor, padding, min_width, min_height, max_width, max_height, one_optional_context_mask)

            for key in result_stitch:
                result_stitch[key].append(stitch[key])
            cropped_image = cropped_image.squeeze(0)
            results_image.append(cropped_image)
            cropped_mask = cropped_mask.squeeze(0)
            results_mask.append(cropped_mask)

        result_image = torch.stack(results_image, dim=0)
        result_mask = torch.stack(results_mask, dim=0)

        return result_stitch, result_image, result_mask
       
    @staticmethod
    # Parts of this function are from KJNodes: https://github.com/kijai/ComfyUI-KJNodes
    def inpaint_crop_single_image(image, mask, context_expand_pixels, context_expand_factor, fill_mask_holes, blur_mask_pixels, invert_mask, blend_pixels, mode, rescale_algorithm, force_width, force_height, rescale_factor, padding, min_width, min_height, max_width, max_height, optional_context_mask=None):
        #Validate or initialize mask
        if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
            non_zero_indices = torch.nonzero(mask[0], as_tuple=True)
            if not non_zero_indices[0].size(0):
                mask = torch.zeros_like(image[:, :, :, 0])
            else:
                assert False, "mask size must match image size"

        # Fill holes if requested
        if fill_mask_holes:
            holemask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
            out = []
            for m in holemask:
                mask_np = m.numpy()
                binary_mask = mask_np > 0
                struct = np.ones((5, 5))
                closed_mask = binary_closing(binary_mask, structure=struct, border_value=1)
                filled_mask = binary_fill_holes(closed_mask)
                output = filled_mask.astype(np.float32) * 255
                output = torch.from_numpy(output)
                out.append(output)
            mask = torch.stack(out, dim=0)
            mask = torch.clamp(mask, 0.0, 1.0)

        # Grow and blur mask if requested
        if blur_mask_pixels > 0.001:
            mask = row_and_blur_mask(mask, blur_mask_pixels)

        # Invert mask if requested
        if invert_mask:
            mask = 1.0 - mask

        # Validate or initialize context mask
        if optional_context_mask is None:
            context_mask = mask
        elif optional_context_mask.shape[1] != image.shape[1] or optional_context_mask.shape[2] != image.shape[2]:
            non_zero_indices = torch.nonzero(optional_context_mask[0], as_tuple=True)
            if not non_zero_indices[0].size(0):
                context_mask = mask
            else:
                assert False, "context_mask size must match image size"
        else:
            context_mask = optional_context_mask + mask 
            context_mask = torch.clamp(context_mask, 0.0, 1.0)

        # Ensure mask dimensions match image dimensions except channels
        initial_batch, initial_height, initial_width, initial_channels = image.shape
        mask_batch, mask_height, mask_width = mask.shape
        context_mask_batch, context_mask_height, context_mask_width = context_mask.shape
        assert initial_height == mask_height and initial_width == mask_width, "Image and mask dimensions must match"
        assert initial_height == context_mask_height and initial_width == context_mask_width, "Image and context mask dimensions must match"

        # Extend image and masks to turn it into a big square in case the context area would go off bounds
        extend_y = (initial_width + 1) // 2 # Intended, extend height by width (turn into square)
        extend_x = (initial_height + 1) // 2 # Intended, extend width by height (turn into square)
        new_height = initial_height + 2 * extend_y
        new_width = initial_width + 2 * extend_x

        start_y = extend_y
        start_x = extend_x

        available_top = min(start_y, initial_height)
        available_bottom = min(new_height - (start_y + initial_height), initial_height)
        available_left = min(start_x, initial_width)
        available_right = min(new_width - (start_x + initial_width), initial_width)

        new_image = torch.zeros((initial_batch, new_height, new_width, initial_channels), dtype=image.dtype)
        new_image[:, start_y:start_y + initial_height, start_x:start_x + initial_width, :] = image
        # Mirror image so there's no bleeding of black border when using inpaintmodelconditioning
        # Top
        new_image[:, start_y - available_top:start_y, start_x:start_x + initial_width, :] = torch.flip(image[:, :available_top, :, :], [1])
        # Bottom
        new_image[:, start_y + initial_height:start_y + initial_height + available_bottom, start_x:start_x + initial_width, :] = torch.flip(image[:, -available_bottom:, :, :], [1])
        # Left
        new_image[:, start_y:start_y + initial_height, start_x - available_left:start_x, :] = torch.flip(new_image[:, start_y:start_y + initial_height, start_x:start_x + available_left, :], [2])
        # Right
        new_image[:, start_y:start_y + initial_height, start_x + initial_width:start_x + initial_width + available_right, :] = torch.flip(new_image[:, start_y:start_y + initial_height, start_x + initial_width - available_right:start_x + initial_width, :], [2])
        # Top-left corner
        new_image[:, start_y - available_top:start_y, start_x - available_left:start_x, :] = torch.flip(new_image[:, start_y:start_y + available_top, start_x:start_x + available_left, :], [1, 2])
        # Top-right corner
        new_image[:, start_y - available_top:start_y, start_x + initial_width:start_x + initial_width + available_right, :] = torch.flip(new_image[:, start_y:start_y + available_top, start_x + initial_width - available_right:start_x + initial_width, :], [1, 2])
        # Bottom-left corner
        new_image[:, start_y + initial_height:start_y + initial_height + available_bottom, start_x - available_left:start_x, :] = torch.flip(new_image[:, start_y + initial_height - available_bottom:start_y + initial_height, start_x:start_x + available_left, :], [1, 2])
        # Bottom-right corner
        new_image[:, start_y + initial_height:start_y + initial_height + available_bottom, start_x + initial_width:start_x + initial_width + available_right, :] = torch.flip(new_image[:, start_y + initial_height - available_bottom:start_y + initial_height, start_x + initial_width - available_right:start_x + initial_width, :], [1, 2])

        new_mask = torch.ones((mask_batch, new_height, new_width), dtype=mask.dtype) # assume ones in extended image
        new_mask[:, start_y:start_y + initial_height, start_x:start_x + initial_width] = mask

        blend_mask = torch.zeros((mask_batch, new_height, new_width), dtype=mask.dtype) # assume zeros in extended image
        blend_mask[:, start_y:start_y + initial_height, start_x:start_x + initial_width] = mask
        # Mirror blend mask so there's no bleeding of border when blending
        # Top
        blend_mask[:, start_y - available_top:start_y, start_x:start_x + initial_width] = torch.flip(mask[:, :available_top, :], [1])
        # Bottom
        blend_mask[:, start_y + initial_height:start_y + initial_height + available_bottom, start_x:start_x + initial_width] = torch.flip(mask[:, -available_bottom:, :], [1])
        # Left
        blend_mask[:, start_y:start_y + initial_height, start_x - available_left:start_x] = torch.flip(blend_mask[:, start_y:start_y + initial_height, start_x:start_x + available_left], [2])
        # Right
        blend_mask[:, start_y:start_y + initial_height, start_x + initial_width:start_x + initial_width + available_right] = torch.flip(blend_mask[:, start_y:start_y + initial_height, start_x + initial_width - available_right:start_x + initial_width], [2])
        # Top-left corner
        blend_mask[:, start_y - available_top:start_y, start_x - available_left:start_x] = torch.flip(blend_mask[:, start_y:start_y + available_top, start_x:start_x + available_left], [1, 2])
        # Top-right corner
        blend_mask[:, start_y - available_top:start_y, start_x + initial_width:start_x + initial_width + available_right] = torch.flip(blend_mask[:, start_y:start_y + available_top, start_x + initial_width - available_right:start_x + initial_width], [1, 2])
        # Bottom-left corner
        blend_mask[:, start_y + initial_height:start_y + initial_height + available_bottom, start_x - available_left:start_x] = torch.flip(blend_mask[:, start_y + initial_height - available_bottom:start_y + initial_height, start_x:start_x + available_left], [1, 2])
        # Bottom-right corner
        blend_mask[:, start_y + initial_height:start_y + initial_height + available_bottom, start_x + initial_width:start_x + initial_width + available_right] = torch.flip(blend_mask[:, start_y + initial_height - available_bottom:start_y + initial_height, start_x + initial_width - available_right:start_x + initial_width], [1, 2])

        new_context_mask = torch.zeros((mask_batch, new_height, new_width), dtype=context_mask.dtype)
        new_context_mask[:, start_y:start_y + initial_height, start_x:start_x + initial_width] = context_mask

        image = new_image
        mask = new_mask
        context_mask = new_context_mask

        original_image = image
        original_mask = mask
        original_width = image.shape[2]
        original_height = image.shape[1]

        # If there are no non-zero indices in the context_mask, adjust context mask to the whole image
        non_zero_indices = torch.nonzero(context_mask[0], as_tuple=True)
        if not non_zero_indices[0].size(0):
            context_mask = torch.ones_like(image[:, :, :, 0])
            context_mask = torch.zeros((mask_batch, new_height, new_width), dtype=mask.dtype)
            context_mask[:, start_y:start_y + initial_height, start_x:start_x + initial_width] += 1.0
            non_zero_indices = torch.nonzero(context_mask[0], as_tuple=True)

        # Compute context area from context mask
        y_min = torch.min(non_zero_indices[0]).item()
        y_max = torch.max(non_zero_indices[0]).item()
        x_min = torch.min(non_zero_indices[1]).item()
        x_max = torch.max(non_zero_indices[1]).item()
        height = context_mask.shape[1]
        width = context_mask.shape[2]
        
        # Grow context area if requested
        y_size = y_max - y_min + 1
        x_size = x_max - x_min + 1
        y_grow = round(max(y_size*(context_expand_factor-1), context_expand_pixels, blend_pixels**1.5))
        x_grow = round(max(x_size*(context_expand_factor-1), context_expand_pixels, blend_pixels**1.5))
        y_min = max(y_min - y_grow // 2, 0)
        y_max = min(y_max + y_grow // 2, height - 1)
        x_min = max(x_min - x_grow // 2, 0)
        x_max = min(x_max + x_grow // 2, width - 1)
        y_size = y_max - y_min + 1
        x_size = x_max - x_min + 1

        effective_upscale_factor_x = 1.0
        effective_upscale_factor_y = 1.0

        # Adjust to preferred size
        if mode == 'forced size':
            #Sub case of ranged size.
            min_width = max_width = force_width
            min_height = max_height = force_height

        if mode == 'ranged size' or mode == 'forced size':
            assert max_width >= min_width, "max_width must be greater than or equal to min_width"
            assert max_height >= min_height, "max_height must be greater than or equal to min_height"
            # Ensure we set an aspect ratio supported by min_width, max_width, min_height, max_height
            current_width = x_max - x_min + 1
            current_height = y_max - y_min + 1
        
            # Calculate aspect ratio of the selected area
            current_aspect_ratio = current_width / current_height

            # Calculate the aspect ratio bounds
            min_aspect_ratio = min_width / max_height
            max_aspect_ratio = max_width / min_height

            # Adjust target width and height based on aspect ratio bounds
            if current_aspect_ratio < min_aspect_ratio:
                # Adjust to meet minimum width constraint
                target_width = min(current_width, min_width)
                target_height = int(target_width / min_aspect_ratio)
                x_min, x_max, y_min, y_max = djust_to_aspect_ratio(x_min, x_max, y_min, y_max, width, height, target_width, target_height)
                x_min, x_max, y_min, y_max = djust_to_preferred(x_min, x_max, y_min, y_max, width, height, start_x, start_x+initial_width, start_y, start_y+initial_height)
            elif current_aspect_ratio > max_aspect_ratio:
                # Adjust to meet maximum width constraint
                target_height = min(current_height, max_height)
                target_width = int(target_height * max_aspect_ratio)
                x_min, x_max, y_min, y_max = djust_to_aspect_ratio(x_min, x_max, y_min, y_max, width, height, target_width, target_height)
                x_min, x_max, y_min, y_max = djust_to_preferred(x_min, x_max, y_min, y_max, width, height, start_x, start_x+initial_width, start_y, start_y+initial_height)
            else:
                # Aspect ratio is within bounds, keep the current size
                target_width = current_width
                target_height = current_height

            y_size = y_max - y_min + 1
            x_size = x_max - x_min + 1

            # Adjust to min and max sizes
            max_rescale_width = max_width / x_size
            max_rescale_height = max_height / y_size
            max_rescale_factor = min(max_rescale_width, max_rescale_height)
            rescale_factor = max_rescale_factor
            min_rescale_width = min_width / x_size
            min_rescale_height = min_height / y_size
            min_rescale_factor = min(min_rescale_width, min_rescale_height)
            rescale_factor = max(min_rescale_factor, rescale_factor)

        # Upscale image and masks if requested, they will be downsized at stitch phase
        if rescale_factor < 0.999 or rescale_factor > 1.001:
            samples = image            
            samples = samples.movedim(-1, 1)
            width = round(samples.shape[3] * rescale_factor)
            height = round(samples.shape[2] * rescale_factor)
            samples = rescale(samples, width, height, rescale_algorithm)
            effective_upscale_factor_x = float(width)/float(original_width)
            effective_upscale_factor_y = float(height)/float(original_height)
            samples = samples.movedim(1, -1)
            image = samples

            samples = mask
            samples = samples.unsqueeze(1)
            samples = rescale(samples, width, height, "nearest")
            samples = samples.squeeze(1)
            mask = samples

            samples = blend_mask
            samples = samples.unsqueeze(1)
            samples = rescale(samples, width, height, "nearest")
            samples = samples.squeeze(1)
            blend_mask = samples

            # Do math based on min,size instead of min,max to avoid rounding errors
            y_size = y_max - y_min + 1
            x_size = x_max - x_min + 1
            target_x_size = int(x_size * effective_upscale_factor_x)
            target_y_size = int(y_size * effective_upscale_factor_y)

            x_min = round(x_min * effective_upscale_factor_x)
            x_max = x_min + target_x_size
            y_min = round(y_min * effective_upscale_factor_y)
            y_max = y_min + target_y_size

        x_size = x_max - x_min + 1
        y_size = y_max - y_min + 1

        # Ensure width and height are within specified bounds, key for ranged and forced size
        if mode == 'ranged size' or mode == 'forced size':
            if x_size < min_width:
                x_max = min(x_max + (min_width - x_size), width - 1)
            elif x_size > max_width:
                x_max = x_min + max_width - 1
    
            if y_size < min_height:
                y_max = min(y_max + (min_height - y_size), height - 1)
            elif y_size > max_height:
                y_max = y_min + max_height - 1

        # Recalculate x_size and y_size after adjustments
        x_size = x_max - x_min + 1
        y_size = y_max - y_min + 1

        # Pad area (if possible, i.e. if pad is smaller than width/height) to avoid the sampler returning smaller results
        if (mode == 'free size' or mode == 'ranged size') and padding > 1:
            x_min, x_max = pply_padding(x_min, x_max, width, padding)
            y_min, y_max = pply_padding(y_min, y_max, height, padding)

        # Ensure that context area doesn't go outside of the image
        x_min = max(x_min, 0)
        x_max = min(x_max, width - 1)
        y_min = max(y_min, 0)
        y_max = min(y_max, height - 1)

        # Crop the image and the mask, sized context area
        cropped_image = image[:, y_min:y_max+1, x_min:x_max+1]
        cropped_mask = mask[:, y_min:y_max+1, x_min:x_max+1]
        cropped_mask_blend = blend_mask[:, y_min:y_max+1, x_min:x_max+1]

        # Grow and blur mask for blend if requested
        if blend_pixels > 0.001:
            cropped_mask_blend = row_and_blur_mask(cropped_mask_blend, blend_pixels)

        # Return stitch (to be consumed by the class below), image, and mask
        stitch = {'x': x_min, 'y': y_min, 'original_image': original_image, 'cropped_mask_blend': cropped_mask_blend, 'rescale_x': effective_upscale_factor_x, 'rescale_y': effective_upscale_factor_y, 'start_x': start_x, 'start_y': start_y, 'initial_width': initial_width, 'initial_height': initial_height}

        return (stitch, cropped_image, cropped_mask)
