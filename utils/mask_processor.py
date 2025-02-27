import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageFilter
import scipy
import torch
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image, predict

from .misc.utility import tensor2pil, pil2tensor

class MaskProcessor:
    """Handles mask generation and modification."""
    def __init__(self, sam2_checkpoint: str = "/home/emon/Models/flux-redux-fill-pipeline/models/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt", 
                 sam2_model_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml", 
                 grounding_dino_checkpoint: str = "/home/emon/Models/flux-redux-fill-pipeline/models/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth", 
                 grounding_dino_config: str = "/home/emon/Models/flux-redux-fill-pipeline/models/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
                 box_threshold: float = 0.35, text_threshold: float = 0.25, device: str = None, mask_threshold: float = 0.5,
                 output_dir: Path = Path("outputs/grounded_sam2_local_demo"), dump_json_results: bool = True):
        self.SAM2_CHECKPOINT = sam2_checkpoint
        self.SAM2_MODEL_CONFIG = sam2_model_config
        self.GROUNDING_DINO_CONFIG = grounding_dino_config
        self.GROUNDING_DINO_CHECKPOINT = grounding_dino_checkpoint
        self.BOX_THRESHOLD = box_threshold
        self.TEXT_THRESHOLD = text_threshold
        self.MASK_THRESHOLD = mask_threshold
        self.DEVICE = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.OUTPUT_DIR = output_dir
        self.DUMP_JSON_RESULTS = dump_json_results

        # create output directory
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # print(torch.backends.cudnn.version())
        # os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"

        # build SAM2 image predictor
        sam2_checkpoint = self.SAM2_CHECKPOINT
        model_cfg = self.SAM2_MODEL_CONFIG
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.DEVICE)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        # build grounding dino model
        self.grounding_model = load_model(
            model_config_path=self.GROUNDING_DINO_CONFIG, 
            model_checkpoint_path=self.GROUNDING_DINO_CHECKPOINT,
            device=self.DEVICE
        )

    def generate_mask(self, image: Image, edit_location: str) -> Image:
        """Generate mask for the image."""

        # setup the input image and text prompt for SAM 2 and Grounding DINO
        # VERY important: text queries need to be lowercased + end with a dot
        text = edit_location.lower().strip()
        if not text.endswith("."):
            text += "."

        image.save("temp.png")

        image_source, image = load_image("temp.png")
        # Ensure the NumPy array is writable
        img_source = image_source.copy()

        self.sam2_predictor.set_image(img_source)
        boxes, confidences, labels = predict(
        model=self.grounding_model,
        image=image,
        caption=text,
        box_threshold=self.BOX_THRESHOLD,
        text_threshold=self.TEXT_THRESHOLD,
        )

        # process the box prompt for SAM 2
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # FIXME: figure how does this influence the G-DINO model
        # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        
        # convert mask to binary
        masks = masks > self.MASK_THRESHOLD

        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            # If there are multiple masks them merge into 1 single mask
            for i in range(1, masks.shape[0]):
                masks[0] = np.logical_or(masks[0], masks[i])
            masks = masks[0]
        
        masks = masks.squeeze(0)


        masks = masks.astype(np.uint8) * 255
        """
        Post-process the output of the model to get the masks, scores, and logits for visualization
        """

        confidences = confidences.numpy().tolist()
        class_names = labels

        class_ids = np.array(list(range(len(class_names))))

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]
        # return Image.new("L", image.size, 128)  # Placeholder
        # return Image.fromarray(masks, mode="L")
        return torch.from_numpy(masks).unsqueeze(0)

    @staticmethod
    def expand_mask(mask, expand, tapered_corners, flip_input, blur_radius, incremental_expandrate, lerp_alpha, decay_factor, fill_holes=False):
        alpha = lerp_alpha
        decay = decay_factor
        if flip_input:
            mask = 1.0 - mask
        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c],
                           [1, 1, 1],
                           [c, 1, c]])
        growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
        out = []
        previous_output = None
        current_expand = expand
        for m in growmask:
            output = m.numpy().astype(np.float32)
            for _ in range(abs(round(current_expand))):
                if current_expand < 0:
                    output = scipy.ndimage.grey_erosion(output, footprint=kernel)
                else:
                    output = scipy.ndimage.grey_dilation(output, footprint=kernel)
            if current_expand < 0:
                current_expand -= abs(incremental_expandrate)
            else:
                current_expand += abs(incremental_expandrate)
            if fill_holes:
                binary_mask = output > 0
                output = scipy.ndimage.binary_fill_holes(binary_mask)
                output = output.astype(np.float32) * 255
            output = torch.from_numpy(output)
            if alpha < 1.0 and previous_output is not None:
                # Interpolate between the previous and current frame
                output = alpha * output + (1 - alpha) * previous_output
            if decay < 1.0 and previous_output is not None:
                # Add the decayed previous output to the current frame
                output += decay * previous_output
                output = output / output.max()
            previous_output = output
            out.append(output)

        if blur_radius != 0:
            # Convert the tensor list to PIL images, apply blur, and convert back
            for idx, tensor in enumerate(out):
                # Convert tensor to PIL image
                pil_image = tensor2pil(tensor.cpu().detach())[0]
                # Apply Gaussian blur
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur_radius))
                # Convert back to tensor
                out[idx] = pil2tensor(pil_image)
            blurred = torch.cat(out, dim=0)
            return (blurred, 1.0 - blurred)
        else:
            return (torch.stack(out, dim=0), 1.0 - torch.stack(out, dim=0),)
    
    @staticmethod
    def apply_mask(image: Image, mask: Image, mode: str = "RGB") -> Image:
        """Apply the mask on the image."""
        print("Image dimension:", image.size)
        print("Mask dimension:", mask.size)
        if mode == "RGB":
            # Create a black background of the same size
            bg = Image.new("RGB", image.size, (0, 0, 0))
            # Composite the image using the mask
            image = Image.composite(image, bg, mask)
        elif mode == "RBGA":
            # Apply mask as alpha channel
            image.putalpha(mask)
        else:
            raise ValueError("Mode must be either 'RGB' or 'RGBA'")
        return image

