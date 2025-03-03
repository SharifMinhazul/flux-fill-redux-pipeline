# main.py
import io
import uuid

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import Optional
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import torch

from utils.image_processor import ImageProcessor
from utils.mask_processor import MaskProcessor
from utils.image_merger import ImageMerger
from utils.redux_processor import ReduxProcessor
from utils.ksampler_processor import KSamplerProcessor

from utils.misc.utility import tensor2pil, pil2tensor
import traceback

# Initialize Mask Processor
image_processor = ImageProcessor(max_resolution=1024)
redux_processor = ReduxProcessor(clip_path="models/clip_vision/sigclip_vision_patch14_384.safetensors")

# Initialize FastAPI
app = FastAPI(swagger_ui_parameters={"tryItOutEnabled": True})

@app.post("/edit")
async def edit_image(
    input_image: UploadFile = File(...),
    style_image: UploadFile = File(...),
    edit_location: str = Form(default="hair"),
    mask_image: Optional[UploadFile] = File(default=None)
):
    try:
        # Step 1: Load images
        merged_img_tensor, merged_mask_tensor = image_processor.process(input_image, style_image, mask_image, edit_location)

        mask_pil = tensor2pil(merged_mask_tensor)[0]

        # Create a BytesIO buffer and save the PIL image into it in PNG format:
        buf = io.BytesIO()
        mask_pil.save(buf, format="PNG")
        buf.seek(0)

        # return generated_mask
        return StreamingResponse(buf, media_type="image/png")
        # return JSONResponse(content={"message": "Mask generated successfully."})

        # Step 6: Process outfit image with Redux
        styled_outfit = ReduxProcessor.apply_redux(outfit_img)
        
        # Step 7: Apply style and pass to KSampler (Flux Fill Dev)
        final_output = KSamplerProcessor.apply_ksampler(merged_img, fitted_mask, styled_outfit)
        
        # Save and return response
        ImageProcessor.save_image(final_output, f"outputs/INPUT_{ImageProcessor.get_image_name_without_extension(input_image)}_OUTFIT_{ImageProcessor.get_image_name_without_extension(outfit_image)}_EDIT_{edit_location}_{uuid.uuid1()}.png")
        return FileResponse("output.png", media_type="image/png")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
