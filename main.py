# main.py
import io

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import Optional
from fastapi.responses import StreamingResponse

from utils.image_processor import ImageProcessor
from utils.redux_processor import ReduxProcessor
from utils.ksampler_processor import KSamplerProcessor
from utils.misc.utility import tensor2pil, pil2tensor
from utils.vae import load_vae

import traceback

# Initialize Mask Processor
image_processor = ImageProcessor(max_resolution=1024)
vae = load_vae()
redux_processor = ReduxProcessor(vae=vae)
k_sampler_processor = KSamplerProcessor(vae=vae, weight_dtype="fp8_e4m3fn")

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
        merged_img_tensor, merged_mask_tensor, resized_style_tensor = image_processor.process(input_image, style_image, mask_image, edit_location)

        # Step 2: Process images with Redux
        positive_conditioning, negative_conditioning, latent = redux_processor.apply_redux(resized_style_tensor, merged_img_tensor, merged_mask_tensor)
        
        # Step 3: Apply KSampler
        generated_images = k_sampler_processor.apply_ksampler(
            seed=0,
            steps=20,
            cfg=1.3,
            sampler_name="euler",
            scheduler="simple",
            positive=positive_conditioning,
            negative=negative_conditioning,
            latent=latent,
            denoise=1.0
        )

        output_pil = tensor2pil(generated_images)[0]

        # Create a BytesIO buffer and save the PIL image into it in PNG format:
        buf = io.BytesIO()
        output_pil.save(buf, format="PNG")
        buf.seek(0)

        # return generated output
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
