# main.py
import io
import uuid

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import Optional
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from utils.image_processor import ImageProcessor
from utils.mask_processor import MaskProcessor
from utils.image_merger import ImageMerger
from utils.redux_processor import ReduxProcessor
from utils.ksampler_processor import KSamplerProcessor

# Initialize Mask Processor
mask_processor = MaskProcessor()

# Initialize FastAPI
app = FastAPI()

@app.post("/edit")
async def edit_image(
    input_image: UploadFile = File(...),
    style_image: UploadFile = File(...),
    edit_location: str = Form(...),
    input_mask: Optional[UploadFile] = File(None)
):
    try:
        # Step 1: Load images
        input_img = ImageProcessor.load_image(input_image)
        style_img = ImageProcessor.load_image(style_image)
        mask_img = ImageProcessor.load_image(input_mask, "L") if input_mask else None
        
        if not style_image:
            # Step 2: Generate segmentation mask (Grounding DINO + SAM2)
            generated_mask = mask_processor.generate_mask(input_img, edit_location) if not mask_img else mask_img
            mask_img, _ = mask_processor.expand_mask(generated_mask, 30, false, false, 5, 1.3, 1, 1, falseZ

        # sked_image = mask_processor.apply_mask(input_img, generated_mask)

        masked_image.save("output.png")

        # return generated_mask
        # return StreamingResponse(io.BytesIO(masked_image.tobytes()), media_type="image/png")
        return JSONResponse(content={"message": "Mask generated successfully."})
        
        # Step 3: Grow the mask or modify it
        modified_mask = mask_processor.grow_mask(generated_mask)
        
        # Step 4: Merge input image with outfit image
        merged_img = ImageMerger.merge_images(input_img, outfit_img)
        
        # Step 5: Fit mask to merged image
        fitted_mask = ImageMerger.fit_mask(modified_mask, merged_img)
        
        # Step 6: Process outfit image with Redux
        styled_outfit = ReduxProcessor.apply_redux(outfit_img)
        
        # Step 7: Apply style and pass to KSampler (Flux Fill Dev)
        final_output = KSamplerProcessor.apply_ksampler(merged_img, fitted_mask, styled_outfit)
        
        # Save and return response
        ImageProcessor.save_image(final_output, f"outputs/INPUT_{ImageProcessor.get_image_name_without_extension(input_image)}_OUTFIT_{ImageProcessor.get_image_name_without_extension(outfit_image)}_EDIT_{edit_location}_{uuid.uuid1()}.png")
        return FileResponse("output.png", media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
