from PIL import Image

class ImageMerger:
    """Handles merging of images and mask fitting."""
    
    @staticmethod
    def merge_images(input_img: Image, outfit_img: Image) -> Image:
        """Merge the input image and outfit image."""
        # TODO: Implement merging logic
        return input_img
    
    @staticmethod
    def fit_mask(mask: Image, merged_img: Image) -> Image:
        """Resize and align mask to fit merged image."""
        return mask.resize(merged_img.size)