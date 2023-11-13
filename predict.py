from cog import BasePredictor, Input, Path
import os
import cv2
import time
import torch
import shutil
import numpy as np
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from typing import List

CONTROLNET_MODEL_NAME = "diffusers/controlnet-canny-sdxl-1.0"

# Use the base SDXL model to fine-tune on specific images.
SDXL_MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"

# Use https://huggingface.co/madebyollin/sdxl-vae-fp16-fix that is modified
# to run in fp16 precision without generating NaNs.
VAE_MODEL_NAME = "madebyollin/sdxl-vae-fp16-fix"

CONTROL_CACHE = "control-cache"
VAE_CACHE = "vae-cache"
MODEL_CACHE = "sdxl-cache"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Initialize ControlNet pipeline.")
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_MODEL_NAME,
            torch_dtype=torch.float16
        )
        print("Loading VAE Autoencoder")
        vae = AutoencoderKL.from_pretrained(
            VAE_MODEL_NAME,
            torch_dtype=torch.float16
        )
        print("Loading SDXL")
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            SDXL_MODEL_NAME,
            vae=vae,
            controlnet=controlnet,
            use_safetensors=True,
            variant="fp16",
            torch_dtype=torch.float16,
        )
        self.pipe = pipe.to("cuda")
        print("Predictor setup complete.")

    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")
    
    def process_image(self, image):
        image = self.load_image(image)
        image_width, image_height = image.size

        "Convert the loaded image into NumPy array."
        image = np.array(image)
        
        "Edge detection operation using the Canny edge detection algorithm from the OpenCV"
        image = cv2.Canny(image, 100, 200)

        "Add extra dimensions to the image"
        image = image[:, :, None]

        """Converts the image to a 3 channel image used in formatting for color 
        images and convert the NumPy array back to image."""
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        return image

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(
            description="Input image to create in different art medium",
            default=None,
        ),
        prompt: str = Input(
            description="Choose art medium to generate in that style. For example: a photo of, a acrylic paint of, an anime drawing of, a caricature of, a cartoon of, a drawing of, a graphitti of, an illustration of, a line art of, an oil painting of, a pencil sketch of",
            default="",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face, blurry, draft, grainy",
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=768,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=768,
        ),
        num_outputs: int = Input(
            description="Number of images to output. > 2 might generate out-of-memory errors.",
            ge=1,
            le=4,
            default=1,
        ),
        seed: int = Input(
            description="Random seed. Set to 0 to randomize the seed. If you need tweaks to a generated image, reuse the same seed number from output logs.", 
            default=0,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference",
            ge=1,
            le=500,
            default=100,
        ),
        guidance_scale: float = Input(
            description="A higher guidance scale value generate images closely to the text prompt at the expense of lower image quality. Guidance scale is enabled when guidance_scale > 1.",
            default=7.5,
        ),
    ) -> List[Path]:
        if (seed is None) or (seed <= 0):
            seed = int.from_bytes(os.urandom(2), "big")
        generator = torch.Generator("cuda").manual_seed(seed)
        print(f"Using seed: {seed}")

        "Process the input image to detect edge using NumPy and OpenCV libraries"
        image = self.process_image(image)

        """The outputs of the ControlNet are multiplied by controlnet_conditioning_scale
        before they are added to the residual in the original unet."""
        condition_scale = 0.5

        output = self.pipe(
            prompt=[prompt] * num_outputs if prompt is not None else None,
            negative_prompt=[negative_prompt] * num_outputs
            if negative_prompt is not None
            else None,
            image=image, 
            controlnet_conditioning_scale=condition_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )

        output_paths = []
        for i, sample in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        print("Prediction complete")
        return output_paths


