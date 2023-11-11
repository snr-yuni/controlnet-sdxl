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
# to run modified to run in fp16 precision without generating NaNs.
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

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(
            description="Input image to create in different art medium",
            default=None,
        ),
        prompt: str = Input(
            description="Input your imagination",
            default="line art",
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
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps",
            ge=1,
            le=500,
             default=50,
        ),
        condition_scale: float = Input(
            description="controlnet conditioning scale for generalization",
            default=0.5,
            ge=0.0,
            le=1.0,
        ),
        art_medium: str = Input(
            default="line art",
            choices=[
                "acrylic paint"
                "caricature",
                "cartoon"
                "cinematic",
                "drawing",
                "graphite"
                "illustration",
                "painting",
            ],
            description="Choose an art medium.",
        ),
        seed: int = Input(
            description="Random seed. Set to 0 to randomize the seed", default=0
        ),
    ) -> List[Path]:
        if (seed is None) or (seed <= 0):
            seed = int.from_bytes(os.urandom(2), "big")
        generator = torch.Generator("cuda").manual_seed(seed)
        print(f"Using seed: {seed}")

        image = self.load_image(image)
        image_width, image_height = image.size

        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)

        images = self.pipe(
            prompt=[art_medium + prompt] * num_outputs if prompt is not None else None,
            negative_prompt=[negative_prompt] * num_outputs
            if negative_prompt is not None
            else None,
            image=image, 
            controlnet_conditioning_scale=condition_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        )

        output_paths = []
        for i, sample in enumerate(output.images):
            if output.nsfw_content_detected and output.nsfw_content_detected[i]:
                continue

            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        print("Prediction complete")
        return output_paths


