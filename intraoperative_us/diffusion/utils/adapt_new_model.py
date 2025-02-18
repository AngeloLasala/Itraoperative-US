"""
Many diffusion systems share the same components, 
allowing you to adapt a pretrained model for one task to an entirely different task.

This guide will show you how to adapt a pretrained 
text-to-image model for inpainting by initializing 
and modifying the architecture of 
a pretrained UNet2DConditionModel.
"""
from diffusers import StableDiffusionPipeline
from intraoperative_us.diffusion.utils.utils import get_numer_parameter

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
unet = pipeline.unet

get_numer_parameter(unet)
