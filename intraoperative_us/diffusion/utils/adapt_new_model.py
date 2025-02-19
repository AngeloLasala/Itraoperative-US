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
from diffusers import UNet2DConditionModel

## Load all the models in Stable Diffusion
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
unet = pipeline.unet
get_numer_parameter(unet)
# for key in unet.config:
#     print(key, unet.config[key])


## adapt the model to a new unput shape
# Initialize a UNet2DConditionModel with the pretrained text-to-image model weights,
# and change in_channels to 9. Changing the number of in_channels means
# you need to set ignore_mismatched_sizes=True and low_cpu_mem_usage=False
# to avoid a size mismatch error because the shape is different now.  
model_id = "runwayml/stable-diffusion-v1-5"
unet = UNet2DConditionModel.from_pretrained(
    model_id, subfolder="unet",
    in_channels=1,
    sample_size=32,
    low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True
)
print(unet.config.in_channels, unet.config.sample_size)

get_numer_parameter(unet)