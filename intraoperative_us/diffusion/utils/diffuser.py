"""
Try use the diffuser of hugginface to generate the diffusion images

Simple tutorial codes
"""
import torch
from diffusers import DDPMPipeline, DDPMScheduler
from diffusers import UNet2DModel 
import matplotlib.pyplot as plt
from PIL import Image   
import numpy as np
import tqdm

## Example of generate image form pretraind DDPM model on celebahq
image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
image_pipe.to("cuda")
print(image_pipe)
# image = image_pipe().images
# plt.imshow(image[0])
# plt.show()


## Have a look at the model
model = UNet2DModel.from_pretrained("google/ddpm-church-256").to("cuda")

## step-by-step inference
torch.manual_seed(0)
noisy_sample = torch.randn(1, 
                model.config.in_channels, 
                model.config.sample_size,
                model.config.sample_size).cuda()
print(noisy_sample.shape)

## MANUAL INFERENCE
# 1. GENERATE THE NOISE - OUTPUT OF THE MODEL
# Note: important red the model card to see what is the model output
with torch.no_grad():
    # noisy_residual is the difference  between 
    # the slightly less noisy image and the input image
    noisy_residual = model(sample = noisy_sample, timestep=1000).sample

# convert to numpy ad plot
# noisy_residual = noisy_residual.cpu().numpy()
# noisy_residual = noisy_residual.transpose(0, 2, 3, 1)
# plt.imshow(noisy_residual[0])

# 2. SCHEDULER - denoise a noise
# noise schedule which is used to add noise to the model during training,
# and also define the algorithm to compute the slightly less noisy sample given the model output 
scheduler = DDPMScheduler.from_pretrained("google/ddpm-church-256")
print(scheduler.config)
less_noisy = scheduler.step(model_output = noisy_residual,
                            timestep=2,
                            sample=noisy_sample).prev_sample
# convert to numpy ad plot
# less_noisy = less_noisy.cpu().numpy()
# less_noisy = less_noisy.transpose(0, 2, 3, 1)
# plt.imshow(less_noisy[0])
# plt.show()

## 3. DENOISING LOOP
def display_sample(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = Image.fromarray(image_processed[0])
    plt.figure(figsize=(10, 10), num=f'Image at step {i}')
    plt.imshow(image_pil)
    plt.show()

sample = noisy_sample.clone()
for i,t in enumerate(tqdm.tqdm(scheduler.timesteps)):
    
    # predict noise residual
    with torch.no_grad():
        residual = model(sample = sample, timestep=t).sample
    
    # compute the less noisy image x_t->x_t-1
    sample = scheduler.step(model_output = residual,
                            timestep=t,
                            sample=sample).prev_sample
    if (i+1) % 100 == 0:
        display_sample(sample, i+1)


## Appendix -  Random initialization of the model
#  model_random = UNet2DModel(**model.config)
#  model_random.save_pretrained("model_random")
#  model_random = UNet2DModel.from_pretrained("model_random")

