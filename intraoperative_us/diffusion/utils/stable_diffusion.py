"""
Pipeline for Stabel Diffusion in hugging face
"""
import torch
from diffusers import AutoencoderKL, PNDMScheduler, DDPMPipeline, DDPMScheduler, UniPCMultistepScheduler
from diffusers import UNet2DModel, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
import matplotlib.pyplot as plt
from PIL import Image   
import numpy as np
from torchvision import transforms
from tqdm.auto import tqdm

## load each element od SDM - VAE, TOKENIZER, TEXT_ENCODER, UNET, SCHEDULER
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True)
tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder", use_safetensors=True)
unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="unet", use_safetensors=True
)
scheduler = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

torch_device = "cuda"
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)

# create TEXT EMBEDDING - text conditional
prompt = ["a photograph of an cat"]
height = 516                       # default height of Stable Diffusion
width = 516                         # default width of Stable Diffusion
num_inference_steps = 25            # Number of denoising steps
guidance_scale = 5                # Scale for classifier-free guidance
# generator = torch.Generator(device=torch_device).manual_seed(0)    # Seed generator to create the initial latent noise
batch_size = len(prompt)

## conditional text embedding
text_input = tokenizer(prompt, 
                       padding="max_length",
                       max_length=tokenizer.model_max_length,
                       truncation=True,
                       return_tensors="pt")

with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

## unconditional text embedding
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
with torch.no_grad():
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

text_embeddings =  torch.cat([uncond_embeddings, text_embeddings])


## CREATE RANDOM NOISE
# this is the latent rappresentation of the image
# divided by 8 because  the vae has got 3 downsampling leyyers
latents = torch.randn(
    (batch_size, unet.config.in_channels, height // 8, width // 8),
    # generator=generator,
    device=torch_device,
).to(torch_device)
latents = latents * scheduler.init_noise_sigma

## DENOISING LOOP
scheduler.set_timesteps(num_inference_steps)
for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample

    # if t % 100 == 0:
    #     image = latents.cpu().numpy().squeeze()[0]
    #     image = (image + 1.0) * 127.5
    #     image = image.astype(np.uint8)
    #     image_pil = Image.fromarray(image)
    #     plt.figure(figsize=(10, 10), num=f"Latent at step {t}")
    #     plt.imshow(image_pil)
    #     plt.show()



## RECONSTRUCT THE IMAGE 
latents = latents * (1 / 0.18215)    # get the correct normalization of stable diffusion

with torch.no_grad():
    image = vae.decode(latents).sample

image = (image / 2 + 0.5).clamp(0, 1)  # Normalize to [0,1]
image = (image.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()
image = Image.fromarray(image[0])  # Remove batch dimension
plt.figure(figsize=(10, 10), num="Final Image")
plt.imshow(image) 

# get the encdore of the image
with torch.no_grad():
    # Convert PIL Image to Tensor
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(torch_device)  # Add batch dimension

    latents = vae.encode(image_tensor).latent_dist.sample()
print(latents.shape)

# plot every channel of the latent
for i in range(latents.shape[1]):
    plt.figure(figsize=(10, 10), num=f"Latent channel {i}")
    plt.imshow(latents.cpu().numpy().squeeze()[i])

## open an image
image = Image.open("/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/RESECT_iUS_dataset/dataset/Case24-US-before/volume/Case24-US-before_212.png")
image = image.resize((256, 256))
image = image.convert("RGB")
image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(torch_device)  # Add batch dimension

# get the encdore of the image
with torch.no_grad():
    latents = vae.encode(image_tensor).latent_dist.sample()
print(latents.shape)

# plot every channel of the latent
plt.figure(figsize=(10, 10), num="Original Image")
plt.imshow(image)

for i in range(latents.shape[1]):
    plt.figure(figsize=(10, 10), num=f"Latent channel iUS {i}")
    plt.imshow(latents.cpu().numpy().squeeze()[i])

plt.show()
