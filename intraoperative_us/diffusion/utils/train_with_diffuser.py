"""
Train simple DDPM with diffuser packages
"""
from dataclasses import dataclass
from datasets import load_dataset
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
from diffusers import UNet2DModel
from diffusers import DDPMScheduler, DDPMPipeline, PNDMScheduler
import math
from PIL import Image
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup

from accelerate import Accelerator, notebook_launcher
from tqdm.auto import tqdm
from pathlib import Path
import os

import yaml
import torch
import os
import json

import matplotlib.pyplot as plt


@dataclass
class TrainingConfig:
    image_size: int = 128                      # the generated image resolution
    train_batch_size: int = 4
    eval_batch_size: int = 4                  # how many images to sample during evaluation
    num_epochs: int = 50
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 10
    save_model_epochs: int = 30
    mixed_precision: str = 'fp16'             # `no` for float32, `fp16` for automatic mixed precision
    output_dir: str = 'butterflies'  # the model name locally and on the HF Hub

    push_to_hub: bool = False                  # whether to upload the saved model to the HF Hub
    hub_private_repo: bool = False  
    overwrite_output_dir: bool = True         # overwrite the old model when re-running the notebook
    seed: int = 0

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size = config.eval_batch_size, 
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

config = TrainingConfig()

## load dataset 
config.dataset_name = "huggan/smithsonian_butterflies_subset"
dataset = load_dataset(config.dataset_name, split="train")
print(dataset)

fig, axs = plt.subplots(1, 4, figsize=(16, 4), num='Example of images')
for i, image in enumerate(dataset[:4]["image"]):
    axs[i].imshow(image)
    axs[i].set_axis_off()
# plt.show()


## prepocess the dataset
preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

dataset.set_transform(transform)

fig, axs = plt.subplots(1, 4, figsize=(16, 4), num='Example of preprocess images')
for i, image in enumerate(dataset[:4]["images"]):
    axs[i].imshow(image.permute(1, 2, 0).numpy() / 2 + 0.5)
    axs[i].set_axis_off()
plt.show()


# data loader
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)


## model
model = UNet2DModel(
    sample_size=config.image_size,    # the target image resolution
    in_channels=3,                    # the number of input channels, 3 for RGB images
    out_channels=3,                   # the number of output channels
    layers_per_block=2,               # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
    down_block_types=( 
        "DownBlock2D",      # a regular ResNet downsampling block
        "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ), 
    up_block_types=(
        "UpBlock2D",        # a regular ResNet upsampling block
        "AttnUpBlock2D",    # a ResNet upsampling block with spatial self-attention
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D"  
      ),
)

##  SCHEDULER with example
noise_scheduler = PNDMScheduler.from_pretrained("sd-legacy/stable-diffusion-v1-5", subfolder="scheduler")

sample_image = dataset[60]["images"].unsqueeze(0)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

fig, axs = plt.subplots(1, 4, figsize=(16, 4), num='Adding noise with scheduler')
for i, timesteps in enumerate([250, 500, 750, 999]):
    timesteps = torch.LongTensor([timesteps])
    noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)
    ## set title
    axs[i].set_title(f"t={timesteps.item()}", fontsize=30)
    axs[i].imshow(noisy_image[0].permute(1, 2, 0).numpy() / 2 + 0.5)
    axs[i].axis('off')
plt.show() 


## IINITIALIZATION OF THE TRAINING
                                     ## MSE los for the predicted noise
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)    ## AdamW optimizer    
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
    )
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")
    
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the 
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process and epoch % config.save_model_epochs == 0:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            evaluate(config, epoch, pipeline)

args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
notebook_launcher(train_loop, args, num_processes=1)

sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
plt.show()
