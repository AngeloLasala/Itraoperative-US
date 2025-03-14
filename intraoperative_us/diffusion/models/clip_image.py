"""
Load CLIP image model for computing the text-like embedding of the image
"""
import os
import torch
from PIL import Image
from transformers import CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor

# Load pre-trained CLIP model
model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

# Load CLIP image processor
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

## save model and processor
model_path = "/home/angelo/Documenti/Itraoperative-US/intraoperative_us/diffusion/models/clip_vision_model"
processor_path = "/home/angelo/Documenti/Itraoperative-US/intraoperative_us/diffusion/models/image_processor"

# Ensure directories exist
# os.makedirs(model_path, exist_ok=True)
# os.makedirs(processor_path, exist_ok=True)

# model.save_pretrained(model_path)
# image_processor.save_pretrained(processor_path)

## load model
model = CLIPVisionModel.from_pretrained(model_path)
image_processor = CLIPImageProcessor.from_pretrained(processor_path)


image = torch.randn(3, 256, 256)
image = Image.fromarray(image.numpy().astype('uint8').transpose(1, 2, 0))

# Preprocess the image
inputs = image_processor(images=image, return_tensors="pt")

# Get image embeddings
with torch.no_grad():
    image_embeddings = model(**inputs).last_hidden_state

print(image_embeddings.shape)  # Output: (batch_size, num_patches, embedding_dim)