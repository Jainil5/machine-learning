# from diffusers import DiffusionPipeline
# import torch
# import matplotlib.pyplot as plt
# import transformers
# 
# model_id = "CompVis/stable-diffusion-v1-4"
# pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# 
# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt).images[0]
# 
# image.save("image-generation/astronaut_rides_horse.png")
# plt.imshow(image)

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)