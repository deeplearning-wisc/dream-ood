# make sure you're logged in with `huggingface-cli login`
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline


pipe = StableDiffusionPipeline.from_pretrained(
    "/nobackup-slow/dataset/my_xfdu/diffusion/diffusers/models--CompVis--stable-diffusion-v1-4/snapshots/7c3034b58f838791fc1c581d435c452ea80af274/",
    # revision="fp16",
    # torch_dtype=torch.float16,
    use_auth_token=True
).to("cuda")

prompt = "five donuts in the plate"
with autocast("cuda"):
    image = pipe(prompt)["sample"][0]

image.save("astronaut_rides_horse.png")