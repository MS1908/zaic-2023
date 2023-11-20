import os
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel


MODEL_PATH = '/mnt/data/minhnq54/document-vision-models-and-logs/weights/zaic/'

# unet = UNet2DConditionModel.from_pretrained(os.path.join(MODEL_PATH, 'checkpoint-13500/unet'))
pipeline = StableDiffusionPipeline.from_pretrained(MODEL_PATH, safety_checker=None, torch_dtype=torch.float16)
pipeline.to("cuda")

image = pipeline(prompt="Ưu đãi 30% khi đặt hàng ngay trong hôm nay. Miễn phí giao hàng toàn quốc.").images[0]
image = image.resize((1024, 533))
image.save("banner.png")
