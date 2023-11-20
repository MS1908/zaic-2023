import os
import torch
import pandas as pd
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from tqdm import tqdm


MODEL_PATH = '/mnt/data/minhnq54/document-vision-models-and-logs/weights/zaic/'
test_df = pd.read_csv('/mnt/data/minhnq54/zaic-2023-banner/test/info.csv', encoding='utf-8')

pipeline = StableDiffusionPipeline.from_pretrained(MODEL_PATH, safety_checker=None, torch_dtype=torch.float16)
pipeline.set_progress_bar_config(disable=True)
pipeline.to("cuda")

os.makedirs('test', exist_ok=True)
for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
    image_name = f"{row['id']}.jpg"
    prompt = ' '.join([s for s in [row['caption'], row['description'], row['moreInfo']] if isinstance(s, str)])
    
    image = pipeline(prompt=prompt).images[0]
    image = image.resize((1024, 533))
    
    image.save(f'test/{image_name}')
