import json
import pandas as pd
import numpy as np


ANNOT_PATH = '/mnt/data/minhnq54/zaic-2023-banner/train/info.csv'
annot_df = pd.read_csv(ANNOT_PATH, encoding='utf-8')

items = []
for i, row in annot_df.iterrows():
    v = [s for s in [row['caption'], row['description'], row['moreInfo']] if isinstance(s, str)]
    text = ' '.join(v)
    im_name = row['bannerImage']
    items.append({
        'image': im_name,
        'text': text
    })
    
with open('/mnt/data/minhnq54/zaic-2023-banner/train/metadata.jsonl', 'w', encoding='utf-8') as w:
    for item in items:
        json.dump(item, w, ensure_ascii=False)
        w.write('\n')
