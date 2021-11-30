import requests
import pandas as pd
from PIL import Image
from io import BytesIO
import parmap
import os

df = pd.read_csv("korea.csv")


headers = {'User-Agent': 'Mozilla/5.0'}
image_paths = []



def func(idx):
    try:
        if not os.path.exists(f"/media/lsh/Samsung_T5/koclip_dataset/{idx}.png"):
            row = df.iloc[idx]
            url = row["image_url"]
            response = requests.get(url, headers=headers)
            image_paths.append(url)
            img = Image.open(BytesIO(response.content))
            img = img.resize((int(img.width / 2), int(img.height / 2)))
            path = f"/media/lsh/Samsung_T5/koclip_dataset/{idx}.png"
            img.save(path)
            
        
    except Exception as e:
        print(e)
    finally:
        return 0

result = parmap.map(func, range(len(df)), pm_pbar=True, pm_processes=16)