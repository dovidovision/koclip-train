import requests
import pandas as pd
from PIL import Image
from io import BytesIO
import parmap
import os

df = pd.read_csv("korea.csv")


headers = {'User-Agent': 'Mozilla/5.0'}
image_paths = []
image_extension = ['.jpg','.png']
image_dir = './data'

if not os.path.exists(image_dir):
    os.makedirs(image_dir,exist_ok=True)

def func(idx):
    try:
        row = df.iloc[idx]
        url = row["image_url"]
        if os.path.splitext(url)[-1] in image_extension:
            response = requests.get(url, headers=headers)
            image_paths.append(url)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            img = img.resize((img.width//4, img.height//4))
            path = os.path.join(image_dir,f"{idx}.jpg")
            img.save(path,'jpeg')
    
    except Exception as e:
        print('>> ERROR :',url)
    finally:
        return 0

result = parmap.map(func, range(len(df)), pm_pbar=True, pm_processes=16)