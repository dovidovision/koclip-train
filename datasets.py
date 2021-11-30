from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms



from glob import glob

import PIL
import pandas as pd
import clip
from transformers import AutoModel, AutoTokenizer
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageTextPairDataset(Dataset):
    def __init__(self):
        
        self.image_list = glob("/media/lsh/Samsung_T5/koclip_dataset/*.png")
        self.image_text_dataframe = pd.read_csv("./korea.csv")
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        _, self.preprocess = clip.load("ViT-B/32")
        self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-small", use_fast=True)
        
    def __getitem__(self, idx):

        
        image_path = self.image_list[idx]

        text_idx = int(image_path.split("/")[-1].split(".")[0])
        text_prompt = self.image_text_dataframe.iloc[text_idx]["text"]
        
        image = self.preprocess(PIL.Image.open(image_path))
        image_tensor = image

        try:
            text_tensor = self.tokenizer(
                text_prompt,
                return_tensors='pt',
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                add_special_tokens=True,
                return_token_type_ids=False
            )


            input_ids = text_tensor['input_ids'][0]
            attention_mask = text_tensor['attention_mask'][0]
            return image_tensor, input_ids, attention_mask # tensor : 3 x 224 x 224, "happy cat"
        
        except:
            return self.__getitem__(idx + 1)
            
        

    def __len__(self):
        
        return len(self.image_list)



dataset = ImageTextPairDataset()
