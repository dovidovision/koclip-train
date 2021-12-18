from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms



from glob import glob
import os
import PIL
import pandas as pd
import clip
from transformers import AutoModel, AutoTokenizer
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageTextPairDataset(Dataset):
    def __init__(self,image_path,type):
        # self.image_list = glob(os.path.join(image_path,"**/*.jpg"))
        self.type=type
        if type=='train':
            self.image_text_dataframe = pd.read_csv("/opt/ml/koclip-train/train.csv")
        if type=='val':
            self.image_text_dataframe = pd.read_csv("/opt/ml/koclip-train/val.csv")
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        _, self.preprocess = clip.load("ViT-B/32")
        self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-small", use_fast=True)
        
    def __getitem__(self, idx):
        if self.type=='val':
            image_path = self.image_text_dataframe["filename"][idx]
            image = self.preprocess(PIL.Image.open('/opt/ml/koclip-train/data/'+image_path))
            return image, self.image_text_dataframe["emotion"][idx]
        
        else:
            image_path = self.image_text_dataframe["filename"][idx]

            text_prompt = self.image_text_dataframe["emotion"][idx]
            image = self.preprocess(PIL.Image.open('/opt/ml/koclip-train/data/'+image_path))
            image_tensor = image

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

            
        

    def __len__(self):
        
        return len(self.image_text_dataframe["filename"])



# dataset = ImageTextPairDataset()
