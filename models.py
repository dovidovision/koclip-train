import torch 
import torchvision.models as models

import timm
from transformers import BertTokenizerFast, BertModel

class ImageEncoder(torch.nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.backbone = timm.create_model("resnet18", num_classes=512, pretrained=True)

    def forward(self, x):
        x = self.backbone(x)
        return x

class TextEncoder(torch.nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.tokenizer_bert = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
        self.model_bert = BertModel.from_pretrained("kykim/bert-kor-base")

    def forward(self, x):
        x = self.tokenizer_bert(x)
        x = self.model_bert(x)
        return x