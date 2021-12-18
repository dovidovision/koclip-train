import torch
import clip
import random
import argparse
from tqdm import tqdm
from torch.autograd.grad_mode import no_grad

import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision
import torch.nn as nn
import math
import time
import numpy as np
import wandb
import clip
from transformers import AutoModel, AutoTokenizer

from models import ImageEncoder, TextEncoder
from datasets import ImageTextPairDataset

class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 1) #self.linear = {w,b}
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

parser = argparse.ArgumentParser(description="Korean Image Text Clip Implementation")

parser.add_argument("--epochs", default=100, type=int,
                help="epochs of training")
parser.add_argument("--batch_size", default=1, type=int,
                help="batch size of training")
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.8685, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--step_size', default=1, type=float,
                    help='Step size for SGD')
parser.add_argument('--num_workers', default=2, type=int,
                    help='Number of workers used in dataloading')       

parser.add_argument('--image_path', default='./data/', type=str,
                    help='image path that has images.')

args = parser.parse_args()



if __name__ == "__main__":
    random.seed(42)
    
    wandb.init(project="ko-clip")


    train_dataset = ImageTextPairDataset(image_path=args.image_path,type='train') # define in dataset.py

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(len(train_dataset))
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True)

    # define in models.py 
    # image encoder includes Projection Head, So dimension size is 512
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model = nn.Sequential(
        clip_model,
        nn.Linear(512,512),
        nn.Sigmoid()
    )
    print(clip_model)
    exit()
    val_dataset = ImageTextPairDataset(image_path=args.image_path,type='val') # define in dataset.py

    device = "cuda" if torch.cuda.is_available() else "cpu"


    text_encoder = TextEncoder().to(device)
    # Validation : Cifar10 
    testloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True)

    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-small", use_fast=True)

    optimizer = optim.SGD(clip_model[1].parameters(), lr=args.lr,
               momentum=args.momentum, weight_decay=args.weight_decay)
    
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="triangular")

    ce = torch.nn.CrossEntropyLoss()
    clip_model[1].train()

    max_acc = 0

    for epoch in range(args.epochs):
        start = time.time()
        loss_for_monitoring = 0

        pbar = tqdm(enumerate(trainloader), total=len(trainloader))
        for idx, (batch_img, batch_input_ids, batch_attention_mask) in pbar:
            
            with no_grad():
                image_embedding = clip_model[0].encode_image(batch_img.cuda()).float() # Output : N x 512
            
            text_embedding = text_encoder(batch_input_ids.cuda(), batch_attention_mask.cuda()).float() # Output : N x 512

            
            # Normalization is need for calculating cosine similarity
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)    
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

            # optimizer
            optimizer.zero_grad()

            loss = 0

            image_to_text = (image_embedding @ text_embedding.T) * math.exp(0.07) # (N x 512) x (512 x N) = N x N
            text_to_image = (text_embedding @ image_embedding.T) * math.exp(0.07) # (N x 512) x (512 x N) = N x N, 0.07 means temperature

            # Optional : How to add the self-supervised loss?

            # Temperature Normalized Cross Entropy loss
            label = torch.arange(args.batch_size, dtype=torch.long).cuda() # Label is 0~N-1 Why? : Because Batch pair (i, i) is Positive, Other pair (i, j) is Negative


            loss = (ce(image_to_text, label) + ce(text_to_image, label)) * 0.5
            loss.backward()

            optimizer.step()
            
            loss_for_monitoring += loss.item()

            pbar.update()
            pbar.set_description(
                f"Train: [{epoch+1:03d}]",
                f"Loss: {(loss_for_monitoring/(idx+1)):.3f}"
            )
            wandb.log({"Train Iteration Loss" : loss.item()})

            # if idx % 100 == 0:
            #     print("Batch : {}, Image text loss : {:.5f}".format(idx, loss.item()))
                
            
        # How we determine our best model?
        pbar.close()
        scheduler.step()

        print("Epoch : {:2d} , image text loss : {:.5f} , Time : {}".format(epoch, loss_for_monitoring / len(trainloader), time.time() - start))
        wandb.log({"Train Epoch Loss" : loss_for_monitoring / len(trainloader)})

        total = 0
        correct = 0
        
        # korean_labels = [ '엎드려있는 고양이',
        #                 '옆으로 누워있는 고양이',
        #                 '앉아있는 고양이', # 앞발은 꼿꼿하고 뒷발은 웅크린 상태
        #                 '서 있는 고양이',
        #                 '안겨있는 고양이',
        #                 '얼굴만 보이는 고양이',
        #                 ' ',
        #                 '박스 고양이',
        #                 '액체 고양이',
        #                 '식빵을 굽는 고양이',
        #                 '엉덩이를 치켜든 고양이',
        #                 '양말을 신은 고양이',
        #                 '무장해제한 고양이',
        #                 '친구와 함께 있는 고양이',
        #                 '그루밍하는 고양이',
        #                 '간식을 먹는 고양이',
        #                 '냥냥펀치를 하는 고양이',
        #                 '놀고 있는 고양이',]
        korean_labels = [
            '자고 있는 고양이',
            '졸린 고양이', # 누워서 자고 있지 않은 것 (하품, 앉아서 자는 고양이 등)
            '행복한 고양이',
            '편안한 고양이',
            '호기심에 가득 찬 고양이',
            '당황한 고양이',
            '무서워하는 고양이',
            '슬픈 고양이',
            '아무 생각이 없는 고양이',
            '얹짢은 고양이',
            '불안한 고양이',
            '화난 고양이',
        ]

        # for korean_label in korean_labels:
        text_tensor = tokenizer(
            korean_labels,
            return_tensors='pt',
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            add_special_tokens=True,
            return_token_type_ids=False
        )
            
        batch_input_ids = text_tensor['input_ids']
        batch_attention_mask = text_tensor['attention_mask']
        
        text_embedding = text_encoder(batch_input_ids.cuda(), batch_attention_mask.cuda()).float() 
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        clip_model[1].eval()
        for data in testloader:
            images, labels = data
            image_embedding = clip_model[0].encode_image(images.cuda()).float()
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

            image_to_text = (image_embedding @ text_embedding.T) * math.exp(0.07)
            _, predictions = torch.max(image_to_text, 1)
            
            for label, prediction in zip(labels, predictions):
                if label == korean_labels[prediction.item()]:
                    correct += 1
                total += 1
        print("Accuracy : {:.3f}".format(correct / total))
        wandb.log({"Validation Acc" : correct / total})

        acc = correct / total
        if max_acc < acc:
            max_acc = acc
            torch.save(token_embedding.state_dict(), "./token_encoder.pth")