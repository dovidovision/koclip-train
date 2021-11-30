import torch
import clip
import random
import argparse
from torch.autograd.grad_mode import no_grad

import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import math
import time
import numpy as np
import wandb
import clip
from transformers import AutoModel, AutoTokenizer

from models import ImageEncoder, TextEncoder
from datasets import ImageTextPairDataset
parser = argparse.ArgumentParser(description="Korean Image Text Clip Implementation")

parser.add_argument("--epochs", default=100, type=int,
                help="epochs of training")
parser.add_argument("--batch_size", default=192, type=int,
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
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')       


args = parser.parse_args()



if __name__ == "__main__":
    random.seed(42)
    
    wandb.init(project="ko-clip", entity="easter3163")


    imagetext_dataset = ImageTextPairDataset() # define in dataset.py

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data_loader = torch.utils.data.DataLoader(
        imagetext_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True)

    # define in models.py 
    # image encoder includes Projection Head, So dimension size is 512
    clip_model, _ = clip.load("ViT-B/32", device=device)
    text_encoder = TextEncoder().to(device)
    tokenizer = AutoModel.from_pretrained("klue/roberta-base")


    optimizer = optim.SGD(text_encoder.parameters(), lr=args.lr,
               momentum=args.momentum, weight_decay=args.weight_decay)
    
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="triangular")

    ce = torch.nn.CrossEntropyLoss()
    text_encoder.train()

    for epoch in range(args.epochs):
        start = time.time()
        loss_for_monitoring = 0

        for idx, (batch_img, batch_input_ids, batch_attention_mask) in enumerate(train_data_loader):
            
            with no_grad():
                image_embedding = clip_model.encode_image(batch_img.cuda()).float() # Output : N x 512
            
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
            wandb.log({"Loss" : loss.item()})
            print("Image text loss : {:.5f}".format(loss.item()))
        # How we determine our best model?

        scheduler.step()

        print("Epoch : {:2d} , image text loss : {:.5f} , Time : {}".format(epoch, loss_for_monitoring / len(train_data_loader), time.time() - start))
        torch.save(text_encoder.state_dict(), "./text_encoder.pth")