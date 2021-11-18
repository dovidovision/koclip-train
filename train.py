import torch
import clip
import random
import argparse

import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import math
import time
import numpy as np
import wandb

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
    
    wandb.init(project="ko-clip", entity="maybe your ID")


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
    image_encoder = ImageEncoder().to(device)
    text_encoder = TextEncoder().to(device)

    optimizer = optim.SGD(list(image_encoder.parameters()) + list(text_encoder.parameters()), lr=args.lr,
               momentum=args.momentum, weight_decay=args.weight_decay)
    
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="triangular")

    ce = torch.nn.CrossEntropyLoss()
    image_encoder.train()
    text_encoder.train()

    for epoch in range(args.epochs):
        start = time.time()
        loss_for_monitoring = 0

        for idx, (batch_img, batch_text) in enumerate(train_data_loader):

            image_embedding = image_encoder(batch_img.cuda()) # Output : N x 512
            text_embedding = text_encoder(batch_text.cuda()) # Output : N x 512


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
            wandb.log({"Loss" : loss_for_monitoring})
        
        # How we determine our best model?

        scheduler.step()

        print("Epoch : {:2d} , audio text loss : {:.5f} , Time : {}".format(epoch, loss_for_monitoring / len(train_data_loader), time.time() - start))
        torch.save(image_encoder.state_dict(), "./image_encoder.pth")
        torch.save(text_encoder.state_dict(), "./text_encoder.pth")