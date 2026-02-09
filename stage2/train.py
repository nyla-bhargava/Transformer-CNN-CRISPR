import os
import time
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from utils.seed import set_seed
from stage2.dataset import OffTargetDataset
from stage2.model import Stage2Model

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--use_stage1", action="store_true")
args = parser.parse_args()

# Config and Hyperparams
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_STAGE1 = args.use_stage1
EPOCHS = 40
WARMUP_EPOCHS = 5
SWA_START = 25  # Start averaging weights at epoch 25
LR = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128

os.makedirs("checkpoints", exist_ok=True)

# Loss Function (Focal Loss)
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()

# Data Preparation
proxy_df = pd.read_csv("data/Proxy_TrainCV.csv")
train_df, val_df = train_test_split(proxy_df, test_size=0.15, stratify=proxy_df["label"], random_state=42)

MAX_LEN = max(proxy_df.gRNA.str.len().max(), proxy_df.OT.str.len().max())
all_gRNAs = np.load("stage1/all_gRNAs.npy", allow_pickle=True)
gRNA_to_idx = {g: i for i, g in enumerate(all_gRNAs)}

if USE_STAGE1:
    sg_embeddings = torch.load("stage1/sg_embeddings.pt", map_location=device)
    sg_dim = sg_embeddings.shape[1]
else:
    sg_embeddings = torch.zeros(len(all_gRNAs), 768)
    sg_dim = 768

train_loader = DataLoader(OffTargetDataset(train_df, sg_embeddings, gRNA_to_idx, MAX_LEN, train=True), 
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(OffTargetDataset(val_df, sg_embeddings, gRNA_to_idx, MAX_LEN, train=False), 
                        batch_size=BATCH_SIZE, shuffle=False)

# Model and Optimizer Setup
model = Stage2Model(sg_dim).to(device)
swa_model = AveragedModel(model)

criterion = FocalLoss(gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Learning Rate Schedule (Warmup + Cosine)
def lr_lambda(epoch):
    if epoch <= WARMUP_EPOCHS:
        return epoch / WARMUP_EPOCHS
    return 0.5 * (1 + math.cos(math.pi * (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)))

scheduler = LambdaLR(optimizer, lr_lambda)
swa_scheduler = SWALR(optimizer, swa_lr=0.05 * LR)

# Training Utils
def run_epoch(m, loader, opt=None, train=False):
    if train: m.train()
    else: m.eval()
    
    total_loss, ys, ps = 0, [], []
    for b in loader:
        pair, mv, pam = b["pair"].to(device), b["mv"].to(device), b["pam"].to(device)
        sg_emb, target = b["sg_emb"].to(device), b["label"].to(device)
        
        if train: opt.zero_grad()
        logits = m(pair, mv, pam, sg_emb).squeeze()
        loss = criterion(logits, target)
        
        if train:
            loss.backward()
            opt.step()
            
        total_loss += loss.item() * len(target)
        ps.append(torch.sigmoid(logits).detach().cpu().numpy())
        ys.append((target > 0.5).cpu().numpy().astype(int))
        
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return total_loss/len(y), roc_auc_score(y, p), average_precision_score(y, p)

# Main loop
log(f"Starting Seed {args.seed} | Mode: {'STAGE-1' if USE_STAGE1 else 'BASELINE'}")

for epoch in range(1, EPOCHS + 1):
    t_loss, t_auc, t_aupr = run_epoch(model, train_loader, optimizer, train=True)
    
    if epoch >= SWA_START:
        swa_model.update_parameters(model)
        swa_scheduler.step()
    else:
        scheduler.step()
        
    v_loss, v_auc, v_aupr = run_epoch(model, val_loader)
    log(f"Epoch {epoch:02d} | Train AUPR: {t_aupr:.4f} | Val AUPR: {v_aupr:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

# Finalize SWA
log("Finalizing SWA Batch Normalization...")
update_bn(train_loader, swa_model, device=device)

# Saving Final Model
suffix = "full" if USE_STAGE1 else "baseline"
ckpt_path = f"checkpoints/best_model_{suffix}_seed{args.seed}.pt"
torch.save(swa_model.module.state_dict(), ckpt_path)
log(f"Model saved to {ckpt_path}")
