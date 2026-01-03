import os
import sys
import time
import numpy as np
import torch
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from utils.seed import set_seed
from stage1.embeddings import compute_sg_embeddings
from stage2.dataset import OffTargetDataset
from stage2.model import Stage2Model


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_STAGE1 = True
EPOCHS = 30
BATCH_SIZE = 128
LR = 2e-4
WEIGHT_DECAY = 1e-4


log("Starting Stage-2 training")
log(f"Device: {device}")
log(f"USE_STAGE1 = {USE_STAGE1}")


log("Loading datasets...")
proxy_df = pd.read_csv("data/Proxy_TrainCV.csv")
trueot_df = pd.read_csv("data/TrueOT_1806uniqueTriplet_gRNA_OT_label.csv")

log(f"Proxy dataset shape: {proxy_df.shape}")
log(f"TrueOT dataset shape: {trueot_df.shape}")


train_df, val_df = train_test_split(
    proxy_df,
    test_size=0.15,
    stratify=proxy_df["label"],
    random_state=42
)

log(f"Train size: {len(train_df)} | Val size: {len(val_df)}")


MAX_LEN = max(
    proxy_df.gRNA.str.len().max(),
    proxy_df.OT.str.len().max()
)

log(f"Max sequence length: {MAX_LEN}")

all_gRNAs = pd.concat([
    train_df.gRNA,
    val_df.gRNA,
    trueot_df.gRNA
]).unique()

log(f"Unique gRNAs: {len(all_gRNAs)}")

gRNA_to_idx = {g: i for i, g in enumerate(all_gRNAs)}

# Save ordering for evaluation (IMPORTANT)
os.makedirs("stage1", exist_ok=True)
np.save("stage1/all_gRNAs.npy", all_gRNAs)


if USE_STAGE1:
    log("Computing Stage-1 sgRNA embeddings...")
    sg_embeddings = compute_sg_embeddings(list(all_gRNAs), device)
    sg_dim = sg_embeddings.shape[1]
    log(f"Embeddings shape: {sg_embeddings.shape}")

    # Optional cache (safe)
    torch.save(sg_embeddings, "stage1/sg_embeddings.pt")
else:
    log("Stage-1 disabled: using zero embeddings")
    sg_embeddings = torch.zeros(len(all_gRNAs), 768)
    sg_dim = 768


train_loader = DataLoader(
    OffTargetDataset(train_df, sg_embeddings, gRNA_to_idx, MAX_LEN),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    OffTargetDataset(val_df, sg_embeddings, gRNA_to_idx, MAX_LEN),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

log("DataLoaders created")


model = Stage2Model(sg_dim).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

log("Model initialized")


def run_epoch(loader, train: bool):
    model.train() if train else model.eval()
    ys, ps = [], []
    total_loss = 0.0

    for i, b in enumerate(loader):
        for k in b:
            b[k] = b[k].to(device)

        if train:
            optimizer.zero_grad()

        logits = model(
            b["pair"],
            b["mv"],
            b["pam"],
            b["sg_emb"]
        )

        loss = criterion(logits, b["label"])

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(b["label"])
        ps.append(torch.sigmoid(logits).detach().cpu().numpy())
        ys.append(b["label"].cpu().numpy())

    y = np.concatenate(ys)
    p = np.concatenate(ps)

    return (
        total_loss / len(y),
        roc_auc_score(y, p),
        average_precision_score(y, p)
    )


best_val_auc = 0.0

log("Starting training loop...")

for epoch in range(1, EPOCHS + 1):
    train_loss, train_auc, train_aupr = run_epoch(train_loader, train=True)
    val_loss, val_auc, val_aupr = run_epoch(val_loader, train=False)

    log(
        f"Epoch {epoch:02d} | "
        f"Train AUC: {train_auc:.4f} | "
        f"Val AUC: {val_auc:.4f}"
    )

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), "best_stage2.pt")
        log(f"Saved new best model (Val AUC = {val_auc:.4f})")


log("Training complete")
log(f"Best validation AUC: {best_val_auc:.4f}")
log("Checkpoint saved as best_stage2.pt")
