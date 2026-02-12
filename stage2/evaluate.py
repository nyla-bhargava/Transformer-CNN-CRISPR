import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

from stage2.dataset import OffTargetDataset
from stage2.model import Stage2Model
from utils.metrics import mc_dropout
from utils.seed import set_seed

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True, help="Seed corresponding to trained checkpoint")
parser.add_argument("--use_stage1", action="store_true", help="Enable Stage-1 (must match training)")
args = parser.parse_args()

# Setup
set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_STAGE1 = args.use_stage1
BATCH_SIZE = 128

# Create results directory
os.makedirs("results", exist_ok=True)

# Loading data
proxy_df = pd.read_csv("data/Proxy_TrainCV.csv")
trueot_df = pd.read_csv("data/TrueOT_1806uniqueTriplet_gRNA_OT_label.csv")

# CRITICAL: Must match training exactly
MAX_LEN = max(proxy_df.gRNA.str.len().max(), proxy_df.OT.str.len().max())

# gRNA indexing
all_gRNAs = np.load("stage1/all_gRNAs.npy", allow_pickle=True)
gRNA_to_idx = {g: i for i, g in enumerate(all_gRNAs)}

# Loading embeddings
if USE_STAGE1:
    print("Loading Stage-1 sgRNA embeddings...")
    sg_embeddings = torch.load("stage1/sg_embeddings.pt", map_location=device)
    sg_dim = sg_embeddings.shape[1]
else:
    sg_embeddings = torch.zeros(len(all_gRNAs), 768).to(device)
    sg_dim = 768

# Initialization
model = Stage2Model(sg_dim).to(device)
mode_str = 'full' if USE_STAGE1 else 'baseline'
ckpt_path = f"checkpoints/best_model_{mode_str}_seed{args.seed}.pt"

if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

# Load checkpoint safely
ckpt = torch.load(ckpt_path, map_location=device)
state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
model.load_state_dict(state_dict)
model.eval()
print(f"Loaded checkpoint: {ckpt_path}")

# Dataloader
test_loader = DataLoader(
    OffTargetDataset(trueot_df, sg_embeddings, gRNA_to_idx, MAX_LEN, train=False),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Evaluation (MC Dropout)
print(f"Running MC Dropout Inference (T=30) for {mode_str}...")
mean_pred, std_pred = mc_dropout(model, test_loader, device, T=30)
y_true = trueot_df.label.values

test_auc = roc_auc_score(y_true, mean_pred)
test_aupr = average_precision_score(y_true, mean_pred)

print(f"\n=== Result (Seed {args.seed}) ===")
print(f"AUC  : {test_auc:.4f}")
print(f"AUPR : {test_aupr:.4f}")

# Saving results for benchmark_stats.py
# We save the raw predictions so benchmark_stats.py can perform subgroup analysis
out_path = f"results/{mode_str}_seed{args.seed}_results.npy"
np.save(out_path, {
    "auc": test_auc,
    "aupr": test_aupr,
    "mean_pred": mean_pred.astype(np.float32),
    "std_pred": std_pred.astype(np.float32),
    "y_true": y_true
})
print(f"Final results saved to {out_path}")
