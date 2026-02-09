import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc, precision_recall_curve

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

# Create results directory immediately
os.makedirs("results", exist_ok=True)

# Loading data
# We load proxy_df only to ensure MAX_LEN is identical to training
proxy_df = pd.read_csv("data/Proxy_TrainCV.csv")
trueot_df = pd.read_csv("data/TrueOT_1806uniqueTriplet_gRNA_OT_label.csv")

# CRITICAL: Must match training exactly
MAX_LEN = max(proxy_df.gRNA.str.len().max(), proxy_df.OT.str.len().max())

# gRNA indexing (must be identical to training)
all_gRNAs = np.load("stage1/all_gRNAs.npy", allow_pickle=True)
gRNA_to_idx = {g: i for i, g in enumerate(all_gRNAs)}

# Loading embeddings
if USE_STAGE1:
    print("Loading Stage-1 sgRNA embeddings...")
    sg_embeddings = torch.load("stage1/sg_embeddings.pt", map_location=device)
    sg_dim = sg_embeddings.shape[1]
else:
    sg_embeddings = torch.zeros(len(all_gRNAs), 768)
    sg_dim = 768

# Initialization
model = Stage2Model(sg_dim).to(device)

ckpt_path = f"checkpoints/best_model_{'full' if USE_STAGE1 else 'baseline'}_seed{args.seed}.pt"

if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded checkpoint: {ckpt_path}")

# Dataloader
test_loader = DataLoader(
    OffTargetDataset(trueot_df, sg_embeddings, gRNA_to_idx, MAX_LEN, train=False),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Evaluation (MC Dropout)
print("Running MC Dropout Inference (T=30)...")
# We use MC Dropout as our primary predictor because it provides 
# a more generalized mean estimate than a single forward pass.
mean_pred, std_pred = mc_dropout(model, test_loader, device, T=30)
y_true = trueot_df.label.values

test_auc = roc_auc_score(y_true, mean_pred)
test_aupr = average_precision_score(y_true, mean_pred)

print("\n=== TRUEOT GENERALIZATION (Gated + MC Dropout) ===")
print(f"AUC  : {test_auc:.4f}")
print(f"AUPR : {test_aupr:.4f}")

# Visualization
# Roc curve
fpr, tpr, _ = roc_curve(y_true, mean_pred)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC (AUC = {test_auc:.3f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve on TrueOT (Out-of-Distribution)")
plt.legend(loc="lower right")
plt.savefig("results/roc_trueot.png")

precision, recall, _ = precision_recall_curve(y_true, mean_pred)
plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label=f"PR (AUPR = {test_aupr:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve on TrueOT")
plt.legend(loc="upper right")
plt.savefig("results/pr_trueot.png")

# Uncertainty rejection
def reject_by_uncertainty(y_t, m_p, s_p, reject_frac):
    if reject_frac == 0: return roc_auc_score(y_t, m_p), average_precision_score(y_t, m_p)
    threshold = np.percentile(s_p, 100 * (1 - reject_frac))
    keep = s_p < threshold
    return roc_auc_score(y_t[keep], m_p[keep]), average_precision_score(y_t[keep], m_p[keep])

print("\nUncertainty Rejection Analysis:")
for r in [0.0, 0.1, 0.2, 0.3]:
    auc_r, aupr_r = reject_by_uncertainty(y_true, mean_pred, std_pred, r)
    print(f"Reject Top {int(r*100):>2}% Unreliable | AUC: {auc_r:.3f} | AUPR: {aupr_r:.3f}")

# Saving results
out_path = f"results/{'full' if USE_STAGE1 else 'baseline'}_seed{args.seed}_results.npy"
np.save(out_path, {
    "auc": test_auc,
    "aupr": test_aupr,
    "mean_pred": mean_pred,
    "std_pred": std_pred,
    "y_true": y_true
})
print(f"\nFinal results saved to {out_path}")
