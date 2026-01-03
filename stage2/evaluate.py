import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve

from stage2.dataset import OffTargetDataset
from stage2.model import Stage2Model
from utils.metrics import mc_dropout
from utils.seed import set_seed

from stage1.embeddings import compute_sg_embeddings


# SETUP
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_STAGE1 = True   # MUST MATCH train.py
BATCH_SIZE = 128

# LOAD DATA
proxy_df = pd.read_csv("data/Proxy_TrainCV.csv")
trueot_df = pd.read_csv("data/TrueOT_1806uniqueTriplet_gRNA_OT_label.csv")

MAX_LEN = max(
    proxy_df.gRNA.str.len().max(),
    proxy_df.OT.str.len().max()
)

# gRNA indexing (must be identical)
all_gRNAs = pd.concat([
    proxy_df.gRNA,
    trueot_df.gRNA
]).unique()

gRNA_to_idx = {g: i for i, g in enumerate(all_gRNAs)}

# LOAD STAGE-1 EMBEDDINGS
if USE_STAGE1:
    print("Computing Stage-1 sgRNA embeddings for evaluation...")
    sg_embeddings = compute_sg_embeddings(list(all_gRNAs), device)
    sg_dim = sg_embeddings.shape[1]
    print(f"Embeddings computed: {sg_embeddings.shape}")
else:
    sg_embeddings = torch.zeros(len(all_gRNAs), 768)
    sg_dim = 768

# MODEL
model = Stage2Model(sg_dim).to(device)
model.load_state_dict(torch.load("best_stage2.pt"))
model.eval()

# TRUEOT DATALOADER
test_loader = DataLoader(
    OffTargetDataset(
        trueot_df,
        sg_embeddings,
        gRNA_to_idx,
        MAX_LEN
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# STANDARD EVALUATION
y_true, y_prob = [], []

with torch.no_grad():
    for b in test_loader:
        for k in b:
            b[k] = b[k].to(device)

        logits = model(
            b["pair"],
            b["mv"],
            b["pam"],
            b["sg_emb"]
        )

        y_prob.append(torch.sigmoid(logits).cpu().numpy())
        y_true.append(b["label"].cpu().numpy())

y_true = np.concatenate(y_true)
y_prob = np.concatenate(y_prob)

test_auc = roc_auc_score(y_true, y_prob)
test_aupr = average_precision_score(y_true, y_prob)

print("\n=== TRUEOT GENERALIZATION ===")
print(f"AUC  : {test_auc:.4f}")
print(f"AUPR : {test_aupr:.4f}")

# MC DROPOUT
mean_pred, std_pred = mc_dropout(
    model,
    test_loader,
    device,
    T=30
)

np.save("results/trueot_pred_mean.npy", mean_pred)
np.save("results/trueot_pred_uncertainty.npy", std_pred)

# SAVE MODEL
torch.save(
    torch.load("best_stage2.pt"),
    "stage2_with_stage1.pth" if USE_STAGE1 else "stage2_no_stage1.pth"
)

# ROC CURVE
fpr, tpr, _ = roc_curve(y_true, mean_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve on TrueOT")
plt.legend()
plt.tight_layout()
plt.savefig("results/roc_trueot.png")
plt.show()

# PR CURVE
precision, recall, _ = precision_recall_curve(y_true, mean_pred)
ap = average_precision_score(y_true, mean_pred)

plt.figure()
plt.plot(recall, precision, label=f"PR (AUPR = {ap:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve on TrueOT")
plt.legend()
plt.tight_layout()
plt.savefig("results/pr_trueot.png")
plt.show()
