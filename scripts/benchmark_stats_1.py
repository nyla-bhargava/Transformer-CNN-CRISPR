import torch
import numpy as np
import pandas as pd
import os
from scipy import stats
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from stage2.dataset import OffTargetDataset
from stage2.model import Stage2Model

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = "checkpoints"

trueot_df = pd.read_csv("data/TrueOT_1806uniqueTriplet_gRNA_OT_label.csv")

def get_mismatches(s1, s2):
    return sum(a != b for a, b in zip(s1, s2))

trueot_df["mismatches"] = trueot_df.apply(
    lambda r: get_mismatches(r["gRNA"], r["OT"]), axis=1
)

all_gRNAs = np.load("stage1/all_gRNAs.npy", allow_pickle=True)
gRNA_to_idx = {g: i for i, g in enumerate(all_gRNAs)}
MAX_LEN = max(trueot_df.gRNA.str.len().max(),
              trueot_df.OT.str.len().max())

y_true = trueot_df["label"].values
m_counts = trueot_df["mismatches"].values

# 2. Storage
metrics = ["global", "hard", "top20"]
results = {
    "baseline": {m: [] for m in metrics},
    "stage1": {m: [] for m in metrics}
}

# 3. Model Inference
def get_probs(model_path, use_stage1):
    if use_stage1:
        sg_embs = torch.load("stage1/sg_embeddings.pt", map_location=device)
        sg_dim = sg_embs.shape[1]
    else:
        sg_embs = torch.zeros(len(all_gRNAs), 768).to(device)
        sg_dim = 768

    model = Stage2Model(sg_dim=sg_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    loader = DataLoader(
        OffTargetDataset(
            trueot_df, sg_embs, gRNA_to_idx, MAX_LEN, train=False
        ),
        batch_size=128,
        shuffle=False
    )

    probs = []
    with torch.no_grad():
        for b in loader:
            logits = model(
                b["pair"].to(device),
                b["mv"].to(device),
                b["pam"].to(device),
                b["sg_emb"].to(device)
            ).squeeze()
            probs.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(probs)

# 4. Run Analysis (10 SEEDS)
print("Running analysis across 10 seeds...")

for seed in range(10):
    for mode in ["baseline", "full"]:
        ckpt = f"{checkpoint_dir}/best_model_{mode}_seed{seed}.pt"
        if not os.path.exists(ckpt):
            continue

        probs = get_probs(ckpt, use_stage1=(mode == "full"))

        global_auc = roc_auc_score(y_true, probs)

        hard_mask = m_counts >= 3
        hard_auc = roc_auc_score(y_true[hard_mask], probs[hard_mask])

        top20_idx = np.argsort(probs)[-20:]
        top20_recall = y_true[top20_idx].sum() / y_true.sum()

        key = "stage1" if mode == "full" else "baseline"
        results[key]["global"].append(global_auc)
        results[key]["hard"].append(hard_auc)
        results[key]["top20"].append(top20_recall)

# 5. Statistical Report
def mean_sd(x):
    return np.mean(x), np.std(x)

print("\n" + "="*80)
print("CRISPR OFF-TARGET PREDICTION — STATISTICAL REPORT (N=10)")
print("="*80)
print(f"{'METRIC':<10} | {'BASELINE (mean±sd)':<22} | {'STAGE-1 (mean±sd)':<22} | {'P-VALUE'}")
print("-"*85)

for m in metrics:
    b = results["baseline"][m]
    s = results["stage1"][m]

    t, p = stats.ttest_rel(s, b)
    sig = "***" if p < 0.01 else "*" if p < 0.05 else "(ns)"

    b_mean, b_sd = mean_sd(b)
    s_mean, s_sd = mean_sd(s)

    print(
        f"{m.upper():<10} | "
        f"{b_mean:.4f}±{b_sd:.3f} | "
        f"{s_mean:.4f}±{s_sd:.3f} | "
        f"{p:.5f} {sig}"
    )

# 6. Reliability & Confidence
def ci_95(data):
    mean = np.mean(data)
    sem = stats.sem(data)
    h = sem * stats.t.ppf(0.975, len(data)-1)
    return mean - h, mean + h

print("\n" + "="*80)
print("RELIABILITY & 95% CONFIDENCE INTERVALS")
print("="*80)
print(f"{'MODEL':<10} | {'METRIC':<8} | {'95% CI':<25} | {'RELIABILITY'}")
print("-"*80)

for model in ["baseline", "stage1"]:
    for m in ["global", "hard"]:
        vals = results[model][m]
        low, high = ci_95(vals)
        mean, sd = mean_sd(vals)
        reliability = (1 - (sd / mean)) * 100

        print(
            f"{model.upper():<10} | "
            f"{m.upper():<8} | "
            f"[{low:.4f} - {high:.4f}] | "
            f"{reliability:.2f}%"
        )

print("="*80)
