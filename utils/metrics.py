import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    auc
)

def compute_basic_metrics(y_true, y_prob):

    auc_score = roc_auc_score(y_true, y_prob)
    aupr_score = average_precision_score(y_true, y_prob)
    return auc_score, aupr_score


@torch.no_grad()
def mc_dropout(model, loader, device, T=30):
    model.train()   # important: enables dropout
    preds = []

    for _ in range(T):
        batch_preds = []
        for b in loader:
            for k in b:
                b[k] = b[k].to(device)

            logits = model(
                b["pair"],
                b["mv"],
                b["pam"],
                b["sg_emb"]
            )
            batch_preds.append(
                torch.sigmoid(logits).cpu().numpy()
            )

        preds.append(np.concatenate(batch_preds))

    model.eval()
    preds = np.stack(preds)

    return preds.mean(0), preds.std(0)


def compute_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def compute_pr_curve(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    return precision, recall, ap
