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
    """Calculates standard classification metrics."""
    auc_score = roc_auc_score(y_true, y_prob)
    aupr_score = average_precision_score(y_true, y_prob)
    return auc_score, aupr_score

@torch.no_grad()
def mc_dropout(model, loader, device, T=30):
    """
    Performs Monte Carlo Dropout inference.
    By keeping the model in .train() mode, we sample from the predictive 
    distribution to estimate epistemic uncertainty.
    """
    model.train()  # Critical: keeps dropout layers active
    all_runs = []

    for t in range(T):
        run_preds = []
        for b in loader:
            # Move batch to device
            for k in b:
                if isinstance(b[k], torch.Tensor):
                    b[k] = b[k].to(device)

            logits = model(
                b["pair"],
                b["mv"],
                b["pam"],
                b["sg_emb"]
            )
            
            probs = torch.sigmoid(logits).cpu().numpy()
            run_preds.append(probs)
        
        all_runs.append(np.concatenate(run_preds))

    model.eval() # Reset to eval mode after sampling
    
    # Shape: [T, Num_Samples]
    all_runs = np.stack(all_runs)
    
    # mean_pred: average probability across T trials
    # std_pred: uncertainty/variance across T trials
    mean_pred = all_runs.mean(axis=0)
    std_pred = all_runs.std(axis=0)
    
    return mean_pred, std_pred

def compute_roc_curve(y_true, y_prob):
    """Returns FPR, TPR, and AUC for plotting."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def compute_pr_curve(y_true, y_prob):
    """Returns Precision, Recall, and AP for plotting."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    return precision, recall, ap
