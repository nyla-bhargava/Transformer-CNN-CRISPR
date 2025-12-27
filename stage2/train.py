import torch, pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from utils.seed import set_seed
from stage1.embeddings import compute_sg_embeddings
from stage2.dataset import OffTargetDataset
from stage2.model import Stage2Model

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_STAGE1 = True   # change to False for ablation
EPOCHS = 30

proxy_df = pd.read_csv("data/Proxy_TrainCV.csv")
trueot_df = pd.read_csv("data/TrueOT_1806uniqueTriplet_gRNA_OT_label.csv")

train_df, val_df = train_test_split(
    proxy_df,
    test_size=0.15,
    stratify=proxy_df["label"],
    random_state=42
)

MAX_LEN = max(proxy_df.gRNA.str.len().max(),
              proxy_df.OT.str.len().max())

all_gRNAs = pd.concat([
    train_df.gRNA, val_df.gRNA, trueot_df.gRNA
]).unique()

gRNA_to_idx = {g: i for i, g in enumerate(all_gRNAs)}

if USE_STAGE1:
    sg_embeddings = compute_sg_embeddings(list(all_gRNAs), device)
    sg_dim = sg_embeddings.shape[1]
else:
    sg_embeddings = torch.zeros(len(all_gRNAs), 768)
    sg_dim = 768

model = Stage2Model(sg_dim).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=2e-4, weight_decay=1e-4
)

def run_epoch(loader, train):
    model.train() if train else model.eval()
    ys, ps, ls = [], [], 0

    for b in loader:
        for k in b:
            b[k] = b[k].to(device)

        if train:
            optimizer.zero_grad()

        logits = model(b["pair"], b["mv"], b["pam"], b["sg_emb"])
        loss = criterion(logits, b["label"])

        if train:
            loss.backward()
            optimizer.step()

        ls += loss.item() * len(b["label"])
        ps.append(torch.sigmoid(logits).detach().cpu().numpy())
        ys.append(b["label"].cpu().numpy())

    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return ls / len(y), roc_auc_score(y, p), average_precision_score(y, p)
