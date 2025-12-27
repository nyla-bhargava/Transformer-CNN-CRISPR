import numpy as np
import torch
from torch.utils.data import Dataset

NUC2IDX = {'A':0,'C':1,'G':2,'T':3}

def one_hot(seq):
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, b in enumerate(seq):
        if b in NUC2IDX:
            arr[i, NUC2IDX[b]] = 1
    return arr

def mismatch_vec(sg, ot):
    L = min(len(sg), len(ot))
    return np.array([0 if sg[i] == ot[i] else 1 for i in range(L)],
                    dtype=np.float32)

def pam_distance_encoding(L, alpha=0.3):
    d = np.arange(L)[::-1]
    return np.exp(-alpha * d).astype(np.float32)

class OffTargetDataset(Dataset):
    def __init__(self, df, sg_embeddings, gRNA_to_idx, MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.sg_embeddings = sg_embeddings
        self.gRNA_to_idx = gRNA_to_idx
        self.MAX_LEN = MAX_LEN

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        sg, ot = r.gRNA, r.OT

        sg_oh = one_hot(sg)
        ot_oh = one_hot(ot)
        mv = mismatch_vec(sg, ot)
        pam = pam_distance_encoding(len(mv))

        def pad(x):
            return np.pad(x, ((0, self.MAX_LEN - len(x)), (0, 0)))

        pair = np.concatenate([pad(sg_oh), pad(ot_oh)], axis=1)
        mv = np.pad(mv, (0, self.MAX_LEN - len(mv)))
        pam = np.pad(pam, (0, self.MAX_LEN - len(pam)))

        return {
            "pair": torch.tensor(pair),
            "mv": torch.tensor(mv),
            "pam": torch.tensor(pam),
            "sg_emb": self.sg_embeddings[self.gRNA_to_idx[sg]],
            "label": torch.tensor(float(r.label))
        }
