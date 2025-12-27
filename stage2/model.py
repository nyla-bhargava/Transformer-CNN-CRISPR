import torch
import torch.nn as nn

class Stage2Model(nn.Module):
    def __init__(self, sg_dim):
        super().__init__()

        self.input_proj = nn.Linear(10, 128)

        self.cnn = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU()
        )

        enc = nn.TransformerEncoderLayer(128, 4, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, 2)

        self.sg_proj = nn.Linear(sg_dim, 128)
        self.dropout = nn.Dropout(0.3)
        self.cls = nn.Linear(256, 1)

    def forward(self, pair, mv, pam, sg_emb):
        x = torch.cat([pair, mv.unsqueeze(-1), pam.unsqueeze(-1)], dim=-1)
        x = self.input_proj(x)
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        x = self.tr(x)
        pooled = x.mean(1)
        fused = torch.cat([pooled, self.sg_proj(sg_emb)], 1)
        fused = self.dropout(fused)
        return self.cls(fused).squeeze(-1)
