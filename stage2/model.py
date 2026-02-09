import torch
import torch.nn as nn

class Stage2Model(nn.Module):
    def __init__(self, sg_dim):
        super().__init__()

        # sequence branch projection
        self.input_proj = nn.Linear(10, 128)

        # backbone CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU()
        )

        # backbone Transformer
        enc = nn.TransformerEncoderLayer(128, 4, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, 2)

        # Stage-1 projection
        self.sg_proj = nn.Linear(sg_dim, 128)
        
        # Gated fusion layer
        # Takes (seq_repr + sg_repr) and outputs a value between 0 and 1
        self.gate_layer = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(0.3)
        
        # Output layer - now 128 because we blend the two 128-dim vectors
        self.cls = nn.Linear(128, 1)

    def forward(self, pair, mv, pam, sg_emb):
        # 1. Process Sequence Features
        mv = mv.unsqueeze(-1)
        pam = pam.unsqueeze(-1)

        x = torch.cat([pair, mv, pam], dim=2)
        x = self.input_proj(x)
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        x = self.tr(x)

        # Global representation from sequence
        seq_repr = x.mean(dim=1)
        
        # 2. Process Stage-1 Features
        sg_repr = self.sg_proj(sg_emb)

        # 3. GATED FUSION
        # We concatenate them to let the gate see both contexts
        gate_input = torch.cat([seq_repr, sg_repr], dim=1)
        g = self.gate_layer(gate_input)
        
        # Weighted blend: if g is high, we trust sg_repr more
        fused = g * sg_repr + (1 - g) * seq_repr

        # 4. Classification
        fused = self.dropout(fused)
        return self.cls(fused)
