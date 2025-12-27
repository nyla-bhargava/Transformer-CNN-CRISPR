import torch
from transformers import AutoTokenizer, AutoModel

@torch.no_grad()
def compute_sg_embeddings(seqs, device):
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6")
    model = AutoModel.from_pretrained("zhihan1996/DNA_bert_6").to(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    embs = []
    for i in range(0, len(seqs), 64):
        batch = seqs[i:i+64]
        toks = tokenizer(batch, padding=True, truncation=True,
                          return_tensors="pt").to(device)
        out = model(**toks)
        embs.append(out.last_hidden_state[:, 0, :].cpu())

    return torch.cat(embs, dim=0)
