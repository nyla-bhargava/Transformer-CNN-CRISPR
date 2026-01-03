## Hyperparameters

All experiments were conducted using a fixed set of hyperparameters to ensure fair comparison and reproducibility across model variants.

### Model Architecture
- Architecture: Hybrid CNN–Transformer
- CNN hidden dimension: 128  
- Transformer layers: 2  
- Transformer attention heads: 4  
- DNABERT embedding dimension (Stage-1): 768  
- Dropout rate: 0.3  

### Training Configuration
- Optimizer: AdamW  
- Learning rate: 2e-4  
- Weight decay: 1e-4  
- Loss function: Binary Cross-Entropy  
- Batch size: 64  
- Number of epochs: 30  

### Reproducibility
- Fixed random seed used for all experiments  
- Identical hyperparameters used for all ablation settings  
- TrueOT data was **not** used for training or hyperparameter tuning  

---

## Results

Model performance was evaluated under a strict generalization setting:
- **Proxy dataset** was used for training and validation
- **TrueOT dataset** was used exclusively for external evaluation

### Evaluation Metrics
- ROC-AUC  
- PR-AUC  

### Results Table

| Model Configuration              | ROC-AUC | PR-AUC |
|---------------------------------|---------|--------|
| Stage-2 only (no Stage-1)       | 0.64 ± 0.03  | 0.22 ± 0.03 |
| Full model (Stage-1 + Stage-2)  | 0.704 ± 0.03 | 0.301 ± 0.03 |

**Note:** TrueOT results represent out-of-distribution generalization performance on experimentally validated CRISPR off-target sites.

---

## Notes
- No TrueOT samples were used during training or validation.
- Performance gains are attributed to pretrained sequence representations rather than hyperparameter tuning.
- All results reported here correspond to those presented in the paper.
