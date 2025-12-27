## Datasets

This project uses two datasets:

1. **Proxy dataset**
   - Used for training and validation
   - Source: prior CRISPR off-target prediction studies
   - File expected: `Proxy_TrainCV.csv`

2. **TrueOT dataset**
   - Used exclusively for external generalization evaluation
   - Source: experimentally validated off-target sites
   - File expected: `TrueOT_1806uniqueTriplet_gRNA_OT_label.csv`

Due to licensing and redistribution restrictions, the datasets are **not included** in this repository.

Please download the datasets from their original sources and place them in this directory before running the code.
