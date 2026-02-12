import random
import numpy as np
import torch
import os

def set_seed(seed=42):
    # Python-level reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    # Torch CPU + CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Enforcing strict determinism
    torch.use_deterministic_algorithms(True)
