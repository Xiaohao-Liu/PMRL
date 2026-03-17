import torch
import numpy as np

def eigenvalue_computation_pmcl(modalities):
    
    stacked_reps = torch.stack(modalities, dim=-1) # [batch_size, dim, num_reps]
    
    # eigvals, _ = torch.linalg.eigh(G.float()) # [batch_size, num_reps]
    U_V, S_V, W_V = torch.linalg.svd(stacked_reps, full_matrices=True)
    
    return U_V, S_V
    