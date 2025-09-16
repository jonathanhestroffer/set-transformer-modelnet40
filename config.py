import torch
import torch.nn as nn

# --- Paths ---
DATA_DIR      = "ModelNet40"
CKPTS_DIR     = "checkpoints"
LOGS_DIR      = "logs" 
RAW_META_PATH = "metadata_modelnet40.csv"
PRC_META_PATH = "processed_metadata.csv"

# --- Training Parameters ---
TRAINING_PARAMS = {
    "num_pnts"    : 100,    # number of points to sample
    "batch_size"  : 256,     
    "num_epochs"  : 300,   
    "device"      : "cuda",  
    "num_workers" : 0,      
    "augment"     : False
}

# --- Model Parameters ---
MODEL_PARAMS = {
    "input_dim"  : 3,       # (x, y, z)
    "output_dim" : 40,      # num_classes
    "embed_dim"  : 128,     # embedding dimension
    "num_heads"  : 4,       # number of attention heads
    "num_sabs"   : 2,       # number of SAB/ISAB layers
    "num_induce" : 16,      # number of inducing points
    "num_seeds"  : 1,       # number of PMA seeds
    "layer_norm" : False,
}

# --- Loss Function Parameters ---
LOSS_FN_PARAMS = {
    "class" : nn.CrossEntropyLoss,
    "kwargs": {}
}

# --- Optimizer Parameters ---
OPTIM_PARAMS = {
    "class" : torch.optim.AdamW,
    "kwargs": {
        "lr" : 1e-3, 
    }
}

# --- Scheduler Parameters ---
SCHDLR_PARAMS = {
    "class" : torch.optim.lr_scheduler.ReduceLROnPlateau,
    "kwargs": {
        "mode"    : "min", 
        "factor"  : 0.3,
        "patience": 10
    }
}