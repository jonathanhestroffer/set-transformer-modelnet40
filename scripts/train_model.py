import torch
import random
import numpy as np
from torch.utils.data import DataLoader

import config
from utils.trainer import Trainer
from utils.dataset import PointDataset
from models.set_transformer import SetTransformer

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():

    seed_all()

    # Set device
    device = torch.device(config.TRAINING_PARAMS["device"])

    # Initialize datasets and dataloaders
    train_data = PointDataset(split="train", augment=config.TRAINING_PARAMS["augment"])
    val_data   = PointDataset(split="test", augment=False)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=config.TRAINING_PARAMS["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=config.TRAINING_PARAMS["num_workers"]
    )

    val_loader = DataLoader(
        dataset=val_data,
        batch_size=config.TRAINING_PARAMS["batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=config.TRAINING_PARAMS["num_workers"]
    )

    # Model
    model = SetTransformer(**config.MODEL_PARAMS).to(device)

    # Loss
    loss_fn_cls      = config.LOSS_FN_PARAMS["class"]
    loss_fn_kwargs   = config.LOSS_FN_PARAMS["kwargs"]
    loss_fn          = loss_fn_cls(**loss_fn_kwargs)

    # Optimizer
    optimizer_cls    = config.OPTIM_PARAMS["class"]
    optimizer_kwargs = config.OPTIM_PARAMS["kwargs"]
    optimizer        = optimizer_cls(model.parameters(), **optimizer_kwargs)

    # Scheduler
    scheduler_cls    = config.SCHDLR_PARAMS["class"]
    scheduler_kwargs = config.SCHDLR_PARAMS["kwargs"]
    scheduler        = scheduler_cls(optimizer, **scheduler_kwargs)

    # Trainer
    model_trainer = Trainer(
        model,
        loss_fn,
        optimizer,
        scheduler,
        train_loader,
        val_loader
    )
    model_trainer.train()

if __name__ == "__main__":
    main()