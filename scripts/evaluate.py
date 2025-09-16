import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import config
from utils.dataset import PointDataset
from models.set_transformer import SetTransformer

def main():
        
    # Set device
    device = torch.device(config.TRAINING_PARAMS["device"])

    # Initialize datasets and dataloaders
    val_data = PointDataset(split="test", augment=False)

    val_loader = DataLoader(
        dataset=val_data,
        batch_size=config.TRAINING_PARAMS["batch_size"],
        shuffle=False,
        pin_memory=True,
    )

    # Model
    model = SetTransformer(**config.MODEL_PARAMS).to(device)
    model_path = os.path.join(config.CKPTS_DIR, "best_model.pth")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    correct = 0

    for points, labels in val_loader:
        points   = points.to(device)
        labels   = labels.to(device)
        outputs  = model(points).squeeze()
        preds    = torch.argmax(torch.softmax(outputs, dim=1), 1)
        correct += torch.sum(preds==labels)

    print(f"Accuracy: {100*correct/len(val_data):.2f}%")
    
    # Plot Loss History
    plot_path    = os.path.join(config.LOGS_DIR, "loss_history.png")
    history_path = os.path.join(config.LOGS_DIR, "history.pth")
    
    history      = torch.load(history_path, weights_only=False)

    fig, ax = plt.subplots()
    plt.plot(history["train_loss"], label="training")
    plt.plot(history["val_loss"], label="validation")
    plt.legend(loc="upper right")
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Average CE Loss")
    plt.savefig(plot_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    main()