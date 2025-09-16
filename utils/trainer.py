import os
import torch
from torch.nn.utils import clip_grad_norm_
import config
from collections import defaultdict

device = torch.device(config.TRAINING_PARAMS["device"])

class Trainer:
    """
    Generic Model Trainer Class.
    """
    def __init__(self, model, loss_fn, optimizer, scheduler, train_loader, val_loader):
        """
        Args:
            model           (nn.Module): Model
            loss_fn         (nn.Module): Loss function
            optimizer       (Optimizer): Optimizer
            scheduler     (LRScheduler): Learning rate scheduler
            train_loader   (DataLoader): DataLoader for training data
            val_loader     (DataLoader): DataLoader for validation data
        """
        self.model        = model
        self.loss_fn      = loss_fn
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.train_loader = train_loader
        self.val_loader   = val_loader
        
        self.num_epochs   = config.TRAINING_PARAMS["num_epochs"]
        self.min_val_loss = torch.inf
        self.history      = defaultdict(list)

    def train(self):

        try:
            for epoch in range(self.num_epochs):
                train_loss = self._train_epoch()
                val_loss   = self._validate()
                
                current_lr = self.optimizer.param_groups[0]["lr"]
                if current_lr < 1e-7:
                    raise ValueError("Learning rate too low to continue...terminating")

                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self._log()

                print(f"Epoch: {epoch+1}/{self.num_epochs}, TrainLoss: {train_loss:.4f}, ValLoss: {val_loss:.4f}, LR: {current_lr:.2e}")

                # save checkpoint at best val_loss
                if val_loss < self.min_val_loss:
                    self._save_state()
                    print(f"  Saved checkpoint for epoch {epoch+1}")
                    self.min_val_loss = val_loss

                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                elif isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                    self.scheduler.step()

        except KeyboardInterrupt:
            self._log()

    def _train_epoch(self):
        """
        Training Loop
        """
        self.model.train()
        total_loss = 0

        for inputs, targets in self.train_loader:
            inputs  = inputs.to(device)
            targets = targets.to(device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss    = self.loss_fn(outputs, targets)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _validate(self):
        """
        Validation Loop
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():

            for inputs, targets, in self.val_loader:
                inputs  = inputs.to(device)
                targets = targets.to(device)
                outputs = self.model(inputs)
                loss    = self.loss_fn(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)
    
    def _log(self):
        """
        Log progress
        """
        torch.save(
            self.history, 
            os.path.join(config.LOGS_DIR, "history.pth")
        )

    def _save_state(self):
        """
        Save model.state_dict()
        """
        torch.save(
            self.model.state_dict(), 
            os.path.join(config.CKPTS_DIR, "best_model.pth")
        )