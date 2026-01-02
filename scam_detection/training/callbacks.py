"""
Custom PyTorch Lightning callbacks for training.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback


class PlottingCallback(Callback):
    """Callback to save training plots."""

    def __init__(self, save_dir: str = "plots/train"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.train_f1s = []
        self.val_f1s = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Collect metrics
        if "train_loss_epoch" in trainer.logged_metrics:
            self.train_losses.append(trainer.logged_metrics["train_loss_epoch"].item())
        if "train_acc_epoch" in trainer.logged_metrics:
            self.train_accs.append(trainer.logged_metrics["train_acc_epoch"].item())
        if "train_f1_epoch" in trainer.logged_metrics:
            self.train_f1s.append(trainer.logged_metrics["train_f1_epoch"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        # Collect validation metrics
        if "val_loss" in trainer.logged_metrics:
            self.val_losses.append(trainer.logged_metrics["val_loss"].item())
        if "val_acc" in trainer.logged_metrics:
            self.val_accs.append(trainer.logged_metrics["val_acc"].item())
        if "val_f1" in trainer.logged_metrics:
            self.val_f1s.append(trainer.logged_metrics["val_f1"].item())

    def on_train_end(self, trainer, pl_module):
        # Save plots
        epochs = range(1, len(self.train_losses) + 1)

        # Loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses, "b-", label="Training Loss")
        if self.val_losses:
            plt.plot(epochs, self.val_losses, "r-", label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.save_dir / "loss.png")
        plt.close()

        # Accuracy plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_accs, "b-", label="Training Accuracy")
        if self.val_accs:
            plt.plot(epochs, self.val_accs, "r-", label="Validation Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(self.save_dir / "accuracy.png")
        plt.close()

        # F1 Score plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_f1s, "b-", label="Training F1")
        if self.val_f1s:
            plt.plot(epochs, self.val_f1s, "r-", label="Validation F1")
        plt.title("Training and Validation F1 Score")
        plt.xlabel("Epochs")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.savefig(self.save_dir / "f1_score.png")
        plt.close()
