import time
from pathlib import Path

import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback


class PlottingCallback(Callback):
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
        if "train_loss_epoch" in trainer.logged_metrics:
            self.train_losses.append(trainer.logged_metrics["train_loss_epoch"].item())
        if "train_acc_epoch" in trainer.logged_metrics:
            self.train_accs.append(trainer.logged_metrics["train_acc_epoch"].item())
        if "train_f1_epoch" in trainer.logged_metrics:
            self.train_f1s.append(trainer.logged_metrics["train_f1_epoch"].item())

        self._save_plots(trainer.current_epoch + 1)

    def on_validation_epoch_end(self, trainer, pl_module):
        if "val_loss" in trainer.logged_metrics:
            self.val_losses.append(trainer.logged_metrics["val_loss"].item())
        if "val_acc" in trainer.logged_metrics:
            self.val_accs.append(trainer.logged_metrics["val_acc"].item())
        if "val_f1" in trainer.logged_metrics:
            self.val_f1s.append(trainer.logged_metrics["val_f1"].item())

    def _save_plots(self, current_epoch: int):
        epochs = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses, "b-", label="Training Loss")
        if len(self.val_losses) >= current_epoch:
            plt.plot(
                range(1, current_epoch + 1),
                self.val_losses[:current_epoch],
                "r-",
                label="Validation Loss",
            )
        plt.title(f"Training and Validation Loss (Epoch {current_epoch})")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.save_dir / f"loss_epoch_{current_epoch}.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_accs, "b-", label="Training Accuracy")
        if len(self.val_accs) >= current_epoch:
            plt.plot(
                range(1, current_epoch + 1),
                self.val_accs[:current_epoch],
                "r-",
                label="Validation Accuracy",
            )
        plt.title(f"Training and Validation Accuracy (Epoch {current_epoch})")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(self.save_dir / f"accuracy_epoch_{current_epoch}.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_f1s, "b-", label="Training F1")
        if len(self.val_f1s) >= current_epoch:
            plt.plot(
                range(1, current_epoch + 1),
                self.val_f1s[:current_epoch],
                "r-",
                label="Validation F1",
            )
        plt.title(f"Training and Validation F1 Score (Epoch {current_epoch})")
        plt.xlabel("Epochs")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.savefig(self.save_dir / f"f1_score_epoch_{current_epoch}.png")
        plt.close()

    def on_train_end(self, trainer, pl_module):
        if not self.train_losses:
            return

        epochs = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses, "b-", label="Training Loss")
        if self.val_losses and len(self.val_losses) == len(self.train_losses):
            plt.plot(epochs, self.val_losses, "r-", label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.save_dir / "loss.png")
        plt.close()

        if self.train_accs:
            plt.figure(figsize=(10, 6))
            acc_epochs = range(1, len(self.train_accs) + 1)
            plt.plot(acc_epochs, self.train_accs, "b-", label="Training Accuracy")
            if self.val_accs and len(self.val_accs) == len(self.train_accs):
                plt.plot(acc_epochs, self.val_accs, "r-", label="Validation Accuracy")
            plt.title("Training and Validation Accuracy")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig(self.save_dir / "accuracy.png")
            plt.close()

        if self.train_f1s:
            plt.figure(figsize=(10, 6))
            f1_epochs = range(1, len(self.train_f1s) + 1)
            plt.plot(f1_epochs, self.train_f1s, "b-", label="Training F1")
            if self.val_f1s and len(self.val_f1s) == len(self.train_f1s):
                plt.plot(f1_epochs, self.val_f1s, "r-", label="Validation F1")
            plt.title("Training and Validation F1 Score")
            plt.xlabel("Epochs")
            plt.ylabel("F1 Score")
            plt.legend()
            plt.savefig(self.save_dir / "f1_score.png")
            plt.close()


class MLflowPlottingCallback(Callback):
    def __init__(self, log_every_n_steps: int = 20, plot_dir: str = "plots/train"):
        self.log_every_n_steps = log_every_n_steps
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.step_count = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.train_f1s = []
        self.val_f1s = []
        self.last_logged_step = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.step_count += 1

        metrics = trainer.callback_metrics

        if "train_loss" in metrics:
            loss_val = metrics["train_loss"]
            if hasattr(loss_val, "item"):
                self.train_losses.append(loss_val.item())
            else:
                self.train_losses.append(float(loss_val))

        if "train_acc" in metrics:
            acc_val = metrics["train_acc"]
            if hasattr(acc_val, "item"):
                self.train_accs.append(acc_val.item())
            else:
                self.train_accs.append(float(acc_val))

        if "train_f1" in metrics:
            f1_val = metrics["train_f1"]
            if hasattr(f1_val, "item"):
                self.train_f1s.append(f1_val.item())
            else:
                self.train_f1s.append(float(f1_val))

        if (
            self.step_count % self.log_every_n_steps == 0
            and self.step_count != self.last_logged_step
        ):
            self.last_logged_step = self.step_count
            self._log_plots_to_mlflow(trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        if "val_loss" in trainer.logged_metrics:
            val_loss = trainer.logged_metrics["val_loss"]
            if hasattr(val_loss, "item"):
                self.val_losses.append(val_loss.item())
            else:
                self.val_losses.append(float(val_loss))

        if "val_acc" in trainer.logged_metrics:
            val_acc = trainer.logged_metrics["val_acc"]
            if hasattr(val_acc, "item"):
                self.val_accs.append(val_acc.item())
            else:
                self.val_accs.append(float(val_acc))

        if "val_f1" in trainer.logged_metrics:
            val_f1 = trainer.logged_metrics["val_f1"]
            if hasattr(val_f1, "item"):
                self.val_f1s.append(val_f1.item())
            else:
                self.val_f1s.append(float(val_f1))

    def _log_plots_to_mlflow(self, trainer):
        if not self.train_losses:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"Training Progress at Step {self.step_count}",
            fontsize=16,
            fontweight="bold",
        )

        steps = range(1, len(self.train_losses) + 1)

        ax1.plot(
            steps,
            self.train_losses,
            "b-",
            linewidth=2,
            label="Training Loss",
            alpha=0.7,
        )
        if self.val_losses:
            steps_per_val = len(self.train_losses) // max(1, len(self.val_losses))
            val_steps = [
                min((i + 1) * steps_per_val, len(self.train_losses))
                for i in range(len(self.val_losses))
            ]
            ax1.plot(
                val_steps,
                self.val_losses,
                "r-",
                linewidth=2,
                marker="o",
                markersize=8,
                label="Validation Loss",
            )
        ax1.set_title("Loss", fontweight="bold")
        ax1.set_xlabel("Training Steps")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        if self.train_accs:
            acc_steps = range(1, len(self.train_accs) + 1)
            ax2.plot(
                acc_steps,
                self.train_accs,
                "g-",
                linewidth=2,
                label="Training Accuracy",
                alpha=0.7,
            )
            if self.val_accs:
                steps_per_val = len(self.train_accs) // max(1, len(self.val_accs))
                val_acc_steps = [
                    min((i + 1) * steps_per_val, len(self.train_accs))
                    for i in range(len(self.val_accs))
                ]
                ax2.plot(
                    val_acc_steps,
                    self.val_accs,
                    "orange",
                    linewidth=2,
                    marker="s",
                    markersize=8,
                    label="Validation Accuracy",
                )
            ax2.set_title("Accuracy", fontweight="bold")
            ax2.set_xlabel("Training Steps")
            ax2.set_ylabel("Accuracy")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(
                0.5,
                0.5,
                "Accuracy data\nnot available yet",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=12,
            )
            ax2.set_title("Accuracy", fontweight="bold")

        if self.train_f1s:
            f1_steps = range(1, len(self.train_f1s) + 1)
            ax3.plot(
                f1_steps,
                self.train_f1s,
                "purple",
                linewidth=2,
                label="Training F1",
                alpha=0.7,
            )
            if self.val_f1s:
                steps_per_val = len(self.train_f1s) // max(1, len(self.val_f1s))
                val_f1_steps = [
                    min((i + 1) * steps_per_val, len(self.train_f1s))
                    for i in range(len(self.val_f1s))
                ]
                ax3.plot(
                    val_f1_steps,
                    self.val_f1s,
                    "brown",
                    linewidth=2,
                    marker="^",
                    markersize=8,
                    label="Validation F1",
                )
            ax3.set_title("F1 Score", fontweight="bold")
            ax3.set_xlabel("Training Steps")
            ax3.set_ylabel("F1 Score")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(
                0.5,
                0.5,
                "F1 data\nnot available yet",
                ha="center",
                va="center",
                transform=ax3.transAxes,
                fontsize=12,
            )
            ax3.set_title("F1 Score", fontweight="bold")

        ax4.axis("off")
        info_text = f"Training Step: {self.step_count}\n\n"
        info_text += "=" * 35 + "\n"
        if self.train_losses:
            info_text += f"Training Loss:      {self.train_losses[-1]:.4f}\n"
        if self.val_losses:
            info_text += f"Validation Loss:    {self.val_losses[-1]:.4f}\n"
        info_text += "-" * 35 + "\n"
        if self.train_accs:
            info_text += f"Training Accuracy:  {self.train_accs[-1]:.4f}\n"
        if self.val_accs:
            info_text += f"Validation Accuracy:{self.val_accs[-1]:.4f}\n"
        info_text += "-" * 35 + "\n"
        if self.train_f1s:
            info_text += f"Training F1 Score:  {self.train_f1s[-1]:.4f}\n"
        if self.val_f1s:
            info_text += f"Validation F1 Score:{self.val_f1s[-1]:.4f}\n"
        info_text += "=" * 35

        ax4.text(
            0.1,
            0.85,
            info_text,
            transform=ax4.transAxes,
            fontsize=11,
            verticalalignment="top",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.6),
        )

        plt.tight_layout()

        plot_path = self.plot_dir / f"training_progress_step_{self.step_count}.png"
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close()

        try:
            if hasattr(trainer, "logger") and trainer.logger is not None:
                trainer.logger.experiment.log_artifact(
                    trainer.logger.run_id,
                    str(plot_path),
                    artifact_path="training_plots",
                )

                if self.train_losses:
                    trainer.logger.experiment.log_metric(
                        trainer.logger.run_id,
                        "step_train_loss",
                        self.train_losses[-1],
                        timestamp=int(time.time() * 1000),
                        step=self.step_count,
                    )
                if self.train_accs:
                    trainer.logger.experiment.log_metric(
                        trainer.logger.run_id,
                        "step_train_accuracy",
                        self.train_accs[-1],
                        timestamp=int(time.time() * 1000),
                        step=self.step_count,
                    )
                if self.train_f1s:
                    trainer.logger.experiment.log_metric(
                        trainer.logger.run_id,
                        "step_train_f1",
                        self.train_f1s[-1],
                        timestamp=int(time.time() * 1000),
                        step=self.step_count,
                    )
                if self.val_losses:
                    trainer.logger.experiment.log_metric(
                        trainer.logger.run_id,
                        "step_val_loss",
                        self.val_losses[-1],
                        timestamp=int(time.time() * 1000),
                        step=self.step_count,
                    )
                if self.val_accs:
                    trainer.logger.experiment.log_metric(
                        trainer.logger.run_id,
                        "step_val_accuracy",
                        self.val_accs[-1],
                        timestamp=int(time.time() * 1000),
                        step=self.step_count,
                    )
                if self.val_f1s:
                    trainer.logger.experiment.log_metric(
                        trainer.logger.run_id,
                        "step_val_f1",
                        self.val_f1s[-1],
                        timestamp=int(time.time() * 1000),
                        step=self.step_count,
                    )

                print(
                    f"✓ Logged training progress plot and metrics to "
                    f"MLflow at step {self.step_count}"
                )
            else:
                print(
                    f"⚠ Warning: Trainer logger not available at step {self.step_count}"
                )
        except Exception as e:
            import traceback

            print(f"✗ Failed to log plot to MLflow at step {self.step_count}: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
