import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import wandb
from dataclasses import dataclass

from data import get_loaders, DataConfig
from model import ConvMath, ConvMathConfig
from utils import evaluate_metrics


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    clip_grad: float = 5.0
    num_epochs = 20

    scheduler_type: str = "step"        # "step", "cosine", "plateau"
    scheduler_step: int = 3
    scheduler_gamma: float = 0.8
    scheduler_patience: int = 3         # for plateau scheduler

    optimizer_type: str = "adamw"       # "sgd", "adam", "adamw"
    momentum: float = 0.9

    save_dir: str = "checkpoints"
    early_stopping_patience: int = 5
    validation_freq: int = 1            # validate every N epochs
    save_freq: int = 1                  # save checkpoint every N epochs
    
    log_freq: int = 16                  # log every N batches
    wandb_project: str = "ConvMath"


class Trainer:
    def __init__(self, model, vocab, config: TrainingConfig, device='cuda'):
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        self.config = config
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        self.save_dir = config.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.best_metric = 0.0
        self.patience_counter = 0
        self.global_step = 0

    def _create_optimizer(self):
        params = self.model.parameters()

        if self.config.optimizer_type.lower() == "sgd":
            return optim.SGD(params, lr=self.config.lr, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        elif self.config.optimizer_type.lower() == "adam":
            return optim.Adam(params, lr=self.config.lr, weight_decay=self.config.weight_decay)
        elif self.config.optimizer_type.lower() == "adamw":
            return optim.AdamW(params, lr=self.config.lr, weight_decay=self.config.weight_decay)
            
        raise ValueError(f"Unknown optimizer: {self.config.optimizer_type}")

    def _create_scheduler(self):
        if self.config.scheduler_type.lower() == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=self.config.scheduler_step, 
                gamma=self.config.scheduler_gamma
            )
        elif self.config.scheduler_type.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.num_epochs
            )
        elif self.config.scheduler_type.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='max',  # maximize BLEU
                patience=self.config.scheduler_patience,
                factor=self.config.scheduler_gamma
            )
        else:
            return None

    def _step(self, batch, train=True):
        imgs, tgts = [x.to(self.device) for x in batch]
        if imgs.numel() == 0:
            return None

        logits = self.model(imgs, tgts[:, :-1])
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgts[:, 1:].reshape(-1))

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.config.clip_grad > 0:
                clip_grad_norm_(self.model.parameters(), self.config.clip_grad)
            
            self.optimizer.step()
            self.global_step += 1

        return loss.item()

    def train_epoch(self, loader, epoch):
        self.model.train()
        losses = []
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            loss = self._step(batch, train=True)
            
            if loss is not None:
                losses.append(loss)
                avg_loss = sum(losses) / len(losses)
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")
                
                if batch_idx % self.config.log_freq == 0:
                    wandb.log({
                        "train_batch_loss": loss,
                        "train_avg_loss": avg_loss,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    })
        
        # Step scheduler if not plateau
        if self.scheduler and self.config.scheduler_type.lower() != "plateau":
            self.scheduler.step()
            
        return sum(losses) / max(1, len(losses))

    def evaluate(self, loader):
        self.model.eval()
        losses = []
        
        for imgs, tgts in tqdm(loader, desc="Evaluating", leave=False):
            imgs, tgts = imgs.to(self.device), tgts.to(self.device)
            if imgs.numel() == 0:
                continue
                
            logits = self.model(imgs, tgts[:, :-1])
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgts[:, 1:].reshape(-1))
            losses.append(loss.item())

        metrics = evaluate_metrics(self.vocab, self.model, loader, device=self.device)
        avg_loss = sum(losses) / max(1, len(losses))
        
        return avg_loss, metrics

    def save_checkpoint(self, epoch, val_loss, val_metrics, best=False):
        fname = "best.pt" if best else f"epoch_{epoch:03d}.pt"
        path = os.path.join(self.save_dir, fname)
        
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "vocab": self.vocab.itos,
            "config": self.config,
            "best_metric": self.best_metric,
        }
        
        if self.scheduler:
            checkpoint["scheduler_state"] = self.scheduler.state_dict()
            
        torch.save(checkpoint, path)
        print(f"{'Best model' if best else 'Checkpoint'} saved to {path}")

    def should_stop_early(self, current_metric):
        """Check if training should stop early based on validation metric."""
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
                return True
        return False

    def load_checkpoint(self, path):
        """Load checkpoint and resume training."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        if self.scheduler and "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            
        self.global_step = checkpoint.get("global_step", 0)
        self.best_metric = checkpoint.get("best_metric", 0.0)
        
        return checkpoint["epoch"]


def train_model(train_loader, val_loader, vocab, config: TrainingConfig, device="cuda", resume_from=None):
    """Main training function."""
    
    model_config = ConvMathConfig(vocab_size=len(vocab))
    model = ConvMath(model_config)
    trainer = Trainer(model, vocab, config, device=device)
    
    start_epoch = 1
    if resume_from:
        print(f"Resuming training from {resume_from}")
        start_epoch = trainer.load_checkpoint(resume_from) + 1
    
    wandb.init(
        project=config.wandb_project,
        config={
            **config.__dict__,
            "model_config": model_config.__dict__,
            "batch_size": train_loader.batch_size,
        }
    )
    
    print(f"Training for {config.num_epochs} epochs starting from epoch {start_epoch}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    try:
        for epoch in range(start_epoch, config.num_epochs + 1):
            train_loss = trainer.train_epoch(train_loader, epoch)
            
            should_validate = epoch % config.validation_freq == 0
            if should_validate:
                val_loss, val_metrics = trainer.evaluate(val_loader)
                
                if trainer.scheduler and config.scheduler_type.lower() == "plateau":
                    trainer.scheduler.step(val_metrics["bleu"])
                
                log_data = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_bleu": val_metrics["bleu"],
                    "val_edit_distance": val_metrics["edit_distance"],
                    "val_exact_match": val_metrics["exact_match"],
                }
                wandb.log(log_data)
                
                print(
                    f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                    f"BLEU={val_metrics['bleu']:.3f}, EditDist={val_metrics['edit_distance']:.4f}, "
                    f"ExactMatch={val_metrics['exact_match']:.3f}"
                )
                
                is_best = val_metrics["bleu"] > trainer.best_metric
                if is_best:
                    trainer.best_metric = val_metrics["bleu"]
                    trainer.save_checkpoint(epoch, val_loss, val_metrics, best=True)
                
                if trainer.should_stop_early(val_metrics["bleu"]):
                    break
            
            if epoch % config.save_freq == 0:
                trainer.save_checkpoint(epoch, 0, {}, best=False)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
    finally:
        wandb.finish()
    
    return model, trainer


if __name__ == "__main__":
    data_config = DataConfig(batch_size=32)
    train_loader, val_loader, test_loader, vocab = get_loaders(data_config)
    
    training_config = TrainingConfig()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, trainer = train_model(
        train_loader, val_loader, vocab, 
        config=training_config, 
        device=device
    )