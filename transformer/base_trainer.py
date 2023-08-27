from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from copy import deepcopy
import torch
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader

class BaseTrainer:
    """Base class for all customized trainers.

    Parameters:
        cfg: experiment configuration
        model: model instance
        loss_fn: loss criterion
        optimizer: optimization algorithm
        lr_skd: learning rate scheduler
        evaluator: task-specific evaluator
    """

    def __init__(self, cfg, model, loss_fn, optimizer, lr_skd, evaluator):
        self.cfg = cfg
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_skd = lr_skd
        self.evaluator = evaluator

        self.device = cfg.DEVICE
        self.epochs = cfg.EPOCH
        self.es_patience = cfg.ES_PATIENCE

        # Model checkpoint
        self.ckpt_metric = cfg.CKPT_METRIC

        self._iter = 0
        self._track_best_model = True

    def train_eval(self, proc_id):
        """Run training and evaluation processes for either one fold or
        one random seed (commonly used when training on whole dataset).

        Parameters:
            proc_id: identifier of the current process, indicating
                current fold number or random seed.

        Return:
            best_model: model instance with the best monitored
                objective (e.g., the lowest validation loss)
            y_preds: inference results on different datasets
        """
        best_val_loss = 1e18
        best_epoch = 0
        try:
            best_model = deepcopy(self.model)
        except RuntimeError as e:
            best_model = None
            self._track_best_model = False
            print("In-memory best model tracking is disabled.")

        for epoch in range(self.epochs):
            train_loss = self._train_epoch()
            val_loss, val_result, _ = self._eval_epoch()

            # Adjust learning rate
            if self.lr_skd is not None:
                if isinstance(self.lr_skd, lr_scheduler.ReduceLROnPlateau):
                    self.lr_skd.step(val_loss)
                else:
                    self.lr_skd.step()

            # Track and log process result
            val_metric_msg = ""
            for metric, score in val_result.items():
                val_metric_msg += f"{metric.upper()} {round(score, 4)} | "
            print(f"Epoch{epoch} | Training loss {train_loss:.4f} | Validation loss {val_loss:.4f} | {val_metric_msg}")

            # Record the best checkpoint
            ckpt_metric_val = val_result[self.ckpt_metric]
            ckpt_metric_val = -ckpt_metric_val
            if ckpt_metric_val < best_val_loss:
                print(f"Validation performance improves at epoch {epoch}!!")
                best_val_loss = ckpt_metric_val
                if self._track_best_model:
                    best_model = deepcopy(self.model)
                else:
                    self._save_ckpt()
                best_epoch = epoch
                es_count = 0
            else:
                es_count += 1
                if es_count > self.es_patience:
                    print("Early stopped!")
                    break

        # Run final evaluation
        if not self._track_best_model:
            self._load_best_ckpt()
            best_model = self.model
        else:
            self.model = best_model
        final_prf_report, y_preds = self._eval_with_best()
        self._log_best_prf(final_prf_report)

        return best_model, y_preds

    @abstractmethod
    def _train_epoch(self):
        """Run training process for one epoch.

        Return:
            train_loss_avg: average training loss over batches
                *Note: If multitask is used, returned object will be
                    a dictionary containing losses of subtasks and the
                    total loss.
        """
        raise NotImplementedError

    @abstractmethod
    def _eval_epoch(self, return_output=False, test=False):
        """Run evaluation process for one epoch.

        Parameters:
            return_output: whether to return inference result of model
            test: if evaluation is run on test set, set it to True
                *Note: The setting is mainly used to disable DAE doping
                    during test phase.

        Return:
            eval_loss_avg: average evaluation loss over batches
            eval_result: evaluation performance report
            y_pred: inference result
        """
        raise NotImplementedError

    def _eval_with_best(self):
        """Run final evaluation process with the best checkpoint.

        Return:
            final_prf_report: performance report of final evaluation
            y_preds: inference results on different datasets
        """
        final_prf_report = {}
        y_preds = {}

        self._disable_shuffle()
        dataloaders = {"train": self.train_loader}
        if self.eval_loader is not None:
            dataloaders["val"] = self.eval_loader

        for datatype, dataloader in dataloaders.items():
            self.eval_loader = dataloader
            eval_loss, eval_result, y_pred = self._eval_epoch(return_output=True)
            final_prf_report[datatype] = eval_result
            y_preds[datatype] = y_pred

        return final_prf_report, y_preds

    def _disable_shuffle(self):
        """Disable shuffle in train dataloader for final evaluation."""
        self.train_loader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=False,  # Reset shuffle to False
            num_workers=self.train_loader.num_workers,
            collate_fn=self.train_loader.collate_fn,
        )

    def _save_ckpt(self, proc_id: int = 0, save_best_only: bool = True) -> None:
        """Save checkpoints.

        Parameters:
            proc_id: identifier of the current process, indicating
                current fold number or random seed
            save_best_only: only checkpoint of the best epoch is saved

        Return:
            None
        """
        torch.save(self.model.state_dict(), "model_tmp.pt")

    def _load_best_ckpt(self, proc_id: int = 0) -> None:
        """Load the best model checkpoint for final evaluation.

        The best checkpoint is loaded and assigned to `self.model`.

        Parameters:
            proc_id: identifier of the current process, indicating
                current fold number or random seed

        Return:
            None
        """
        device = torch.device(self.device)
        self.model.load_state_dict(
            torch.load("model_tmp.pt", map_location=device)
        )
        self.model = self.model.to(device)

    def _log_best_prf(self, prf_report: Dict[str, Any]) -> None:
        """Log performance evaluated with the best model checkpoint.

        Parameters:
            prf_report: performance report

        Return:
            None
        """
        import json

        print(">>>>> Performance Report - Best Ckpt <<<<<")
        print(json.dumps(prf_report, indent=4))
        