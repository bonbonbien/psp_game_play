import gc
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from tqdm import tqdm
import torch
from torch import optim, Tensor
import torch.nn.functional as F

from base_trainer import BaseTrainer

class CustTrainer(BaseTrainer):
    """Main trainer.

    Parameters:
        cfg: experiment configuration
        model: model instance
        loss_fn: loss criterion
        optimizer: optimization algorithm
        lr_scheduler: learning rate scheduler
        train_loader: training data loader
        eval_loader: validation data loader
    """

    def __init__(self, cfg, model, loss_fn, optimizer, lr_skd, evaluator, 
                train_loader, eval_loader):
        super(CustTrainer, self).__init__(
            cfg, model, loss_fn, optimizer, lr_skd, evaluator
        )
        self.cat_feats = [feat for feat in cfg.FEATS if feat in cfg.CAT_FEATS]
        self.train_loader = train_loader
        self.eval_loader = eval_loader if eval_loader else train_loader

    def _train_epoch(self):
        """Run training process for one epoch.

        Return:
            train_loss_avg: average training loss over batches
        """
        train_loss_total = 0

        self.model.train()
        for i, batch_data in enumerate(tqdm(self.train_loader)):
            # self.optimizer.zero_grad(set_to_none=True)
            self.optimizer.zero_grad()

            # Retrieve batched raw data
            x_num = batch_data["X_num"].to(self.device)
            x_cat = batch_data["X_cat"].long().to(self.device)
            x_cat_tuple = torch.split(x_cat, 1, dim=2)
            x_cat_tuple = [cur_cat.squeeze(2) for cur_cat in x_cat_tuple]
            
            y = batch_data["y"].to(self.device)
            src_mask = batch_data["mask"].to(self.device)

            # Forward pass
            output = self.model(x_num, *x_cat_tuple, src_mask)
            self._iter += 1
            
            # Apply mask to target
            # mask = y!=-1
            # y = y[mask]
            # output = output[mask]

            # Derive loss
            loss = self.loss_fn(output, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            train_loss_total += loss.item()

            # Free mem.
            del x_num, x_cat, x_cat_tuple, y, output
            _ = gc.collect()

        train_loss_avg = train_loss_total / len(self.train_loader)

        return train_loss_avg

    @torch.no_grad()
    def _eval_epoch(
        self,
        return_output: bool = False,
        test: bool = False,
    ) -> Tuple[float, Dict[str, float], Optional[Tensor]]:
        """Run evaluation process for one epoch.

        Parameters:
            return_output: whether to return inference result of model
            test: always ignored, exists for compatibility

        Return:
            eval_loss_avg: average evaluation loss over batches
            eval_result: evaluation performance report
            y_pred: inference result
        """
        eval_loss_total = 0
        y_true = None
        y_pred = None

        self.model.eval()
        y_true = []
        y_pred = []
        for i, batch_data in enumerate(self.eval_loader):
            # Retrieve batched raw data
            x_num = batch_data["X_num"].to(self.device)
            x_cat = batch_data["X_cat"].long().to(self.device)
            x_cat_tuple = torch.split(x_cat, 1, dim=2)
            x_cat_tuple = [cur_cat.squeeze(2) for cur_cat in x_cat_tuple]
            
            y = batch_data["y"].to(self.device)
            src_mask = batch_data["mask"].to(self.device)

            # Forward pass
            output = self.model(x_num, *x_cat_tuple, src_mask)
            
            # Apply mask to target
            # mask = y!=-1
            # y = y[mask]
            # output = output[mask]

            # Derive loss
            loss = self.loss_fn(output, y)
            eval_loss_total += loss.item()

            # Record batched output
            y_true.append(y.detach().cpu())
            y_pred.append(output.detach().cpu())

            del x_num, x_cat, x_cat_tuple, y, output
            _ = gc.collect()
        
        print('@@', len(y_true), torch.cat(y_true).shape)

        y_true = torch.cat(y_true).reshape(-1, self.cfg.N_QNS)
        y_pred = torch.cat(y_pred).reshape(-1, self.cfg.N_QNS)
        
        print('##', y_true.shape)


        y_pred = F.sigmoid(y_pred)  # Tmp. workaround (because loss has built-in sigmoid)
        eval_loss_avg = eval_loss_total / len(self.eval_loader)
        eval_result = self.evaluator.evaluate(y_true, y_pred)

        if return_output:
            return eval_loss_avg, eval_result, y_pred
        else:
            return eval_loss_avg, eval_result, None
