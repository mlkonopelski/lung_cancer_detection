import argparse
import datetime
import os
import sys
import time
from timeit import timeit

import numpy as np
import torch
import torch.nn as nn
from data import Luna3DClassificationDataset
from mlmodels.classifiers import base_cnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.config import CONFIG

METRIX_LABEL_NDX=0
METRICS_PRED_NDX=1
METRICS_LOSS_NDX=2
METRICS_SIZE=3

class LunaTrainingApp:
    def __init__(self, log_prefix: str, sys_argv=None, dev: bool = False) -> None:
        
        self.log_prefix = log_prefix
        self.dev = dev
        
        if sys_argv is None:
            sys_argv = sys.argv[1:]
            
        self.device = CONFIG.general.device
        self.pin_memory = True if CONFIG.general.device in ['cuda', 'mps'] else False
        
        # TODO: As additional method to config.yml implement this parser for cmd line arguments
        # parser = argparse.ArgumentParser()
        # parser.add_argument('--num-workers', 
        #                     help='Number of worker processes for background dta loading',
        #                     default=8,
        #                     type=int,)
        # self.cli_args = parser.parse_args(sys_argv)
        # print(self.cli_args)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        
        self.total_training_samples_count = 0
        self.model = self._init_model()
        self.optimizer = self._init_optimizer()
        
        self._init_tensorboard_writers()
        
        
    def _init_model(self):
        
        """Helper method to instantiate ML model. 
        In future (when more models will be developed) 
        the selection will be based on config.yml or cmd line arguments

        Returns:
            _type_: instance of nn.Module model
        """
        model = base_cnn.CNN()
        model = model.to(self.device)
        return model 
    
    def _init_optimizer(self):
        """Helper metod to instantiate optimizer. 
        As there are few possible popular optimizers this method will be
        based on config.yml or cmd line arguments

        Returns:
            _type_: instance of class from torch.optim module
        """
        optimizer = torch.optim.SGD(params=self.model.parameters(),
                                    lr=0.001,
                                    momentum=0.9)
        return optimizer
    
    
    def _init_train_dl(self):
        """_summary_

        Returns:
            _type_: torch DataLoader with 
        """
        
        balance_ratio = 1 if CONFIG.training.balanced else None
        augmentation_dict = CONFIG.training.augmentation if CONFIG.training.augmentation else {}
        
        if not self.dev:
            train_ds = Luna3DClassificationDataset(val_stride=10,
                                   is_val_set=False,
                                   ratio=balance_ratio,
                                   augmentations=augmentation_dict)
        else:
            EXAMPLE_UID = '1.3.6.1.4.1.14519.5.2.1.6279.6001.277445975068759205899107114231'
            train_ds = Luna3DClassificationDataset(val_stride=10,
                                   is_val_set=False,
                                   series_uid=EXAMPLE_UID,
                                   ratio=balance_ratio,
                                   augmentations=augmentation_dict)            
        # TODO: it errors on mac when choose many workers due to some pickle error. 
        # Solve it in order to speed up training.
        train_dl = DataLoader(train_ds, 
                              batch_size=CONFIG.training.batch_size,
                              shuffle=True,
                              #num_workers=CONFIG.training.num_workers,
                              pin_memory=self.pin_memory,
                              )
        return train_dl
        
    def _init_val_dl(self):
        if not self.dev:
            val_ds = Luna3DClassificationDataset(val_stride=10,
                                   is_val_set=True)
        else:
            EXAMPLE_UID = '1.3.6.1.4.1.14519.5.2.1.6279.6001.277445975068759205899107114231'
            val_ds = Luna3DClassificationDataset(val_stride=10,
                                   is_val_set=True,
                                   series_uid=EXAMPLE_UID)       
        val_dl = DataLoader(val_ds,
                            batch_size=CONFIG.training.batch_size,
                            shuffle=True,
                            #num_workers=CONFIG.training.num_workers,
                            pin_memory=self.pin_memory,
                            )
        return val_dl
    
    def _init_test_dl(self):
        ...

    def _log_metrics(self, epoch_ndx,  mode_str, metrics_t, start, classification_treshold=0.5):
        neg_label_mask = metrics_t[METRIX_LABEL_NDX] <= classification_treshold
        neg_pred_mask = metrics_t[METRICS_PRED_NDX] <= classification_treshold
        
        pos_label_mask = ~neg_label_mask
        pos_pred_mask =  ~neg_pred_mask
        
        neg_count = int(neg_label_mask.sum())
        pos_count = int(pos_label_mask.sum())
        
        true_neg_count = int((neg_label_mask & neg_pred_mask).sum())
        true_pos_count = int((pos_label_mask & pos_pred_mask).sum())
        
        false_pos_count = neg_count - true_neg_count
        false_neg_count = pos_count - true_pos_count
        
        if true_pos_count != 0:
            precision = true_pos_count / np.float32(true_pos_count + false_pos_count)
            recall = true_pos_count / np.float32(true_pos_count + false_neg_count)
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            precision, recall, f1_score = 0, 0, 0
        
        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDX, neg_label_mask].mean()
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_NDX, pos_label_mask].mean()
        metrics_dict['correct/all'] = (true_pos_count + true_neg_count) / np.float32(metrics_t.shape[1]) * 100
        metrics_dict['correct/neg'] = true_neg_count / neg_count * 100
        metrics_dict['correct/pos'] = true_pos_count / pos_count * 100
        
        metrics_dict['pr/precision'] = precision
        metrics_dict['pr/recall'] = recall
        metrics_dict['pr/f1'] = f1_score
        
        for key, value in metrics_dict.items():
            if mode_str == 'train':
                self.trn_writer.add_scalar(key, value, self.total_training_samples_count)
            elif mode_str == 'val':
                self.val_writer.add_scalar(key, value, self.total_training_samples_count)
                
        #print(f'EP:{epoch_ndx} {mode_str} results: {metrics_dict} finished in {round(time.time() - start, 3)}')

    def _init_tensorboard_writers(self):
        log_dir = os.path.join('.runs', self.log_prefix, self.time_str)
        self.trn_writer = SummaryWriter(log_dir=log_dir + '/trn')
        self.val_writer = SummaryWriter(log_dir=log_dir + '/val')

    def _compute_loss(self, batch_ndx, batch, metrics_g):
        input, labels, series_list, center_list = batch
        input = input.to(self.device)
        labels = labels.to(self.device)
        
        logits, probabilities = self.model(input)
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fn(logits, labels[:, 1])
        
        # log results into matrix
        start_ndx = batch_ndx * CONFIG.training.batch_size
        end_ndx = start_ndx + labels.size(0)
        metrics_g[METRIX_LABEL_NDX, start_ndx:end_ndx] = labels[:,1].detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = probabilities[:, 1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss.detach()
                
        return loss.mean(), metrics_g       
    
    def _do_training(self, epoch_ndx, dl):
        start = time.time()
        self.model.train()
        metrics = torch.zeros(
            METRICS_SIZE,
            len(dl.dataset),
            device=CONFIG.general.device
        )
        for batch_ndx, batch in enumerate(dl):
            self.optimizer.zero_grad()
            loss_var, metrics_g = self._compute_loss(batch_ndx, batch, metrics_g=metrics)
            loss_var.backward()
            self.optimizer.step()
        
        self._log_metrics(epoch_ndx=epoch_ndx, mode_str='train', metrics_t=metrics_g, start=start)
            
        self.total_training_samples_count += len(dl.dataset)
        
    def _do_validation(self, epoch_ndx, dl):
        with torch.no_grad():
            self.model.eval()
            metrics = torch.zeros(
                METRICS_SIZE,
                len(dl.dataset),
                device=CONFIG.general.device
            )
            for batch_ndx, batch in enumerate(dl):
                start = time.time()
                loss_var, metrics_g =  self._compute_loss(batch_ndx, batch, metrics_g=metrics)
                
            self._log_metrics(epoch_ndx=epoch_ndx, mode_str='val', metrics_t=metrics_g, start=start)


    def main(self):

        train_dl = self._init_train_dl()
        val_dl = self._init_val_dl()

        for epoch_ndx in range(1, CONFIG.training.epochs+1):
            self._do_training(epoch_ndx, train_dl)
            self._do_validation(epoch_ndx, val_dl)



if __name__ == '__main__':

    LunaTrainingApp(log_prefix='dev', dev=True).main()