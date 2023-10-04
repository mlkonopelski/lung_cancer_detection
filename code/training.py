import argparse
import datetime
import os
import sys
import time
from timeit import timeit
from abc import ABC, abstractmethod
from typing import Dict, List
from pathlib import Path

import scipy.ndimage as ndimage

import numpy as np
import torch
import torch.nn as nn
from data import CandidateInfo, Ct, CtConversion, Luna3DClassificationDataset, TrainLuna2DSegmentationDataset, BaseLuna2DSegmentationDataset, SegmentationAugmentation, get_ct
from mlmodels.classifiers import base_cnn
from mlmodels.segmentation import unet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.config import CONFIG


EXAMPLE_UID = '1.3.6.1.4.1.14519.5.2.1.6279.6001.511347030803753100045216493273'

# BASE MATRIX
METRICS_ACCURACY=0

# CLASSIFICATION MATRIX
CLASSIFICATION_METRICS_LABEL_NDX = 0
CLASSIFICATION_METRICS_PRED_NDX = 1
CLASSIFICATION_METRICS_LOSS_NDX = 2

# SEGMENTATION MATRIX
METRICS_LOSS_NDX = 0
METRICS_TP_NDX = 1
METRICS_FN_NDX = 2
METRICS_FP_NDX = 3

METRICS_SIZE = 10


class BaseTrainingApp(ABC):
    def __init__(self, Model: nn.Module, sys_argv=None) -> None:
        # TODO: Write all Docstrings in TrainingApp
        
        self.dev = CONFIG.general.dev
        self.metrics_size = 1
        self.classification_threshold = 0.5
        self.best_score = 0
        self.validation_cadence = 1
        self.scoring_metric = 'pr/recall'

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
        self.time_str = datetime.datetime.now().strftime('%Y%m%d%H%M')
        
        self.total_training_samples_count = 0
        
        self.model = self._init_model(Model)
        self.augmentation_model = None
        self.optimizer = self._init_optimizer()
        self._init_tensorboard_writers()
        
        
    def _init_model(self, Model):
        
        """Helper method to instantiate ML model. 
        In future (when more models will be developed) 
        the selection will be based on config.yml or cmd line arguments

        Returns:
            _type_: instance of nn.Module model
        """
        model = Model()
        model = model.to(self.device)
        return model 
        
    @abstractmethod    
    def _init_optimizer(self):
        """Helper metod to instantiate optimizer. 
        As there are few possible popular optimizers this method will be
        based on config.yml or cmd line arguments. Popular optimizers:
        1. torch.optim.SGD(params, lr=0.001, momentum=0.9)
        1. torch.optim.Adam(params)

        Returns:
            _type_: instance of class from torch.optim module
        """
        ...

    @abstractmethod
    def loss_fn(self):
        ...

    @abstractmethod
    def _init_dl(self, mode: str) -> DataLoader:
        ...
    
    @abstractmethod
    def _log_images(self, epoch_ix: int, mode: str, dl: DataLoader) -> None:
        ...
                
    def _init_tensorboard_writers(self):
        env_path = 'prod' if not self.dev else 'dev'
        model_path = type(self.model).__name__
        log_dir = os.path.join(CONFIG.paths.tensorboard, env_path, model_path, self.time_str)
        self.train_writer = SummaryWriter(log_dir=log_dir + '/train')
        self.val_writer = SummaryWriter(log_dir=log_dir + '/val')

    def _compute_loss(self, batch_ndx, batch, metrics):
        input, labels, _series_list, _center_list = batch
        input = input.to(self.device)
        labels = labels.to(self.device)
        
        if self.model.training and self.augmentation_model:
            input, labels = self.augmentation_model(input, labels)

        logits, probabilities = self.model(input)
        loss = self.loss_fn(logits, labels)
        
        metrics = self._calculate_batch_metrics(metrics, probabilities, labels, loss, batch_ndx)
                
        return loss.mean(), metrics

    def _calculate_batch_metrics(self, metrics, probabilities, labels, loss, batch_ndx):
        start_ndx = batch_ndx * CONFIG.cls_training.batch_size
        end_ndx = start_ndx + labels.size(0)
        accuracy = probabilities[:, 1].detach() > self.classification_threshold == labels[:,1].detach() 
        metrics[METRICS_ACCURACY, start_ndx:end_ndx] = accuracy
        return metrics
    
    def _calculate_epoch_metrics(self, epoch_ndx, mode_str, metrics_t, start):
        metrics_dict = {}
        metrics_dict['accuracy'] = metrics_t[METRICS_ACCURACY].mean()
        print(f'EP:{epoch_ndx} {mode_str} results: {metrics_dict} finished in {round(time.time() - start, 3)}')
        return metrics_dict
        
    def _write_epoch_metrics(self, mode_str: str, metrics_dict: Dict):
            
        for key, value in metrics_dict.items():
            if mode_str == 'train':
                self.train_writer.add_scalar(key, value, self.total_training_samples_count)
            elif mode_str == 'val':
                self.val_writer.add_scalar(key, value, self.total_training_samples_count)

    def _write_epoch_images(self, mode: str, image: torch.Tensor, tag: str):
        writer = getattr(self, mode + '_writer')
        writer.add_image(tag=f'{mode}/{tag}', 
                         img_tensor=image, 
                         global_step=self.total_training_samples_count,
                         dataformats='HWC')

    def _do_training(self, epoch_ix, dl):
        start = time.time()
        self.model.train()
        epoch_raw_metrics = torch.zeros(
            self.metrics_size,
            len(dl.dataset),
            device=CONFIG.general.device
        )
        for batch_ix, batch in enumerate(dl):
            self.optimizer.zero_grad()
            loss_var, epoch_raw_metrics = self._compute_loss(batch_ix, batch, metrics=epoch_raw_metrics)
            loss_var.backward()
            self.optimizer.step()
        
        epoch_metrics = self._calculate_epoch_metrics(epoch_ndx=epoch_ix, mode_str='train', metrics_t=epoch_raw_metrics, start=start)
        self._write_epoch_metrics(mode_str='train', metrics_dict=epoch_metrics)
        self.total_training_samples_count += len(dl.dataset)

    def _do_validation(self, epoch_ix, dl):
        with torch.no_grad():
            self.model.eval()
            epoch_raw_metrics = torch.zeros(
                self.metrics_size,
                len(dl.dataset),
                device=CONFIG.general.device
            )
            for batch_ix, batch in enumerate(dl):
                start = time.time()
                _loss_var, epoch_raw_metrics =  self._compute_loss(batch_ix, batch, metrics=epoch_raw_metrics)
                
            epoch_metrics = self._calculate_epoch_metrics(epoch_ndx=epoch_ix, mode_str='val', metrics_t=epoch_raw_metrics, start=start)
            self._write_epoch_metrics(mode_str='val', metrics_dict=epoch_metrics)
            
            self.epoch_score = epoch_metrics[self.scoring_metric]
          

    def save_model(self, epoch_ix: int) -> None:

        state = {
            'sys_argv': sys.argv,
            'time': self.time_str,
            'model_state': self.model.state_dict(),
            'model_name': type(self.model).__name__,
            'optimizer_state' : self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ix,
            'total_training_samples_count': self.total_training_samples_count,
            'augmentations': str(self.augmentation_model)
        }

        model_path = Path(os.path.join(CONFIG.paths.best_model, f"{'dev' if self.dev else 'prod'}"))
        model_path.mkdir(parents=True, exist_ok=True)
        model_name = Path(f"{state['model_name']}_{state['time']}.best.state")

        torch.save(state, f=model_path / model_name)
        print(f"Saved model params to {model_path}/{model_name}")


    def main(self):

        train_dl = self._init_dl(mode='train')
        val_dl = self._init_dl(mode='val')

        for epoch_ix in range(1, CONFIG.cls_training.epochs+1):
            self._do_training(epoch_ix, train_dl)
            self._do_validation(epoch_ix, val_dl)
            if self.epoch_score > self.best_score:
                self.save_model(epoch_ix)
            if epoch_ix == 1 or epoch_ix % self.validation_cadence == 0:
                self._log_images(epoch_ix=epoch_ix, mode='train', dl=train_dl)
                self._log_images(epoch_ix=epoch_ix, mode='val', dl=val_dl)

class ClassificationTrainingApp(BaseTrainingApp):
    def __init__(self, Model: nn.Module, sys_argv=None) -> None:
        super().__init__(Model, sys_argv)
        self.metrics_size = 3
        self.scoring_metric = 'pr/f1'
    
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
    
    def loss_fn(self, logits, labels):
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        return loss_fn(logits, labels[:, 1])

    def _init_dl(self, mode: str) -> DataLoader:

        assert mode in ['train', 'val']

        balance_ratio = 1 if CONFIG.cls_training.balanced else None
        augmentation_dict = CONFIG.cls_training.classification_augmentation if CONFIG.cls_training.classification_augmentation else {}
        
        # This part is used only for development process to speed up the testing and validation of each part. 
        # It might be removed in future when the repositoty will be project will be ready. 
        if self.dev: 
            if mode == 'train':
                ds = Luna3DClassificationDataset(val_stride=10,
                                                 is_val_set=False,
                                                 ratio=balance_ratio,
                                                 augmentations=augmentation_dict,
                                                 series_uid=EXAMPLE_UID)
            elif mode == 'val':
                ds = Luna3DClassificationDataset(val_stride=10,
                                                 is_val_set=True,
                                                 series_uid=EXAMPLE_UID) 
                
            dl = DataLoader(ds, batch_size=CONFIG.cls_training.batch_size, shuffle=True)
            return dl
        
        # Training sample should be balanced and added augmnetations
        if mode == 'train':
            ds = Luna3DClassificationDataset(val_stride=10,
                                             is_val_set=False,
                                             ratio=balance_ratio,
                                             augmentations=augmentation_dict)
        elif mode == 'val':
            ds = Luna3DClassificationDataset(val_stride=10, is_val_set=True)
                      
        dl = DataLoader(ds,
                        batch_size=CONFIG.cls_training.batch_size,
                        shuffle=True,
                        #num_workers=CONFIG.training.num_workers,
                        pin_memory=self.pin_memory,
                        )
        return dl

    def _calculate_batch_metrics(self, metrics, probabilities, labels, loss, batch_ndx):

        # Metrics in Bbse class are calculated for Accuracy/Recall/Precision as the most popular choices. 
        # In special cases it needs to be reimplemented together with self._write_epoch_metrics

        # log results into matrix
        start_ndx = batch_ndx * CONFIG.cls_training.batch_size
        end_ndx = start_ndx + labels.size(0)
        metrics[CLASSIFICATION_METRICS_LABEL_NDX, start_ndx:end_ndx] = labels[:,1].detach()
        metrics[CLASSIFICATION_METRICS_PRED_NDX, start_ndx:end_ndx] = probabilities[:, 1].detach()
        metrics[CLASSIFICATION_METRICS_LOSS_NDX, start_ndx:end_ndx] = loss.detach()
        return metrics

    def _calculate_epoch_metrics(self, epoch_ndx, mode_str, metrics_t, start):
        neg_label_mask = metrics_t[CLASSIFICATION_METRICS_LABEL_NDX] <= self.classification_threshold
        neg_pred_mask = metrics_t[CLASSIFICATION_METRICS_PRED_NDX] <= self.classification_threshold
        
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
        metrics_dict['loss/all'] = metrics_t[CLASSIFICATION_METRICS_LOSS_NDX].mean().item()
        metrics_dict['loss/neg'] = metrics_t[CLASSIFICATION_METRICS_LOSS_NDX, neg_label_mask].mean().item()
        metrics_dict['loss/pos'] = metrics_t[CLASSIFICATION_METRICS_LOSS_NDX, pos_label_mask].mean().item()
        metrics_dict['correct/all'] = (true_pos_count + true_neg_count) / np.float32(metrics_t.shape[1]) * 100
        metrics_dict['correct/neg'] = true_neg_count / neg_count * 100
        metrics_dict['correct/pos'] = true_pos_count / pos_count * 100
        
        metrics_dict['pr/precision'] = precision
        metrics_dict['pr/recall'] = recall
        metrics_dict['pr/f1'] = f1_score
        
        print(f'EP:{epoch_ndx} {mode_str} results: {metrics_dict} finished in {round(time.time() - start, 3)}')
        return metrics_dict
                
    def _log_images(self, epoch_ix: int, mode: str, dl: DataLoader):
        ...
  
class SegmentationTrainingApp(BaseTrainingApp):
    def __init__(self, Model: nn.Module, sys_argv=None) -> None:
        super().__init__(Model, sys_argv)
        self._init_augmentations_model()
        self.metrics_size = 4
        self.scoring_metric = 'pr/f1_score'
    
    def _init_optimizer(self):
        """Helper metod to instantiate optimizer. 
        As there are few possible popular optimizers this method will be
        based on config.yml or cmd line arguments

        Returns:
            _type_: instance of class from torch.optim module
        """
        optimizer = torch.optim.Adam(self.model.parameters())
        return optimizer
    
    def _init_augmentations_model(self):
        augmentation_dict = CONFIG.cls_training.segmentation_augmentation if CONFIG.cls_training.segmentation_augmentation else {}
        if not self.dev:
            self.augmentation_model = SegmentationAugmentation(**CONFIG.cls_training.segmentation_augmentation)


    def loss_fn(self, predictions, labels, epsilon: int = 1):
        '''
        Loss function: Dice Loss. 
        Advantage of using Dice loss over a per-pixel cross-entropy loss is that 
        Dice handles the case where only a small portion of the overall image is flagged as positive. 
        
        Calculation: 
        DiceLoss =  \frac{ 2 * TruePositive}{True + Positive}
        It is twice the joint area (true positives, striped) divided by the sum of the entire predicted area and the entire ground-truth marked area (the overlap being counted twice).
        '''
        dice_label = labels.sum(dim=[1, 2, 3])
        dice_predictions = predictions.sum(dim=[1, 2, 3])
        dice_correct = (labels * predictions).sum(dim=[1, 2, 3])
        dice_ratio = (2 * dice_correct * epsilon) / (dice_label + dice_predictions + epsilon)
        return 1 - dice_ratio

    def _calculate_batch_metrics(self, metrics, predictions, labels, loss, batch_ndx):
        start_ndx = batch_ndx * CONFIG.cls_training.batch_size
        end_ndx = start_ndx + labels.size(0)

        with torch.no_grad():
            predictions_bool = (predictions[:, 0:1]
                                > self.classification_threshold).to(torch.float32)

            true_positive = (     predictions_bool *  labels).sum(dim=[1,2,3])
            false_negative = ((1 - predictions_bool) *  labels).sum(dim=[1,2,3])
            false_positive = (     predictions_bool * (~labels)).sum(dim=[1,2,3])

            metrics[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss
            metrics[METRICS_TP_NDX, start_ndx:end_ndx] = true_positive
            metrics[METRICS_FN_NDX, start_ndx:end_ndx] = false_negative
            metrics[METRICS_FP_NDX, start_ndx:end_ndx] = false_positive

        return metrics

    def _calculate_epoch_metrics(self, epoch_ndx, mode_str, metrics_t, start):
        
        metrics_a = metrics_t.cpu().numpy()
        sum_a = metrics_a.sum(axis=1)
        assert np.isfinite(metrics_a).all()

        allLabel_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_NDX].mean()

        metrics_dict['percent_all/tp'] = \
            sum_a[METRICS_TP_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fn'] = \
            sum_a[METRICS_FN_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fp'] = \
            sum_a[METRICS_FP_NDX] / (allLabel_count or 1) * 100


        precision = metrics_dict['pr/precision'] = sum_a[METRICS_TP_NDX] \
            / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FP_NDX]) or 1)
        recall    = metrics_dict['pr/recall']    = sum_a[METRICS_TP_NDX] \
            / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]) or 1)

        metrics_dict['pr/f1_score'] = 2 * (precision * recall) \
            / ((precision + recall) or 1)
        
        print(f'EP:{epoch_ndx} {mode_str} results: {metrics_dict} finished in {round(time.time() - start, 3)}')
        return metrics_dict

    def _compute_loss(self, batch_ndx, batch, metrics):
        input, labels, _series_list, _slice_list = batch
        input = input.to(self.device)
        labels = labels.to(self.device)
        
        if self.model.training and self.augmentation_model:
            input, labels = self.augmentation_model(input, labels)

        predictions = self.model(input)

        all_loss = self.loss_fn(predictions, labels)
        positive_loss = self.loss_fn(predictions * labels, labels)
        
        metrics = self._calculate_batch_metrics(metrics, predictions, labels, all_loss, batch_ndx)
        weighted_loss = all_loss.mean() + positive_loss.mean() * 8   
                
        return weighted_loss, metrics
    

    def _init_dl(self, mode: str):
        """_summary_

        Returns:
            _type_: torch DataLoader with 
        """
        assert mode in ['train', 'val']
        
        # This part is used only for development process to speed up the testing and validation of each part. 
        # It might be removed in future when the repositoty will be project will be ready. 
        if self.dev: 
            if mode == 'train':
                ds = TrainLuna2DSegmentationDataset(val_stride=1,
                                                    is_val_set=False,
                                                    series_uid=EXAMPLE_UID)
            elif mode == 'val':
                ds = BaseLuna2DSegmentationDataset(val_stride=1, 
                                                   is_val_set=True,
                                                    series_uid=EXAMPLE_UID)
                
            dl = DataLoader(ds,
                            batch_size=CONFIG.cls_training.batch_size,
                            shuffle=True
                            )
            return dl
        
        # FIXME: Check if series remove from train set are removed from validation since we use different classes. 
        if mode == 'train':
            ds = TrainLuna2DSegmentationDataset(val_stride=10, is_val_set=False)
        elif mode == 'val':
            ds = BaseLuna2DSegmentationDataset(val_stride=10, is_val_set=True)
                      
        dl = DataLoader(ds,
                        batch_size=CONFIG.cls_training.batch_size,
                        shuffle=True,
                        #num_workers=CONFIG.training.num_workers,
                        pin_memory=self.pin_memory,
                        )
        return dl

    def _log_images(self, epoch_ix: int, mode: str, dl: DataLoader) -> None:
        self.model.eval()
        images = sorted(dl.dataset.series)[:12] if not self.dev else [EXAMPLE_UID]
        for series_ix, series_uid in enumerate(images):
            ct = get_ct(series_uid)
            range_ = range(6) if not self.dev else range(1)
            for slice_ix in range_:
                ct_ix = slice_ix * (ct.hu.shape[0] - 1) // 5 if not self.dev else 68 # For selected EXAMPLE_UID 68 is the best visible index
                ct_t, label_t, _series_uid, slice_ix = dl.dataset.get_full_slice(series_uid, slice_ndx=ct_ix)
               
                input = ct_t.to(self.device).unsqueeze(0)
                labels = label_t.to(self.device).unsqueeze(0)

                predictions = self.model(input)[0]
                predictions = predictions.to('cpu').detach().numpy()[0] > 0.5
                labels = labels.cpu().numpy()[0][0] > 0.5
               
                ct_t[:-1,:,:] /= 2000
                ct_t[:-1,:,:] += 0.5
                
                ct_slice = ct_t[dl.dataset.context_slice_count].numpy()
                img = np.zeros((512, 512, 3), dtype=np.float32)
                img[:,:,:] = ct_slice.reshape((512, 512, 1))
                img[:,:,0] += predictions & (1 - labels)
                img[:,:,0] += (1 - predictions) & labels
                img[:,:,1] += ((1 - predictions) & labels) * 0.5
                img[:,:,1] += predictions & labels
                img *= 0.5
                img = img.clip(0, 1, img)
                
                self._write_epoch_images(mode=mode, image=img, tag=f'{series_ix}_prediction_{slice_ix}')

if __name__ == '__main__':

    ClassificationTrainingApp(Model=base_cnn.CNN).main()
    SegmentationTrainingApp(Model=unet.UNETLuna).main()