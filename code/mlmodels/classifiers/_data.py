import datetime
import pickle
from typing import Tuple
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


class Dev3DClassifierDataset(Dataset):
    def __init__(self, path: str) -> None:
        super().__init__()

        with open(path, 'rb') as f:
            self.candidates = pickle.load(f)
            
    def __len__(self):
        return len(self.candidates)

    def __getitem__(self, ndx) -> Tuple[torch.Tensor]:
        ct_chunk, label, series_uid, pos_chunk = self.candidates[ndx]
        return  ct_chunk, label, series_uid, pos_chunk


class DevClassifierTrainingApp:
    
    def __init__(self, model, batch_size: int = 32) -> None:
        self.time_str = datetime.datetime.now().strftime('%Y%m%d%H%M')
        self.device = 'cpu'
        
        self.model = model
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=0.001, momentum=0.9)
        self.total_trn_samples_count = 0
        self.total_val_samples_count = 0
        
        self.trn_ds = Dev3DClassifierDataset('.data/.dev/classifier/train_100.pkl')
        self.trn_dl = DataLoader(self.trn_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
        self.val_ds = Dev3DClassifierDataset('.data/.dev/classifier/val_20.pkl')
        self.val_dl = DataLoader(self.trn_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    def _calculate_batch_metrics(self, metrics, probabilities, labels, batch_ndx):
        start_ndx = batch_ndx * self.batch_size
        end_ndx = start_ndx + labels.size(0)
        accuracy = (probabilities[:, 1].detach() >= 0.5) == labels.detach() 
        metrics[0, start_ndx:end_ndx] = accuracy
        return metrics
    
    def run(self, epochs: int,tensorboard: bool = True):

        if tensorboard:
            env_path = 'dev'
            model_path = type(self.model).__name__
            log_dir = os.path.join('.runs', env_path, model_path, self.time_str)
            self.trn_writer = SummaryWriter(log_dir=log_dir + '/train')
            self.val_writer = SummaryWriter(log_dir=log_dir + '/val')
        
        for epoch in range(1, epochs+1):
            #--------- Training ---------------
            self.model.train()
            trn_metrics = torch.zeros(1, len(self.trn_dl.dataset), device=self.device)
            for i, batch in enumerate(self.trn_dl):
                ct_chunk, labels, _series_uid, _center_irc = batch
                input = ct_chunk.unsqueeze(dim=1).to(self.device)
                labels = labels[:,1].to(self.device)

                self.optimizer.zero_grad()
                logits, probabilities = self.model(input)
                trn_loss = self.loss_fn(logits, labels)
                trn_loss.backward()
                self.optimizer.step()
                
                trn_metrics = self._calculate_batch_metrics(trn_metrics, probabilities, labels, i)
                
            trn_metrics_dict = {}
            trn_metrics_dict['accuracy'] = trn_metrics[0].mean().item()
            trn_metrics_dict['loss'] = trn_loss
            
            if tensorboard:
                for key, value in trn_metrics_dict.items():
                    self.trn_writer.add_scalar(key, value, self.total_trn_samples_count)
            self.total_trn_samples_count += len(self.trn_dl.dataset)

            #--------- Validation ---------------
            self.model.eval()
            val_metrics = torch.zeros(1, len(self.val_dl.dataset), device=self.device)
            for i, batch in enumerate(self.val_dl):
                ct_chunk, labels, _series_uid, _center_irc = batch
                input = ct_chunk.unsqueeze(dim=1).to(self.device)
                labels = labels[:,1].to(self.device)

                with torch.no_grad():
                    logits, probabilities = self.model(input)
                    val_loss = self.loss_fn(logits, labels)
                
                val_metrics = self._calculate_batch_metrics(val_metrics, probabilities, labels, i)
            
            val_metrics_dict = {}
            val_metrics_dict['accuracy'] = val_metrics[0].mean().item()
            val_metrics_dict['loss'] = val_loss
            
            for key, value in val_metrics_dict.items():
                self.val_writer.add_scalar(key, value, self.total_val_samples_count)
            self.total_val_samples_count += len(self.val_dl.dataset)
            

            print(
                f'epoch: {epoch} | i: {i}'
                f"\n train: loss: {trn_loss} | acc: {trn_metrics_dict['accuracy']}"
                f"\n val: loss: {val_loss} | acc: {val_metrics_dict['accuracy']}",
                '\n'+'-'*10
            )

    
if __name__ == '__main__':

    train_path = '.data/.dev/classifier/train_100.pkl'
    train_ds = Dev3DClassifierDataset(train_path)
    print(train_ds[0])
    
