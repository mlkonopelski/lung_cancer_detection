from typing import Dict
import yaml

class General:
    def __init__(self, general_dict: Dict) -> None:
        self.device = general_dict['device']
        self.dev = general_dict['dev']
        

class Paths:
    def __init__(self, paths_dict: Dict) -> None:
        self.data = paths_dict['data']
        self.tensorboard = paths_dict['tensorboard']
        self.logs = paths_dict['logs']
        self.model_checkpoints = paths_dict['model_checkpoints']
        self.best_model = paths_dict['best_model']

class CLSTraining:
    def __init__(self, training_dict: Dict) -> None:
        self.epochs = training_dict['epochs']
        self.iter_per_epoch = training_dict['iter_per_epoch']
        self.early_stopping = training_dict['early_stopping']
        self.num_workers = training_dict['num_workers']
        self.batch_size = training_dict['batch_size']
        self.balanced = training_dict['balanced']
        self.augmentation = training_dict['augmentation']

class SEGTraining:
    def __init__(self, training_dict: Dict) -> None:
        self.epochs = training_dict['epochs']
        self.iter_per_epoch = training_dict['iter_per_epoch']
        self.early_stopping = training_dict['early_stopping']       
        self.num_workers = training_dict['num_workers']
        self.batch_size = training_dict['batch_size']
        self.augmentation = training_dict['augmentation']

    
class Config:
    def __init__(self, conf_yaml: Dict) -> None:
        self.general = General(conf_yaml['general'])
        self.paths = Paths(conf_yaml['paths'])
        self.cls_training = CLSTraining(conf_yaml['cls_training'])
        self.seg_training = SEGTraining(conf_yaml['seg_training'])
        
    # TODO: Implement a nice __str__ method

        
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)
    
CONFIG = Config(config)


if __name__ == '__main__':
    
    print(CONFIG.general.device)