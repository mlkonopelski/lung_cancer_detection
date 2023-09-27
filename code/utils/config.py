from typing import Dict
import yaml

class General:
    def __init__(self, general_dict: Dict) -> None:
        self.device = general_dict['device']
        
        
class Training:
    def __init__(self, training_dict: Dict) -> None:
        self.epochs = training_dict['epochs']
        self.num_workers = training_dict['num_workers']
        self.batch_size = training_dict['batch_size']
        self.balanced = training_dict['balanced']
        self.classification_augmentation = training_dict['classification_augmentation']
        self.segmentation_augmentation = training_dict['segmentation_augmentation']
        
class Config:
    def __init__(self, conf_yaml: Dict) -> None:
        self.general = General(conf_yaml['general'])
        self.training = Training(conf_yaml['training'])
        
    # TODO: Implement a nice __str__ method

        
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)
    
CONFIG = Config(config)


if __name__ == '__main__':
    
    print(CONFIG.general.device)