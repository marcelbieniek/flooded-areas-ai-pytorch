import yaml
import models
import losses
import optimizers
import metrics

class Config():
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.model = self.load_model()
        self.loss = self.load_loss()
        self.optimizer = self.load_optimizer()
        self.metrics = self.load_metrics()

    def load_config(self, path):
        with open(path) as f:
            return yaml.safe_load(f)
    
    def load_model(self):
        return getattr(models, self.config["model"]["name"])(**self.config["model"]["params"])
    
    def load_loss(self):
        return getattr(losses, self.config["loss"])()
    
    def load_optimizer(self):
        return getattr(optimizers, self.config["optimizer"]["name"])(self.model.parameters(), self.config["optimizer"]["lr"])

    def load_metrics(self):
        return [getattr(metrics, name)(**self.config["metrics"][name]) for name in self.config['metrics']]
