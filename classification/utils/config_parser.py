import yaml
from classification import models, losses, optimizers, metrics

TRAIN_SEGMENT = "train"
VAL_SEGMENT = "val"
TEST_SEGMENT = "test"

class Config():
    def __init__(self, config_path: str):
        self.config = self.__load_config(config_path)
        self.config_name = config_path.replace("/", "_").replace(".yml", "").replace(".yaml", "")
        self.task = self.__get_task()
        self.model = self.__load_model()
        self.loss = self.__load_loss()
        self.optimizer = self.__load_optimizer()
        self.metrics = self.__load_metrics()
        self.metrics_names = self.__get_metrics_names()
        self.epochs = self.__get_epochs()
        self.batch_size = self.__get_batch_size()
        self.train_data_paths = self.__get_data_paths(TRAIN_SEGMENT)
        self.val_data_paths = self.__get_data_paths(VAL_SEGMENT)
        self.test_data_paths = self.__get_data_paths(TEST_SEGMENT)

    def __load_config(self, path):
        with open(path) as f:
            return yaml.safe_load(f)

    def __get_task(self):
        return self.config["task"]
    
    def __load_model(self):
        return getattr(models, self.config["model"]["name"])(**self.config["model"]["params"])
    
    def __load_loss(self):
        return getattr(losses, self.config["loss"])()
    
    def __load_optimizer(self):
        return getattr(optimizers, self.config["optimizer"]["name"])(self.model.parameters(), self.config["optimizer"]["lr"])

    def __load_metrics(self):
        return [getattr(metrics, name)(**self.config["metrics"][name]) for name in self.config['metrics']]
    
    def __get_metrics_names(self):
        return [name for name in self.config['metrics']]

    def __get_epochs(self):
        return self.config["train"]["epochs"]
    
    def __get_batch_size(self):
        return self.config["batch_size"]
    
    def __get_data_paths(self, segment):
        return (self.config[segment]["inputs"], self.config[segment]["targets"])