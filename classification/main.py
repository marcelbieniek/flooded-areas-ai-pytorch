from utils.config_parser import load_config
import models
import losses
import optimizers
import metrics

from dataloader.dataloader import get_dataloaders
# from train import train_model

# config = load_config("config/inceptionnetv3_config.yaml")
# print(config)
# inception = getattr(models, config["model"])(config["num_classes"], config["pretrained"], config["aux_logits"])
# print(inception)
# loss_fn = getattr(losses, config["loss"])()
# print(loss_fn)
# optimizer = getattr(optimizers, config["optimizer"]["name"])(inception.parameters(), config["optimizer"]["lr"])
# print(optimizer)
# metrics_fns = [getattr(metrics, name)(**config["metrics"][name]) for name in config['metrics']]
# print(metrics_fns)

train_data, test_data = get_dataloaders()
print(train_data, test_data)
