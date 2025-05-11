import torch
from torch.utils.data import DataLoader
from utils.config_parser import Config
from utils.logger import TimeLogger, DataLogger

def train_model(dataloader: DataLoader, config: Config, timer: TimeLogger, logger: DataLogger, device: str, verbose: bool):
    if verbose:
        print("Training...")
    model = config.model
    loss_fn = config.loss
    optimizer = config.optimizer
    metrics = config.metrics

    size = len(dataloader.dataset)
    current = 0
    running_loss = 0.0
    all_outputs = []
    all_targets = []

    log_name = f"{model.name}_train"
    timer.start(log_name)

    model.move_to_device(device)
    model.train_mode()
    for batch, (X, y) in enumerate(dataloader):
        X, y = prepare_data(X, y, device, config.task)

        optimizer.zero_grad()

        outputs = model.forward(X)
        loss = model.calculate_loss(loss_fn, outputs, y)

        # backpropagation
        loss.backward()
        optimizer.step()

        if len(outputs) > 1:
            all_outputs.append(outputs[0])
        else:
            all_outputs.append(outputs)
        all_targets.append(y)

        # if batch % 100 == 0:
        loss, current = loss.item(), current + len(X)
        running_loss += loss
        if verbose:
            print(f"batch: {batch}, loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    outputs_tensor = torch.cat(all_outputs)
    target_tensor = torch.cat(all_targets)
    logger.log(f"{log_name}_loss", running_loss/len(dataloader)) # log average epoch loss

    for idx, metric in enumerate(metrics):
        metric = metric.to(device)
        result = metric(outputs_tensor, target_tensor)
        logger.log(f"{log_name}_{config.metrics_names[idx]}", result.item())
        if verbose:
            print(f"{config.metrics_names[idx]}: {result.item()}")

    if device == 'cuda':
        torch.cuda.synchronize()
    timer.end(log_name)

def prepare_data(X, y, device, task):
    X = X.to(device)

    if task == "classification":
        y = y.to(device).float().unsqueeze(1)
    
    if task == "segmentation":
        y.to(device).squeeze().long()
    
    return X, y