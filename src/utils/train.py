import torch
from torch.utils.data import DataLoader
from utils.config_parser import Config
from utils.logging import TimeLogger, DataLogger

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

    log_name = f"{config.config_name}_train"
    time_log_name = f"{log_name}_time"
    timer.start(time_log_name)

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

        if not isinstance(outputs, (torch.Tensor, torch.LongTensor, torch.FloatTensor)):
            all_outputs.append(outputs[0])
        else:
            all_outputs.append(outputs)
        all_targets.append(y)

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

        if torch.numel(result) > 1:
            for i, el in enumerate(result):
                logger.log(f"{log_name}_{config.metrics_names[idx]}_{i}", el.item())
                if verbose:
                    print(f"{config.metrics_names[idx]}_{i}: {el.item()}")

            mean_result = result.mean().item()
            logger.log(f"{log_name}_{config.metrics_names[idx]}_mean", mean_result)
            if verbose:
                print(f"{config.metrics_names[idx]}_mean: {mean_result}")
        else:
            logger.log(f"{log_name}_{config.metrics_names[idx]}", result.item())
            if verbose:
                print(f"{config.metrics_names[idx]}: {result.item()}")

    if device == 'cuda':
        torch.cuda.synchronize()
    timer.end(time_log_name)

def prepare_data(X, y, device, task):
    X = X.to(device)

    if task == "classification":
        y = y.to(device).float().unsqueeze(1)
    
    if task == "segmentation":
        y = y.to(device).squeeze().long()
    
    return X, y
