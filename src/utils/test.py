import torch
from torch.utils.data import DataLoader
from utils.Config import Config
from utils.logging import TimeLogger, DataLogger
from utils.utils import prepare_data

def test_model(dataloader: DataLoader, config: Config, timer: TimeLogger, logger: DataLogger, device: str, verbose: bool):
    if verbose:
        print("Testing...")
    model = config.model
    loss_fn = config.loss
    metrics = config.metrics

    running_loss = 0.0
    all_outputs = []
    all_targets = []

    log_name = f"{config.config_name}_test"
    time_log_name = f"{log_name}_time"
    timer.start(time_log_name)

    model.move_to_device(device)
    model.eval_mode()
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = prepare_data(X, y, device, config.task)

            outputs = model.forward(X)
            loss = model.calculate_loss(loss_fn, outputs, y)

            running_loss += loss.item()
            all_outputs.append(outputs)
            all_targets.append(y)

        outputs_tensor = torch.cat(all_outputs)
        target_tensor = torch.cat(all_targets)

    logger.log(f"{log_name}_loss", running_loss/len(dataloader)) # log average loss

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
