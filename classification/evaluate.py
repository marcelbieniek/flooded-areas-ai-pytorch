import torch
from torch.utils.data import DataLoader
from utils.config_parser import Config
from utils.logger import TimeLogger, DataLogger

def evaluate_model(dataloader: DataLoader, config: Config, timer: TimeLogger, logger: DataLogger, device: str, verbose: bool):
    print("Validating...")
    model = config.model
    loss_fn = config.loss
    metrics = config.metrics

    running_loss = 0.0
    all_outputs = []
    all_targets = []

    log_name = f"{config.config_name}_val"
    time_log_name = f"{log_name}_time"
    timer.start(time_log_name)

    model.move_to_device(device)
    model.eval_mode()
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = prepare_data(X, y, device, config.task)
            # print(f"X: {X.size()}")
            # print(f"y: {y.size()}")

            outputs = model.forward(X)
            # print(f"outputs: {outputs.size()}")
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
        logger.log(f"{log_name}_{config.metrics_names[idx]}", result.item())
        print(f"{config.metrics_names[idx]}: {result.item()}")

    if device == 'cuda':
        torch.cuda.synchronize()
    timer.end(time_log_name)

    def prepare_data(X, y, device, task):
        X = X.to(device)

        if task == "classification":
            y = y.to(device).float().unsqueeze(1)

        if task == "segmentation":
            y = y.to(device).unsqueeze().long()

        return X, y
