import torch
from torch.utils.data import DataLoader
from utils.config_parser import Config
from utils.logger import TimeLogger, DataLogger

def test_model(dataloader: DataLoader, config: Config, timer: TimeLogger, logger: DataLogger, device: str):
    print("Testing...")
    model = config.model
    metrics = config.metrics

    all_outputs = []
    all_targets = []

    log_name = f"{model.name}_val"
    timer.start(log_name)

    model.move_to_device(device)
    model.eval_mode()
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            outputs = model.train(X)

            all_outputs.append(outputs.squeeze())
            all_targets.append(y)

        outputs_tensor = torch.cat(all_outputs)
        target_tensor = torch.cat(all_targets)

    for idx, metric in enumerate(metrics):
        metric = metric.to(device)
        result = metric(outputs_tensor, target_tensor)
        logger.log(f"{config.model.name}_{config.metrics_names[idx]}", result.item())
        print(f"{config.metrics_names[idx]}: {result.item()}")

    if device == 'cuda':
        torch.cuda.synchronize()
    timer.end(log_name)
    #         preds = torch.sigmoid(outputs) > 0.5
    #         correct += (preds.squeeze().long() == y).sum().item()
    #         total += y.size(0)
    # # print(f"Test error: \n Accuracy: {(100*correct):>0.1f}%, avg loss: {test_loss:>8f} \n")
    # print(f"Test error: \n Accuracy: {correct / total * 100:.2f}%")
