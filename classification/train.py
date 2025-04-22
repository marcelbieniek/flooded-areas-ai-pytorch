import torch
from torch.utils.data import DataLoader
from utils.config_parser import Config
from utils.logger import TimeLogger, DataLogger

def train_model(dataloader: DataLoader, config: Config, timer: TimeLogger, logger: DataLogger, device: str):
    print("Training...")
    model = config.model
    loss_fn = config.loss
    optimizer = config.optimizer

    size = len(dataloader.dataset)
    current = 0
    batch_losses = []

    timer_name = f"{model.name}_train"
    timer.start(timer_name)

    model.move_to_device(device)
    model.train_mode()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device).float().unsqueeze(1)
        # compute prediction error
        optimizer.zero_grad()

        outputs = model.train(X)
        loss = model.calculate_loss(loss_fn, outputs, y)

        # backpropagation
        loss.backward()
        optimizer.step()

        # if batch % 100 == 0:
        loss, current = loss.item(), current + len(X)
        batch_losses.append(loss)
        print(f"batch: {batch}, loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    logger.log(f"{model.name}_train_loss", sum(batch_losses)/len(batch_losses)) # log average epoch loss

    if device == 'cuda':
        torch.cuda.synchronize()
    timer.end(timer_name)
