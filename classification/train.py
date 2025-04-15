from torch.utils.data import DataLoader

def train_model(dataloader: DataLoader, model, loss_fn, optimizer, device):
    print("Training...")
    size = len(dataloader.dataset)
    current = 0
    model.move_to_device(device)
    model.train_mode()
    # print(f"num batches: {len(dataloader)}")
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
        print(f"batch: {batch}, loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
