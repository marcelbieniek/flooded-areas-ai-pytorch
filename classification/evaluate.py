import torch
from torch.utils.data import DataLoader

def test_model(dataloader: DataLoader, model, metrics, device):
    print("Testing...")
    # size = len(dataloader.dataset)
    # print(f"size: {size}")
    # num_batches = len(dataloader)
    # print(f"num batches: {num_batches}")
    model.move_to_device(device)
    model.eval_mode()
    # correct = 0
    # total = 0

    all_outputs = []
    all_targets = []

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            outputs = model.train(X)

            all_outputs.append(outputs.squeeze())
            all_targets.append(y)

        outputs_tensor = torch.cat(all_outputs)
        target_tensor = torch.cat(all_targets)

    for metric in metrics:
        metric = metric.to(device)
        result = metric(outputs_tensor, target_tensor)
        print(f"{metric}: {result}")
    #         preds = torch.sigmoid(outputs) > 0.5
    #         correct += (preds.squeeze().long() == y).sum().item()
    #         total += y.size(0)
    # # print(f"Test error: \n Accuracy: {(100*correct):>0.1f}%, avg loss: {test_loss:>8f} \n")
    # print(f"Test error: \n Accuracy: {correct / total * 100:.2f}%")
