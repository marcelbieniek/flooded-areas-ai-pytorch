import torch
import matplotlib.pyplot as plt

def plot_predictions(model, data_loader, classes):
    model.move_to_device("cpu")
    model.eval_mode()

    with torch.inference_mode():
        images, labels = next(iter(data_loader))
        images, labels = images.to("cpu"), labels.to("cpu")
        outputs = model.forward(images)
        preds = torch.sigmoid(outputs) > 0.5
        preds = preds.squeeze().long()

    plt.figure(figsize=(12, 6))
    for i in range(6):  # Show 6 predictions
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i].permute(1, 2, 0))  # CHW to HWC
        plt.title(f"Pred: {classes[preds[i].item()]}\nTrue: {classes[labels[i].item()]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("plot")

def prepare_data(X, y, device, task):
    X = X.to(device)

    if task == "classification":
        y = y.to(device).float().unsqueeze(1)

    if task == "segmentation":
        y = y.to(device).squeeze().long()

    return X, y
