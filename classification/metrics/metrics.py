from torchmetrics import Accuracy

def accuracy(**kwargs):
    return Accuracy(**kwargs)

def recall(task, num_classes):
    pass
