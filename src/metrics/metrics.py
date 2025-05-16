from torchmetrics import Accuracy, Precision, Recall, MatthewsCorrCoef, CohenKappa, JaccardIndex

def accuracy(**kwargs):
    return Accuracy(**kwargs)

def precision(**kwargs):
    return Precision(**kwargs)

def recall(**kwargs):
    return Recall(**kwargs)

def mcc(**kwargs):
    """Matthews Correlation Coefficient"""
    return MatthewsCorrCoef(**kwargs)

def cohen_kappa(**kwargs):
    """Cohen's kappa score"""
    return CohenKappa(**kwargs)

def iou(**kwargs):
    """Intersection over Union (Jaccard index)"""
    return JaccardIndex(**kwargs)
