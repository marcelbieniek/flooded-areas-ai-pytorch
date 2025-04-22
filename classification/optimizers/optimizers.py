import torch

def adam(model_parameters, lr):
    return torch.optim.Adam(model_parameters, lr=lr)

def sgd(model_parameters, lr):
    return torch.optim.SGD(model_parameters, lr=lr)
