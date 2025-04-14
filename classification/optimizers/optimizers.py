import torch

def adam(model_parameters, lr):
    return torch.optim.Adam(model_parameters, lr=lr)
