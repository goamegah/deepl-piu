import torch.optim as optim

def get_optimizer(model, optimizer, learning_rate):
    if optimizer == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer == "radam":
        return optim.RAdam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Optimiseur inconnu.")
    
def get_scheduler(optimizer, scheduler):
    if scheduler == "none":
        return None
    elif scheduler == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    elif scheduler == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
    else:
        raise ValueError("Scheduler inconnu.")
