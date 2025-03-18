import torch
import torch.optim as optim
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
import torch.nn as nn
import numpy as np
import random
from definitions import *

def set_seed(seed: int = 1):
    """ Assure la reproductibilité du code """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

set_seed()

class EarlyStopping:
    def __init__(self, patience=10, delta=0.001, path='best_model.pth'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_loss = np.inf
        self.counter = 0

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("*Early stopping activé !")
                return True 
        return False  # Continue training


def train_model(
        model: torch.nn.Module, 
        train_loader: torch.utils.data.DataLoader, 
        test_loader: torch.utils.data.DataLoader, 
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: torch.nn.Module,
        num_epochs: int = 300, 
        patience: int = 10,
        checkpoint_path: str = CHECKPOINT_PATH,
        use_balanced_accuracy: bool = False 
    ) -> None:
    model.train()
    early_stopping = EarlyStopping(patience=patience, path=f"{checkpoint_path}/best_model.pth")

    for epoch in range(num_epochs):
        train_loss = 0.0
        y_true = []
        y_pred = []

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

        train_loss /= len(train_loader)
        train_accuracy = balanced_accuracy_score(y_true, y_pred) if use_balanced_accuracy else accuracy_score(y_true, y_pred)
        train_precision = precision_score(y_true, y_pred, average='weighted')
        train_recall = recall_score(y_true, y_pred, average='weighted')
        train_f1 = f1_score(y_true, y_pred, average='weighted')

        test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, criterion, use_balanced_accuracy)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()

        wandb.log({
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1
        })

        print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f} - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")

        if early_stopping(test_loss, model):
            break


def evaluate_model(
        model: torch.nn.Module, 
        test_loader: torch.utils.data.DataLoader, 
        criterion: torch.nn.Module, 
        use_balanced_accuracy: bool = False
    ) -> tuple:
    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    test_loss /= len(test_loader)
    test_accuracy = balanced_accuracy_score(y_true, y_pred) if use_balanced_accuracy else accuracy_score(y_true, y_pred)
    test_precision = precision_score(y_true, y_pred, average='weighted')
    test_recall = recall_score(y_true, y_pred, average='weighted')
    test_f1 = f1_score(y_true, y_pred, average='weighted')

    return test_loss, test_accuracy, test_precision, test_recall, test_f1









































# import torch
# import torch.optim as optim
# import wandb
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
# import torch.nn as nn
# import numpy as np
# from definitions import *

# class EarlyStopping:

#     def __init__(self, patience=10, delta=0.001, path='best_model.pth'):
#         self.patience = patience
#         self.delta = delta
#         self.path = path
#         self.best_loss = np.inf
#         self.counter = 0

#     def __call__(self, val_loss, model):
#         if val_loss < self.best_loss - self.delta:
#             self.best_loss = val_loss
#             self.counter = 0
#             torch.save(model.state_dict(), self.path)
#         else:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 print("*Early stopping activé !")
#                 return True 
#         return False  # Continue training

# def train_model(
#         model: torch.nn.Module, 
#         train_loader: torch.utils.data.DataLoader, 
#         test_loader: torch.utils.data.DataLoader, 
#         optimizer: torch.optim.Optimizer,
#         scheduler: torch.optim.lr_scheduler._LRScheduler,
#         criterion: torch.nn.Module,
#         num_epochs: int=300, 
#         patience: int=10,
#         checkpoint_path: str=CHECKPOINT_PATH,
#         imbalanced: bool=False 
#     ) -> None:
#     model.train()
#     early_stopping = EarlyStopping(patience=patience, path=f"{checkpoint_path}/best_model.pth")

#     for epoch in range(num_epochs):
#         train_loss = 0.0
#         y_true = []
#         y_pred = []

#         for inputs, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()

#             _, predicted = torch.max(outputs, 1)
#             y_true.extend(labels.numpy())
#             y_pred.extend(predicted.numpy())

#         train_loss /= len(train_loader)
#         train_accuracy = accuracy_score(y_true, y_pred) if not imbalanced else balanced_accuracy_score(y_true, y_pred)
#         train_precision = precision_score(y_true, y_pred, average='weighted')
#         train_recall = recall_score(y_true, y_pred, average='weighted')
#         train_f1 = f1_score(y_true, y_pred, average='weighted')

#         test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, criterion, imbalanced=imbalanced)

#         if scheduler is not None:
#             scheduler.step(test_loss)

#         wandb.log({
#             'train_loss': train_loss,
#             'train_accuracy': train_accuracy,
#             'train_precision': train_precision,
#             'train_recall': train_recall,
#             'train_f1': train_f1,
#             'test_loss': test_loss,
#             'test_accuracy': test_accuracy,
#             'test_precision': test_precision,
#             'test_recall': test_recall,
#             'test_f1': test_f1
#         })

#         print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f} - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")

#         if early_stopping(test_loss, model):
#             break
    


# def evaluate_model(
#         model: torch.nn.Module, 
#         test_loader: torch.utils.data.DataLoader, 
#         criterion: torch.nn.Module, 
#         imbalanced: bool=False
#     ) -> tuple:
#     model.eval()
#     test_loss = 0.0
#     y_true = []
#     y_pred = []

#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             test_loss += loss.item()

#             _, predicted = torch.max(outputs, 1)
#             y_true.extend(labels.numpy())
#             y_pred.extend(predicted.numpy())

#     test_loss /= len(test_loader)
#     test_accuracy = accuracy_score(y_true, y_pred) if not imbalanced else balanced_accuracy_score(y_true, y_pred)
#     test_precision = precision_score(y_true, y_pred, average='weighted')
#     test_recall = recall_score(y_true, y_pred, average='weighted')
#     test_f1 = f1_score(y_true, y_pred, average='weighted')

#     return test_loss, test_accuracy, test_precision, test_recall, test_f1