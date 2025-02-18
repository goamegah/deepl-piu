import torch
import torch.optim as optim
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import numpy as np
from definitions import *

class EarlyStopping:
    """Arrête l'entraînement si la perte de validation ne s'améliore pas après un certain nombre d'epochs."""
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
            torch.save(model.state_dict(), self.path)  # Sauvegarde le meilleur modèle
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("⏳ Early stopping activé !")
                return True  # Stop training
        return False  # Continue training

def train_model(
        model: torch.nn.Module, 
        train_loader: torch.utils.data.DataLoader, 
        test_loader: torch.utils.data.DataLoader, 
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        num_epochs: int=300, 
        patience: int=10,
        checkpoint_path: str=CHECKPOINT_PATH,
) -> None:
    """Entraîne le modèle avec early stopping et régularisation L2."""
    model.train()
    early_stopping = EarlyStopping(patience=patience, path=f'{checkpoint_path}/best_model.pth')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"total: {total}")
        train_accuracy = correct / total if total > 0 else 0.0
        train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        # Ajout de la validation après chaque époque
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(model, test_loader, criterion)

        # Logger sur WandB
        wandb.log({
            'train/loss': train_loss, 
            'train/accuracy': train_accuracy,
            'eval/loss': val_loss, 
            'eval/accuracy': val_accuracy,
            'eval/precision': val_precision, 
            'eval/recall': val_recall, 
            'eval/f1_score': val_f1,
            'epoch': epoch + 1
        })

        print(f'🔄 Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Eval Loss: {val_loss:.4f} | Eval Acc: {val_accuracy:.4f}')

        # Vérifier si on doit stopper l'entraînement
        if early_stopping(val_loss, model):
            break  # Stop training si early stopping activé

    # Charger le meilleur modèle sauvegardé
    model.load_state_dict(torch.load(f'{checkpoint_path}/best_model.pth'))
    print("✅ Meilleur modèle chargé après Early Stopping.")

def evaluate_model(model, test_loader, criterion):
    """Évalue le modèle sur le test set et retourne les métriques."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Gestion des cas où test_loader est vide
    if len(all_preds) == 0 or len(all_labels) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Évite les warnings de division par zéro avec `zero_division=1`
    accuracy = accuracy_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)

    return total_loss / len(test_loader) if len(test_loader) > 0 else 0.0, accuracy, precision, recall, f1
