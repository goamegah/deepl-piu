import numpy as np
import torch
import wandb
from sklearn.metrics import balanced_accuracy_score

class EarlyStopping:
    """
    Arrête l'entraînement si la perte de validation ne s'améliore pas après un certain nombre d'epochs.
    """
    def __init__(self, patience=5, delta=0.0, verbose=False):
        """
        Args:
            patience (int): Nombre d'epochs sans amélioration à tolérer.
            delta (float): Amélioration minimale pour considérer qu'il y a progrès.
            verbose (bool): Si True, affiche les messages lors de l'amélioration.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Appelé à chaque epoch pour mettre à jour le suivi de la loss de validation.
        """
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Amélioration détectée : nouvelle meilleure loss = {self.best_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"Aucune amélioration ({self.counter}/{self.patience}).")
            if self.counter >= self.patience:
                self.early_stop = True

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, metric="accuracy", early_stopping_patience=5, early_stopping_delta=0.0):
    print("🚀 Début de l'entraînement...")
    
    # Instanciation de l'early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, delta=early_stopping_delta, verbose=True)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_y = []

        for X_seq, X_static, y in train_loader:
            X_seq, X_static, y = X_seq.to(device), X_static.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X_seq, X_static)
            outputs = torch.nan_to_num(outputs, nan=0.0)
            loss = criterion(outputs, y)

            if torch.isnan(loss):
                print("❌ NaN détecté dans la loss, batch ignoré")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_y.extend(y.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        
        # Calcul de la métrique d'entraînement
        if metric == "balanced_accuracy":
            train_acc = balanced_accuracy_score(all_y, all_preds) * 100
        else:
            correct = sum(1 for pred, true in zip(all_preds, all_y) if pred == true)
            train_acc = 100 * correct / len(all_y)

        print(f"🔄 Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.2f}%")
        wandb.log({"Train Loss": avg_loss, "Train Accuracy": train_acc})

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, metric=metric)
        print(f"📊 Validation - Loss: {val_loss:.4f} - Accuracy: {val_acc:.2f}%")
        wandb.log({"Val Loss": val_loss, "Val Accuracy": val_acc})

        # Vérification de l'early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"🛑 Early stopping activé après {epoch+1} epochs sans amélioration suffisante.")
            break

        if scheduler:
            scheduler.step()


def evaluate(model, val_loader, criterion, device, metric="accuracy"):
    model.eval()
    total_loss = 0
    all_preds = []
    all_y = []

    with torch.no_grad():
        for X_seq, X_static, y in val_loader:
            X_seq, X_static, y = X_seq.to(device), X_static.to(device), y.to(device)
            outputs = model(X_seq, X_static)
            outputs = torch.nan_to_num(outputs, nan=0.0)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_y.extend(y.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    if metric == "balanced_accuracy":
        score = balanced_accuracy_score(all_y, all_preds) * 100
    else:
        correct = sum(1 for pred, true in zip(all_preds, all_y) if pred == true)
        score = 100 * correct / len(all_y)

    return avg_loss, score
