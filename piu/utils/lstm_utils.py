from sklearn.metrics import balanced_accuracy_score
import torch
import wandb

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args):
    print("🚀 Début de l'entraînement...")

    for epoch in range(args.num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []

        for X_seq, X_static, y in train_loader:
            X_seq, X_static, y = X_seq.to(device), X_static.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X_seq, X_static)

            # 🔹 Gestion des NaN
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
            correct += (predicted == y).sum().item()
            total += y.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        # Choix de la métrique
        if args.metric == "balanced_accuracy":
            train_acc = balanced_accuracy_score(all_labels, all_preds) * 100
        else:
            train_acc = 100 * correct / total

        avg_loss = total_loss / len(train_loader)

        if scheduler:
            scheduler.step()

        print(f"🔄 Epoch [{epoch+1}/{args.num_epochs}] - Loss: {avg_loss:.4f} - Train {args.metric}: {train_acc:.2f}%")
        wandb.log({"Train Loss": avg_loss, f"Train {args.metric}": train_acc})

        # 🔍 Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, args)
        print(f"📊 Validation - Loss: {val_loss:.4f} - {args.metric.capitalize()}: {val_acc:.2f}%")
        wandb.log({"Val Loss": val_loss, f"Val {args.metric}": val_acc})


def evaluate(model, val_loader, criterion, device, args):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_seq, X_static, y in val_loader:
            X_seq, X_static, y = X_seq.to(device), X_static.to(device), y.to(device)

            outputs = model(X_seq, X_static)
            outputs = torch.nan_to_num(outputs, nan=0.0)

            loss = criterion(outputs, y)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    if args.metric == "balanced_accuracy":
        val_acc = balanced_accuracy_score(all_labels, all_preds) * 100
    else:
        val_acc = 100 * correct / total

    return total_loss / len(val_loader), val_acc
