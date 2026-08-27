import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm

CHECKPOINT_PATH = "checkpoint_last_epoch.pth"
BEST_MODEL_PATH = "best_deepfake_detector.pth"

def save_checkpoint(epoch, model, optimizer, best_auc, path=CHECKPOINT_PATH):
    torch.save({
        'epoch': epoch, 
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_auc': best_auc,
    }, path)
    print(f"  [Checkpoint] Saved checkpoint for epoch {epoch+1} -> {path}")

def load_checkpoint(model, optimizer, path=CHECKPOINT_PATH):
    if os.path.exists(path):
        print(f"  [Checkpoint] Found checkpoint at '{path}'. Resuming...")
        checkpoint = torch.load(path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_auc = checkpoint['best_auc']
        print(f"  [Checkpoint] Resuming from epoch {start_epoch + 1}, best AUC so far: {best_auc:.4f}")
        return start_epoch, best_auc
    elif os.path.exists(BEST_MODEL_PATH):
        print(f"  [Warm Start] No full checkpoint found, but found best model at '{BEST_MODEL_PATH}'.")
        print("  [Warm Start] Loading weights to continue from your previous best state...")
        weights = torch.load(BEST_MODEL_PATH, weights_only=False)
        # Handle if file is just state_dict or a dict with metadata
        if isinstance(weights, dict) and 'model_state_dict' in weights:
            model.load_state_dict(weights['model_state_dict'])
        else:
            model.load_state_dict(weights)
        print("  [Warm Start] Success! Training will start from Epoch 1 using these weights.")
        return 0, 0.0
    else:
        print("  [Checkpoint] No checkpoint or best model found. Starting fresh.")
        return 0, 0.0

def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda', lr=1e-4):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Only optimize parameters that require gradients (PEFT strategy)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # --- Resume from checkpoint if it exists ---
    start_epoch, best_auc = load_checkpoint(model, optimizer)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Train Loss: {epoch_loss:.4f}")

        # Validation Phase
        if val_loader:
            model.eval()
            all_preds = []
            all_labels = []
            val_loss = 0.0

            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc="Validation"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)

                    probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of Fake
                    all_preds.extend(probs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_loss = val_loss / len(val_loader.dataset)
            auc = roc_auc_score(all_labels, all_preds)
            preds_cls = [1 if p > 0.5 else 0 for p in all_preds]
            f1 = f1_score(all_labels, preds_cls)

            print(f"Val Loss: {val_loss:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")

            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"  [Best Model] New best AUC {auc:.4f}! Saved -> {BEST_MODEL_PATH}")

        # Save checkpoint after every epoch
        save_checkpoint(epoch, model, optimizer, best_auc)

    print(f"\nTraining complete. Best Val AUC: {best_auc:.4f}")
