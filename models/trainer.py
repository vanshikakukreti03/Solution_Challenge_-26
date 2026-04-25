"""
Training pipeline for FraudGCN with class-weighted loss and early stopping.
"""
import os
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm


def train_model(model, data, device, epochs=100, lr=0.01, weight_decay=5e-4, patience=15, save_path=None):
    data = data.to(device)
    model = model.to(device)

    # Compute class weights for imbalanced dataset
    train_labels = data.y[data.train_mask]
    counts = torch.bincount(train_labels[train_labels >= 0])
    weights = (1.0 / counts.float())
    weights = weights / weights.sum() * len(counts)
    weights = weights.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(1, epochs + 1), desc="Training GCN"):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=weights)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_loss = F.cross_entropy(out[data.test_mask], data.y[data.test_mask], weight=weights).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[INFO] Early stopping at epoch {epoch}")
                break

    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, weights_only=True))

    return model


def evaluate_model(model, data, device):
    data = data.to(device)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1)
        preds = out.argmax(dim=1)

    mask = data.test_mask
    y_true = data.y[mask].cpu().numpy()
    y_pred = preds[mask].cpu().numpy()
    y_prob = probs[mask, 1].cpu().numpy()

    return {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1_score': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'auc_roc': round(roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0, 4),
    }
