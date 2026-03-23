# BC_MLP_Custom_Model_Trainer.py
#
# Binary Classification using a Custom MLP Model
# 
# This script allows you to define your own model architecture and train/evaluate it
# using the exact same procedure as train_and_save_best_model() from IM_Binary_Quality_Recognition.py

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, auc, roc_curve, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import matplotlib.pyplot as plt
import time
import math
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project root (works from any subfolder)

# Import necessary components from main script
import sys
sys.path.insert(0, str(BASE_DIR / "src"))
from BC_MLP_IM import (
    BinaryFocalLoss, 
    EarlyStopping, 
    load_dataset,
    find_best_threshold
)


# ============================================================================
# DEFINE YOUR CUSTOM MODEL HERE
# ============================================================================

class CustomModel(nn.Module):
    """
    Define your custom model architecture here.
    
    Example 1: Simple MLP with 2 hidden layers
    def __init__(self, input_size=16, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    Example 2: Model with BatchNorm
    def __init__(self, input_size=16, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    Example 3: Deeper network
    def __init__(self, input_size=16, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    """
    
    def __init__(self, input_size=16, dropout=0.2):
        super().__init__()
        
        # ====== CUSTOMIZE YOUR ARCHITECTURE HERE ======
        self.net = nn.Sequential(
            nn.Linear(input_size, 67),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(67, 25),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(25, 1)
        )
        # =============================================
    
    def forward(self, x):
        return self.net(x)


# ============================================================================
# HYPERPARAMETERS - CUSTOMIZE THESE
# ============================================================================

CUSTOM_HYPERPARAMETERS = {
    'dropout': 0.2,
    'batch_size': 32,
    'lr': 0.00334,            # Learning rate
    'weight_decay': 3.39e-6,  # L2 regularization
    'alpha': 0.4197,          # Focal loss alpha
    'gamma': 0.9338,          # Focal loss gamma
    'patience': 20,           # Early stopping patience
    'max_epochs': 200,        # Maximum training epochs
}


# ============================================================================
# TRAINING FUNCTIONS (Same as in main script)
# ============================================================================

def init_weights_with_prior(model: nn.Module, pos_prior: float | None = None, method: str = "kaiming"):
    """Initialize weights with proper initialization."""
    # Find last Linear layer (output)
    last_linear = None
    if hasattr(model, "net"):
        for m in reversed(model.net):
            if isinstance(m, nn.Linear):
                last_linear = m
                break

    def _init(m):
        if isinstance(m, nn.Linear):
            if m is last_linear:
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    if pos_prior is not None:
                        p = max(min(float(pos_prior), 1 - 1e-4), 1e-4)
                        m.bias.data.fill_(math.log(p / (1 - p)))
                    else:
                        nn.init.zeros_(m.bias)
            else:
                if method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    model.apply(_init)
    return model


def train_one_fold(model, train_loader, val_loader, device, criterion, optimizer, patience=20, max_epochs=200, 
                  plot_metrics=False, fold=0, model_name="Custom"):
    """Train model for one fold with early stopping."""
    
    early_stopping = EarlyStopping(patience)

    train_losses = []
    val_losses = []
    val_aucs = []

    for epoch in range(max_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        probs, targets = [], []
        epoch_val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                prob = torch.sigmoid(outputs).float()
                probs.extend(prob.cpu().numpy())
                targets.extend(yb.cpu().numpy())
                epoch_val_loss += criterion(outputs, yb).item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        auc_score = roc_auc_score(targets, probs)
        val_aucs.append(auc_score)

        early_stopping(-avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(early_stopping.best_model_state)

    # Plot training curves if requested
    if plot_metrics:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1.plot(train_losses, label='Train Loss', color='blue')
        ax1.plot(val_losses, label='Val Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{model_name} - Training and Validation Loss Fold {fold + 1}')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # AUC plot
        ax2.plot(val_aucs, label='Val AUC', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('AUC')
        ax2.set_title(f'{model_name} - Validation AUC Fold {fold + 1}')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(str(BASE_DIR / f'outputs/BC/images/{model_name}_training_curves_fold_{fold + 1}.png'))
        plt.close()

    return model


def evaluate_model(model, loader, device):
    """Evaluate model and return AUC."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).float()
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    auc_score = roc_auc_score(all_labels, all_probs)
    return auc_score


def evaluate_and_plot_custom_results(model, model_name, X_test, y_test, device, threshold):
    """Evaluate model on test set and generate plots."""
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        test_outputs_presigmoid = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        test_outputs_prob = torch.sigmoid(test_outputs_presigmoid).float().cpu().numpy().flatten()
    
    # Compute metrics at multiple thresholds
    thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    results = {}
    for thresh in thresholds:
        preds = (test_outputs_prob > thresh).astype(float)
        f1 = f1_score(y_test, preds)
        acc = accuracy_score(y_test, preds)
        results[thresh] = {"f1": f1, "accuracy": acc}
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, test_outputs_prob)
    roc_auc = auc(fpr, tpr)
    
    # Print results
    print(f"\n=== {model_name} Model Test Results ===")
    print(f"ROC AUC on Test: {roc_auc:.4f}")
    print(f"Validation-selected threshold: {threshold:.2f}")
    for thresh, metrics in results.items():
        marker = " <-- VALIDATION SELECTED" if abs(thresh - threshold) < 0.01 else ""
        print(f"Threshold {thresh} - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}{marker}")
    
    # Metrics at validation-selected threshold
    y_pred = (test_outputs_prob > threshold).astype(int).flatten()
    f1_selected = f1_score(y_test, y_pred)
    acc_selected = accuracy_score(y_test, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n=== Confusion Matrix (Threshold = {threshold:.2f}) ===")
    print(f"True Negatives: {cm[0, 0]}, False Positives: {cm[0, 1]}")
    print(f"False Negatives: {cm[1, 0]}, True Positives: {cm[1, 1]}")
    
    # Plot confusion matrix
    fig = plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bad (0)', 'Good (1)'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'{model_name} Model - Confusion Matrix\n(Threshold = {threshold:.2f}, AUC = {roc_auc:.4f})')
    plt.tight_layout()
    plt.savefig(str(BASE_DIR / f'outputs/BC/images/confusion_matrix_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'{model_name} Model (AUC = {roc_auc:.4f})', color='purple', linestyle='-', linewidth=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Random Guess', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} Model - ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig(str(BASE_DIR / f'outputs/BC/images/roc_curve_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved:")
    print(f"  - outputs/BC/images/confusion_matrix_{model_name}.png")
    print(f"  - outputs/BC/images/roc_curve_{model_name}.png")
    
    return {
        'auc': roc_auc,
        'f1': f1_selected,
        'accuracy': acc_selected,
        'threshold': threshold
    }


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_custom_model(csv_path_train=str(BASE_DIR / 'data/IM_Data_Train.csv'), 
                      csv_path_test=str(BASE_DIR / 'data/IM_Data_Test.csv'),
                      model_name='Custom'):
    """
    Train custom model using the same procedure as train_and_save_best_model().
    """
    
    print("="*70)
    print(f"Training {model_name} Model")
    print("="*70)
    
    # Hyperparameters
    batch_size = CUSTOM_HYPERPARAMETERS['batch_size']
    dropout = CUSTOM_HYPERPARAMETERS['dropout']
    lr = CUSTOM_HYPERPARAMETERS['lr']
    weight_decay = CUSTOM_HYPERPARAMETERS['weight_decay']
    alpha = CUSTOM_HYPERPARAMETERS['alpha']
    gamma = CUSTOM_HYPERPARAMETERS['gamma']
    patience = CUSTOM_HYPERPARAMETERS['patience']
    max_epochs = CUSTOM_HYPERPARAMETERS['max_epochs']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load training data
    X_train, y_train, groups = load_dataset(csv_path_train, return_groups=True)
    print(f"\nTraining data: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # Use StratifiedGroupKFold if groups are available
    if groups is not None:
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        print(f"Using StratifiedGroupKFold to keep shots together in folds")
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        print(f"Using StratifiedKFold")

    # Track best model across folds
    auc_scores = []
    best_auc = 0
    best_model = None
    best_threshold = 0.5
    thresholds = []

    # Cross-validation
    if groups is not None:
        fold_iterator = skf.split(X_train, y_train, groups=groups)
    else:
        fold_iterator = skf.split(X_train, y_train)

    print(f"\n{'='*70}")
    print(f"Starting 5-Fold Cross-Validation")
    print(f"{'='*70}")

    for fold, (train_idx, val_idx) in enumerate(fold_iterator):
        print(f"\n--- Fold {fold + 1}/5 ---")
        
        # Initialize model
        model = CustomModel(input_size=X_train.shape[1], dropout=dropout).to(device)
        criterion = BinaryFocalLoss(alpha=alpha, gamma=gamma)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        init_weights_with_prior(model, pos_prior=float(y_train.mean()))

        # Prepare data loaders
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        train_ds = TensorDataset(torch.tensor(X_train_fold, dtype=torch.float32),
                                 torch.tensor(y_train_fold, dtype=torch.float32).unsqueeze(1))
        val_ds = TensorDataset(torch.tensor(X_val_fold, dtype=torch.float32),
                               torch.tensor(y_val_fold, dtype=torch.float32).unsqueeze(1))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        # Train
        model = train_one_fold(model, train_loader, val_loader, device, criterion, optimizer, 
                              patience, max_epochs, plot_metrics=True, fold=fold, model_name=model_name)

        # Find best threshold on validation set
        threshold_fold, score_fold = find_best_threshold(model, val_loader, device, metric='balanced')
        thresholds.append(threshold_fold)
        print(f"Fold {fold+1} - Best threshold: {threshold_fold:.2f} (balanced score: {score_fold:.4f})")

        # Evaluate
        auc_score = evaluate_model(model, val_loader, device)
        auc_scores.append(auc_score)
        print(f"Fold {fold+1} - Validation AUC: {auc_score:.4f}")

        if auc_score > best_auc:
            best_auc = auc_score
            best_model = model
            best_threshold = threshold_fold

    print(f"\n{'='*70}")
    print(f"Cross-Validation Results")
    print(f"{'='*70}")
    print(f"Best AUC across folds: {best_auc:.4f}")
    print(f"Mean AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"\nThreshold statistics:")
    print(f"Mean threshold: {np.mean(thresholds):.2f} ± {np.std(thresholds):.2f}")
    print(f"Best fold threshold: {best_threshold:.2f}")

    # Save best model
    torch.save(best_model.state_dict(), str(BASE_DIR / f"outputs/BC/models/best_model_{model_name}.pt"))
    print(f"\nBest model saved to: outputs/BC/models/best_model_{model_name}.pt")

    # Evaluate on test set
    print(f"\n{'='*70}")
    print(f"Final Test Set Evaluation")
    print(f"{'='*70}")
    X_test, y_test, _ = load_dataset(csv_path_test, return_groups=True)
    print(f"Test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    metrics = evaluate_and_plot_custom_results(best_model, model_name, X_test, y_test, device, best_threshold)
    
    print(f"\n{'='*70}")
    print(f"Final Results at Validation-Selected Threshold ({best_threshold:.2f})")
    print(f"{'='*70}")
    print(f"ROC AUC: {metrics['auc']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Mean Score: {(metrics['auc'] + metrics['f1'] + metrics['accuracy'])/3:.4f}")
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}\n")
    
    return best_model, metrics


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    start_time = time.time()
    
    # Train the custom model
    model, metrics = train_custom_model(
        csv_path_train=str(BASE_DIR / 'data/IM_Data_Train.csv'),
        csv_path_test=str(BASE_DIR / 'data/IM_Data_Test.csv'),
        model_name='Custom'
    )
    
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
