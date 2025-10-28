# IM_Binary_Quality_Recognition.py
#
# This code is part of a machine learning project for binary classification using MLPs with hyperparameter optimization (HPO) and pruning techniques.
# It includes model definition, training, evaluation, and visualization of results.

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split, GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, auc, roc_curve, RocCurveDisplay, roc_auc_score
from sklearn.feature_selection import mutual_info_classif
import time
import argparse
from plotly.io import show
import matplotlib.pyplot as plt
import optuna
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_rank
from optuna.visualization.matplotlib import plot_slice
from optuna.visualization.matplotlib import plot_timeline
import math

# Global variables
best_auc_global = 0
best_model_global = None
best_auc_RS_global = 0
best_model_RS_global = None
best_params_RS_global = None

batch_size = 32
# gamma = 3.0
dropout = 0.2
# lr = 2.5e-3
# n_layers = 1
# weight_decay = 5e-5

test_csv_path = 'data/IM_Data_Test.csv'  # Path to the test dataset
train_csv_path = 'data/IM_Data_Train.csv'  # Path to the training dataset


# === Model Definition ===
class BinaryClassifier(nn.Module):
    def __init__(self, input_size=18, layers_dim=1, dropout=0.2):
        super().__init__()

        layers = []

        for layer_dim in layers_dim:
            layers.append(nn.Linear(input_size, layer_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_size = layer_dim

        layers.append(nn.Linear(input_size, 1))

        self.net = nn.Sequential(*layers) 

    def forward(self, x):
        return self.net(x)


# === Binary Focal Loss ===
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).clamp(min=1e-7, max=1 - 1e-7)
        targets = targets.float()
        loss_pos = -self.alpha * (1 - probs) ** self.gamma * targets * torch.log(probs)
        loss_neg = -(1 - self.alpha) * probs ** self.gamma * (1 - targets) * torch.log(1 - probs)
        loss = loss_pos + loss_neg
        return loss.mean() if self.reduction == 'mean' else loss.sum()


# === EarlyStopping Callback ===
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, score, model):
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# === Data Loader ===
def load_dataset(csv_path, return_groups=False):
    df = pd.read_csv(csv_path)
    
    # Check if 'shot' column exists for grouping
    if 'shot' in df.columns and return_groups:
        groups = df['shot'].values
        X = df.drop(columns=['shot']).iloc[:, :-1].values
    else:
        groups = None
        X = df.iloc[:, :-1].values
    
    y = df.iloc[:, -1].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    if return_groups:
        return X, y, groups
    return X, y


# === Weight Initialization with Prior ===
def init_weights_with_prior(model: nn.Module, pos_prior: float | None = None, method: str = "kaiming"):
    """
    Initialize Linear layers:
      - Hidden: Kaiming (He) for ReLU
      - Output: Xavier; bias set to logit(pos_prior) if provided
    """
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

# === Outlier Detection using IQR ===
def detect_outliers_iqr(series, multiplier=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (series < lower_bound) | (series > upper_bound)


# === Training Function ===
def train_one_fold_test(model, train_loader, val_loader, device, criterion, optimizer, patience=5, max_epochs=100, plot_metrics=False, print_early_stopping=False, fold=0, sampler=""):
    
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

        # early_stopping(auc_score, model)
        early_stopping(-avg_val_loss, model)
        if early_stopping.early_stop:
            if print_early_stopping:
                print(f"Early stopping at epoch {epoch + 1}")
            break

    if print_early_stopping and not early_stopping.early_stop:
        print(f"Training completed after {max_epochs} epochs without early stopping.")

    model.load_state_dict(early_stopping.best_model_state)

    # Plot training curves if requested
    if plot_metrics:

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1.plot(train_losses, label='Train Loss', color='blue')
        ax1.plot(val_losses, label='Val Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('{} - Training and Validation Loss Fold {}'.format(sampler, fold + 1))
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # AUC plot
        ax2.plot(val_aucs, label='Val AUC', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('AUC')
        ax2.set_title('{} - Validation AUC Fold {}'.format(sampler, fold + 1))
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('images/{}_training_curves_fold_{}.png'.format(sampler, fold + 1))
        plt.close()

    return model


# === Training Function ===
def train_one_fold_hpo(model, train_loader, val_loader, device, criterion, optimizer, patience=5, max_epochs=100):
    
    early_stopping = EarlyStopping(patience)

    for epoch in range(max_epochs):
        # Training phase
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

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

        # auc_score = roc_auc_score(targets, probs)

        # early_stopping(auc_score, model)
        early_stopping(-avg_val_loss, model)
        if early_stopping.early_stop:
            break

    model.load_state_dict(early_stopping.best_model_state)

    return model


# === Evaluate Model ===
def evaluate_model(model, loader, device, metric='f1'):
    if metric not in ['f1', 'accuracy', 'auc']:
        raise ValueError("Metric must be 'f1' or 'accuracy'")
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).float()
            preds = (probs > 0.5).float()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    if metric == 'f1':
        f1 = f1_score(all_labels, all_preds)
        return f1
    elif metric == 'accuracy':
        accuracy = accuracy_score(all_labels, all_preds)
        return accuracy
    elif metric == 'auc':
        auc_score = roc_auc_score(all_labels, all_probs)
        return auc_score

# === Evaluate Model and plot results ===
def evaluate_and_plot_results(models_tp, models_rs, X_test, y_test, device, save_path="images/test_results_AUC.png", roc_curve_path="images/auc_opt_roc_curve.png"):
        """
        Evaluate all models from different folds on test set and select the best one based on test AUC.
        
        Parameters:
        - models_tp: list of TPE models from each fold
        - models_rs: list of RS models from each fold
        - X_test: test features
        - y_test: test labels
        - device: torch device
        """
        
        print("\n=== Evaluating TPE models on test set ===")
        best_auc_tp = 0
        best_model_tp = None
        best_fold_tp = -1
        test_aucs_tp = []
        
        for fold, model_tp in enumerate(models_tp):
            model_tp.eval()
            with torch.no_grad():
                test_outputs_presigmoid = model_tp(torch.tensor(X_test, dtype=torch.float32).to(device))
                test_outputs_prob = torch.sigmoid(test_outputs_presigmoid).float().cpu().numpy()
            
            fpr, tpr, _ = roc_curve(y_test, test_outputs_prob)
            test_auc = auc(fpr, tpr)
            test_aucs_tp.append(test_auc)
            
            print(f"Fold {fold+1} - Test AUC: {test_auc:.4f}")
            
            if test_auc > best_auc_tp:
                best_auc_tp = test_auc
                best_model_tp = model_tp
                best_fold_tp = fold
        
        print(f"\nBest TPE model: Fold {best_fold_tp+1} with Test AUC: {best_auc_tp:.4f}")
        print(f"TPE - Mean test AUC across folds: {np.mean(test_aucs_tp):.4f} ± {np.std(test_aucs_tp):.4f}")
        
        # Save best TPE model
        torch.save(best_model_tp.state_dict(), f"models/best_model_AUC_TP_fold_{best_fold_tp+1}.pt")
        
        print("\n=== Evaluating RS models on test set ===")
        best_auc_rs = 0
        best_model_rs = None
        best_fold_rs = -1
        test_aucs_rs = []
        
        for fold, model_rs in enumerate(models_rs):
            model_rs.eval()
            with torch.no_grad():
                test_outputs_presigmoid = model_rs(torch.tensor(X_test, dtype=torch.float32).to(device))
                test_outputs_prob = torch.sigmoid(test_outputs_presigmoid).float().cpu().numpy()
            
            fpr, tpr, _ = roc_curve(y_test, test_outputs_prob)
            test_auc = auc(fpr, tpr)
            test_aucs_rs.append(test_auc)
            
            print(f"Fold {fold+1} - Test AUC: {test_auc:.4f}")
            
            if test_auc > best_auc_rs:
                best_auc_rs = test_auc
                best_model_rs = model_rs
                best_fold_rs = fold
        
        print(f"\nBest RS model: Fold {best_fold_rs+1} with Test AUC: {best_auc_rs:.4f}")
        print(f"RS - Mean test AUC across folds: {np.mean(test_aucs_rs):.4f} ± {np.std(test_aucs_rs):.4f}")
        
        # Save best RS model
        torch.save(best_model_rs.state_dict(), f"models/best_model_AUC_RS_fold_{best_fold_rs+1}.pt")
        
        # Detailed evaluation of BEST models
        print(f"\n=== Detailed evaluation of best models ===")
        
        # Evaluate best TPE model
        best_model_tp.eval()
        with torch.no_grad():
            test_outputs_presigmoid_tp = best_model_tp(torch.tensor(X_test, dtype=torch.float32).to(device))
            test_outputs_prob_tp = torch.sigmoid(test_outputs_presigmoid_tp).float().cpu().numpy()
            thresholds_tp = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
            test_preds_tp = {threshold: (test_outputs_prob_tp > threshold).astype(float) for threshold in thresholds_tp}

        fpr_tp, tpr_tp, _ = roc_curve(y_test, test_outputs_prob_tp)
        roc_auc_tp = auc(fpr_tp, tpr_tp)

        results_tp = {}
        for threshold, preds in test_preds_tp.items():
            f1 = f1_score(y_test, preds)
            acc = accuracy_score(y_test, preds)
            results_tp[threshold] = {"f1": f1, "accuracy": acc}

        print(f"\nBest TPE Model (Fold {best_fold_tp+1}) - ROC AUC on Test: {roc_auc_tp:.4f}")
        for threshold, metrics in results_tp.items():
            print(f"Threshold {threshold} - F1 Score on Test (TP): {metrics['f1']:.4f}, Accuracy on Test (TP): {metrics['accuracy']:.4f}")

        # Evaluate best RS model
        best_model_rs.eval()
        with torch.no_grad():
            test_outputs_presigmoid_rs = best_model_rs(torch.tensor(X_test, dtype=torch.float32).to(device))
            test_outputs_prob_rs = torch.sigmoid(test_outputs_presigmoid_rs).float().cpu().numpy()
            thresholds_rs = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
            test_preds_rs = {threshold: (test_outputs_prob_rs > threshold).astype(float) for threshold in thresholds_rs}

        fpr_rs, tpr_rs, _ = roc_curve(y_test, test_outputs_prob_rs)
        roc_auc_rs = auc(fpr_rs, tpr_rs)

        results_rs = {}
        for threshold, preds in test_preds_rs.items():
            f1 = f1_score(y_test, preds)
            acc = accuracy_score(y_test, preds)
            results_rs[threshold] = {"f1": f1, "accuracy": acc}

        print(f"\nBest RS Model (Fold {best_fold_rs+1}) - ROC AUC on Test: {roc_auc_rs:.4f}")
        for threshold, metrics in results_rs.items():
            print(f"Threshold {threshold} - F1 Score on Test (RS): {metrics['f1']:.4f}, Accuracy on Test (RS): {metrics['accuracy']:.4f}")

        # Plot ROC curves together
        plt.figure(figsize=(10, 6))
        plt.plot(fpr_tp, tpr_tp, label=f'TPE Model Fold {best_fold_tp+1} (AUC = {roc_auc_tp:.4f})', color='red', linestyle='-', linewidth=2)
        plt.plot(fpr_rs, tpr_rs, label=f'Random Search Model Fold {best_fold_rs+1} (AUC = {roc_auc_rs:.4f})', color='blue', linestyle='--', linewidth=2)
        plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Random Guess', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison - Best Models Selected on Test Set')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.savefig(roc_curve_path)
        plt.close()
        
        print(f"\nROC curve saved to {roc_curve_path}")
        
        return best_model_tp, best_model_rs, best_fold_tp, best_fold_rs


# === Objective Function ===
def objective(trial, csv_path='data/DATA_ABS_&_PP_Binary.csv', n_startup_trials=10, sampler="RandomSampler"):
    global best_auc_global
    global best_model_global
    global best_auc_RS_global
    global best_model_RS_global
    global best_params_RS_global
    global batch_size
    # global gamma
    global dropout
    # global n_layers
    # global lr
    # global weight_decay

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y, groups = load_dataset(csv_path, return_groups=True)
    
    # Use StratifiedGroupKFold if groups are available, otherwise use StratifiedKFold
    if groups is not None:
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    neuron_min_limit = int(1/6 * X.shape[1])
    neuron_max_limit = 5 * X.shape[1]

    # Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 0.9)
    gamma = trial.suggest_float("gamma", 0.5, 5.0)
    # batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    # dropout = trial.suggest_float("dropout", 0.0, 0.4)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    n_layers = trial.suggest_int("n_layers", 1, 4)
    layers_dim = []
    for i in range(n_layers):
        size_layer = trial.suggest_int("size_layer{}".format(i), neuron_min_limit, neuron_max_limit, log=True)  # TODO Here I can start from 4 or mmaybe from X.shape[1] ??
        neuron_max_limit = size_layer  # Ensure next layer has less or equal neurons
        neuron_min_limit = 1
        layers_dim.append(size_layer)

    auc_scores = []
    best_auc = 0
    best_model = None

    # Split with groups if available
    if groups is not None:
        fold_iterator = skf.split(X, y, groups=groups)
    else:
        fold_iterator = skf.split(X, y)
    
    for fold, (train_idx, val_idx) in enumerate(fold_iterator):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                               torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        model = BinaryClassifier(input_size=X.shape[1], layers_dim=layers_dim, dropout=dropout).to(device)
        # Initialize using training fold positive prior
        init_weights_with_prior(model, pos_prior=float(y_train.mean()))
        criterion = BinaryFocalLoss(alpha=alpha, gamma=gamma)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Train
        model = train_one_fold_hpo(model, train_loader, val_loader, device, criterion, optimizer)

        # Evaluate
        auc_score = evaluate_model(model, val_loader, device, 'auc')
        auc_scores.append(auc_score)

        if auc_score > best_auc:
            best_auc = auc_score
            best_model = model
        
        trial.report(np.mean(auc_score), fold)


        if trial.should_prune():
            raise optuna.TrialPruned()

    if best_model is not None:
        if best_model_global is None or np.mean(auc_scores) > best_auc_global:
            best_auc_global = np.mean(auc_scores)
            best_model_global = best_model
            torch.save(best_model_global.state_dict(), "models/best_model_AUC_global.pt")
            if (sampler == "TPESampler") and (trial.number < n_startup_trials):
                    best_auc_RS_global = best_auc_global
                    best_model_RS_global = best_model_global
                    best_params_RS_global = trial.params  # Store the TRIAL parameters (hyperparameters), not model weights
                    torch.save(best_model_RS_global.state_dict(), "models/best_model_AUC_RS.pt")

    return np.mean(auc_scores)


# === Retrain Final Model ===
def train_and_save_best_model(params_tpe, params_rs, epochs=100, csv_path_train='data/IM_Data_Train.csv', csv_path_test='data/IM_Data_Test.csv'):
    global batch_size
    global dropout
    # global gamma
    # global lr
    # global n_layers
    # global weight_decay

    print(f"\nTraining the best model TPE and RS...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train data
    X_train, y_train, groups = load_dataset(csv_path_train, return_groups=True)
    print(f"The Data has {X_train.shape[0]} samples and {X_train.shape[1]} features.")

    # Use StratifiedGroupKFold if groups are available, otherwise use StratifiedKFold
    if groups is not None:
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        print(f"Using StratifiedGroupKFold to keep shots together in folds")
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        print(f"Using StratifiedKFold (no shot grouping available)")

    # Store ALL models from each fold
    models_tp = []
    val_auc_scores_tp = []

    models_rs = []
    val_auc_scores_rs = []

    # In final training I increase the patience of the early stopping to 20 and the numkber of epochs to 200
    early_stopping_patience = 20
    num_epochs = 200

    # Split with groups if available
    if groups is not None:
        fold_iterator = skf.split(X_train, y_train, groups=groups)
    else:
        fold_iterator = skf.split(X_train, y_train)

    for fold, (train_idx, val_idx) in enumerate(fold_iterator):
        
        # Initialize models for each fold
        # TPE
        model_tp = BinaryClassifier(input_size=X_train.shape[1], layers_dim=[params_tpe["size_layer{}".format(i)] for i in range(params_tpe["n_layers"])], dropout=dropout).to(device)
        criterion_tp = BinaryFocalLoss(alpha=params_tpe["alpha"], gamma=params_tpe["gamma"])
        optimizer_tp = torch.optim.AdamW(model_tp.parameters(), lr=params_tpe["lr"], weight_decay=params_tpe["weight_decay"])
        init_weights_with_prior(model_tp, pos_prior=float(y_train.mean()))
        #RS
        model_rs = BinaryClassifier(input_size=X_train.shape[1], layers_dim=[params_rs["size_layer{}".format(i)] for i in range(params_rs["n_layers"])], dropout=dropout).to(device)
        criterion_rs = BinaryFocalLoss(alpha=params_rs["alpha"], gamma=params_rs["gamma"])
        optimizer_rs = torch.optim.AdamW(model_rs.parameters(), lr=params_rs["lr"], weight_decay=params_rs["weight_decay"])
        init_weights_with_prior(model_rs, pos_prior=float(y_train.mean()))

        # Prepare data loaders
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        train_ds = TensorDataset(torch.tensor(X_train_fold, dtype=torch.float32),
                                 torch.tensor(y_train_fold, dtype=torch.float32).unsqueeze(1))
        val_ds = TensorDataset(torch.tensor(X_val_fold, dtype=torch.float32),
                               torch.tensor(y_val_fold, dtype=torch.float32).unsqueeze(1))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        # Train TP and RS
        model_tp = train_one_fold_test(model_tp, train_loader, val_loader, device, criterion_tp, optimizer_tp, early_stopping_patience, num_epochs, plot_metrics=True, print_early_stopping=True, fold=fold, sampler="TPE")
        model_rs = train_one_fold_test(model_rs, train_loader, val_loader, device, criterion_rs, optimizer_rs, early_stopping_patience, num_epochs, plot_metrics=True, print_early_stopping=True, fold=fold, sampler="RS")

        # Evaluate on validation set (for reference only)
        val_auc_tp = evaluate_model(model_tp, val_loader, device, 'auc')
        val_auc_rs = evaluate_model(model_rs, val_loader, device, 'auc')
        
        print(f"Fold {fold+1} - TPE validation AUC: {val_auc_tp:.4f}, RS validation AUC: {val_auc_rs:.4f}")
        
        # Store ALL models and their validation scores
        models_tp.append(model_tp)
        val_auc_scores_tp.append(val_auc_tp)
        
        models_rs.append(model_rs)
        val_auc_scores_rs.append(val_auc_rs)
        
        # Save each fold's model
        torch.save(model_tp.state_dict(), f"models/model_AUC_TP_fold_{fold+1}.pt")
        torch.save(model_rs.state_dict(), f"models/model_AUC_RS_fold_{fold+1}.pt")

    print(f"\nTPE: Mean validation AUC across folds: {np.mean(val_auc_scores_tp):.4f} ± {np.std(val_auc_scores_tp):.4f}")
    print(f"RS: Mean validation AUC across folds: {np.mean(val_auc_scores_rs):.4f} ± {np.std(val_auc_scores_rs):.4f}")

    # Model evaluation on test set
    print(f"\n=== Evaluating all models on test set ===")
    # Load test data (with return_groups=True to remove 'shot' column)
    X_test, y_test, _ = load_dataset(csv_path_test, return_groups=True)
    best_model_tp, best_model_rs, best_fold_tp, best_fold_rs = evaluate_and_plot_results(models_tp, models_rs, X_test, y_test, device=device, save_path="images/test_results_AUC.png", roc_curve_path="images/auc_opt_roc_curve.png")

    return best_model_tp, best_model_rs, best_fold_tp, best_fold_rs


# === Run Optuna Optimization ===
def run_optimization(sampler, pruner, csv_path='data/DATA_ABS_&_PP_Binary.csv', n_trials=100, n_startup_trials=10):
    global best_auc_RS_global
    global best_model_RS_global
    global best_params_RS_global
    
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(lambda trial: objective(trial, csv_path=csv_path, n_startup_trials=n_startup_trials, sampler=sampler.__class__.__name__), n_trials=n_trials, timeout=3600)

    # Best trial overall (hoping is TPE)
    if sampler.__class__.__name__ == "TPESampler":
        print("\n=== Best model TPE - after initial {} RS and {} TPE trials ===".format(n_startup_trials, (n_trials - n_startup_trials)))
    elif sampler.__class__.__name__ == "RandomSampler":
        print("\n=== RandomSampler ===")
    print("Best trial:")
    trial = study.best_trial
    print(f"  AUC Score: {trial.value:.4f}")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    # If TPE optimizer, Best trial RS
    if (sampler.__class__.__name__ == "TPESampler") and (n_startup_trials > 0) and (best_params_RS_global is not None):
            print("\n=== Best model RS - found with initial {} RS trials ===".format(n_startup_trials))
            print(f"  AUC Score: {best_auc_RS_global:.4f}")
            for key, value in best_params_RS_global.items():
                print(f"  {key}: {value}")

    # Visualizations
    # fig = plot_optimization_history(study)
    # fig = plot_intermediate_values(study)
    # fig = plot_parallel_coordinate(study)
    # fig = plot_contour(study)
    # fig = plot_slice(study)
    # fig = plot_param_importances(study)
    # fig = plot_edf(study)
    # fig = plot_rank(study)
    # fig = plot_timeline(study)
    # plt.title("Parameters importances")
    # plt.show()

    return trial


if __name__ == "__main__":
    start_time = time.time()
    # # Argument parser
    # parser = argparse.ArgumentParser(description="Train a binary classification model.")
    # parser.add_argument('--first_data', action='store_true', help="Decide whether to use first dataset, if selected only the first dataset will be used")
    # parser.add_argument('--first_data_full', action='store_true', help="Decide whether to use first dataset with also measurements, if selected only the first dataset with measurement will be used")
    # parser.add_argument('--pp_data', action='store_true', help="Decide whether to use PP dataset, can be used together with ABS dataset")
    # parser.add_argument('--abs_data', action='store_true', help="Decide whether to use the ABS dataset, can be used together with PP dataset")
    # parser.add_argument('--pp_p1_data', action='store_true', help="Decide whether to use the PP Position 1 dataset")
    # parser.add_argument('--pp_weight_data', action='store_true', help="Decide whether to use the PP Weight dataset")
    # args = parser.parse_args()
    # # Load data path
    # if args.first_data:
    #     print("\nUsing the first dataset.\n")
    #     csv_path = 'data/IM_Data.csv'
    # elif args.first_data_full:
    #     print("\nUsing the first dataset with all measurments.\n")
    #     csv_path = 'data/IM_Data_Full.csv'
    # elif args.pp_p1_data:
    #     print("\nUsing only the PP Position 1 dataset.\n")
    #     csv_path = 'data/DATA_PP_P1_W.csv'
    # else:
    #     if args.pp_data:
    #         if args.abs_data:
    #             print("\nUsing the full dataset, both PP and ABS data.\n")
    #             csv_path = 'data/DATA_ABS_&_PP_Binary.csv'
    #         else:
    #             print("\nUsing only the PP dataset.\n")
    #             csv_path = 'data/DATA_PP_Binary.csv'
    #     elif args.abs_data:
    #         print("\nUsing only the ABS dataset.\n")
    #         csv_path = 'data/DATA_ABS_Binary.csv'
    #     else:
    #         print("\nUsing by default the full dataset, both PP and ABS data.\n")
    #         csv_path = 'data/DATA_ABS_&_PP_Binary.csv'

    # CSV path
    csv_path_1 = 'data/DATA_PP_P1_W.csv'
    csv_path_2 = 'data/DATA_PP_P2_W.csv'

    # Read data
    df_1 = pd.read_csv(csv_path_1)
    df_2 = pd.read_csv(csv_path_2)
    
    # Check if both datasets have 'shot' column
    if 'shot' not in df_1.columns or 'shot' not in df_2.columns:
        raise ValueError("Both datasets must have 'shot' column for synchronized splitting")
    
    print(f"Dataset P1: {len(df_1)} samples")
    print(f"Dataset P2: {len(df_2)} samples")

    # Detect outliers in both datasets
    outliers_p1 = detect_outliers_iqr(df_1['Product weight g'])
    outliers_p2 = detect_outliers_iqr(df_2['Product weight g'])

    print(f"Dataset P1 - Outliers detected: {outliers_p1.sum()}")
    print(f"Dataset P2 - Outliers detected: {outliers_p2.sum()}")

    # Remove outliers from both datasets just for statistics computation
    df_1_clean = df_1[~outliers_p1].reset_index(drop=True)
    df_2_clean = df_2[~outliers_p2].reset_index(drop=True)

    # Compute statistics for clean dataset 1 (P1)
    mean_w_clean_1 = df_1_clean['Product weight g'].mean()
    std_dev_w_clean_1 = df_1_clean['Product weight g'].std()
    print(f"Dataset P1 (clean from outlayers) - Product weight g: mean={mean_w_clean_1:.4f}, std={std_dev_w_clean_1:.4f}")

    # Compute statistics for clean dataset 2 (P2)
    mean_w_clean_2 = df_2_clean['Product weight g'].mean()
    std_dev_w_clean_2 = df_2_clean['Product weight g'].std()
    print(f"Dataset P2 (clean from outlayers) - Product weight g: mean={mean_w_clean_2:.4f}, std={std_dev_w_clean_2:.4f}")

    # Create Product_Goodness column for dataset 1 (P1) using its own statistics
    df_1['Product_Goodness'] = (
        (df_1['Product weight g'] >= mean_w_clean_1 - std_dev_w_clean_1) & 
        (df_1['Product weight g'] <= mean_w_clean_1 + std_dev_w_clean_1)
    ).astype(int)
    
    # Create Product_Goodness column for dataset 2 (P2) using its own statistics
    df_2['Product_Goodness'] = (
        (df_2['Product weight g'] >= mean_w_clean_2 - std_dev_w_clean_2) & 
        (df_2['Product weight g'] <= mean_w_clean_2 + std_dev_w_clean_2)
    ).astype(int)
    
    print(f"Created 'Product_Goodness' column for both datasets based on their own statistics")

    # Compute percentage of good and bad parts and % of bad outlayers for both datasets
    print("\n=== Labeling Statistics ===")
    print(f"P1 - Total samples: {len(df_1)}, Good: {df_1['Product_Goodness'].sum()}, Bad: {(df_1['Product_Goodness'] == 0).sum()}")
    print(f"P1 - Outliers that are labeled as BAD: {(outliers_p1 & (df_1['Product_Goodness'] == 0)).sum()} / {outliers_p1.sum()}")
    print(f"P2 - Total samples: {len(df_2)}, Good: {df_2['Product_Goodness'].sum()}, Bad: {(df_2['Product_Goodness'] == 0).sum()}")
    print(f"P2 - Outliers that are labeled as BAD: {(outliers_p2 & (df_2['Product_Goodness'] == 0)).sum()} / {outliers_p2.sum()}")
    
    # Drop Product weight g column from both datasets (but KEEP 'shot' column for grouping)
    df_1 = df_1.drop(columns=['Product weight g'])
    df_2 = df_2.drop(columns=['Product weight g'])
    
    print(f"Removed 'Product weight g' columns from both datasets (kept 'shot' for group-based CV)")
    
    # Split shot numbers into train and test (80/20)
    unique_shots = df_1['shot'].unique()
    np.random.seed(41)
    shuffled_shots = np.random.permutation(unique_shots)
    split_idx = int(len(shuffled_shots) * 0.8)
    train_shots = shuffled_shots[:split_idx]
    test_shots = shuffled_shots[split_idx:]
    print(f"Train shots: {len(train_shots)}, Test shots: {len(test_shots)}")

    # Split df_1 based on shot numbers
    Data_train_df_1 = df_1[df_1['shot'].isin(train_shots)].copy()
    Data_test_df_1 = df_1[df_1['shot'].isin(test_shots)].copy()

    print(f"Dataset P1 - Train: {len(Data_train_df_1)} samples, Test: {len(Data_test_df_1)} samples")
    
    # Split df_2 based on shot numbers
    Data_train_df_2 = df_2[df_2['shot'].isin(train_shots)].copy()
    Data_test_df_2 = df_2[df_2['shot'].isin(test_shots)].copy()
    
    print(f"Dataset P2 - Train: {len(Data_train_df_2)} samples, Test: {len(Data_test_df_2)} samples")
    
    # Combine train datasets from P1 and P2
    Data_train_combined = pd.concat([Data_train_df_1, Data_train_df_2], axis=0, ignore_index=True)
    
    # Combine test datasets from P1 and P2
    Data_test_combined = pd.concat([Data_test_df_1, Data_test_df_2], axis=0, ignore_index=True)
    
    print(f"Combined - Train: {len(Data_train_combined)} samples, Test: {len(Data_test_combined)} samples")
    
    # Compute percentage of 0s in Product_Goodness column for train/test
    train_zero_pct = (Data_train_combined['Product_Goodness'] == 0).sum() / len(Data_train_combined) * 100
    test_zero_pct = (Data_test_combined['Product_Goodness'] == 0).sum() / len(Data_test_combined) * 100
    
    print(f"Percentage of 0s in Product_Goodness - Train: {train_zero_pct:.2f}%, Test: {test_zero_pct:.2f}%")
    
    # Shuffle but keep shot groups together (shuffle at shot level, not row level)
    train_shots_order = Data_train_combined['shot'].unique()
    np.random.seed(42)
    np.random.shuffle(train_shots_order)
    Data_train_combined = Data_train_combined.set_index('shot').loc[train_shots_order].reset_index()
    
    test_shots_order = Data_test_combined['shot'].unique()
    np.random.seed(42)
    np.random.shuffle(test_shots_order)
    Data_test_combined = Data_test_combined.set_index('shot').loc[test_shots_order].reset_index()

    print(f"Shuffled datasets (preserving shot groups) - Train: {len(Data_train_combined)} samples, Test: {len(Data_test_combined)} samples")

    # print(f"Shuffled datasets - Train: {len(Data_train_df_1)} samples, Test: {len(Data_test_df_1)} samples")
    
    # Save combined datasets
    Data_train_combined.to_csv(train_csv_path, index=False)
    Data_test_combined.to_csv(test_csv_path, index=False)
    print(f"Saved combined train and test datasets to {train_csv_path} and {test_csv_path}")
    # print(Data_train_combined, Data_test_combined)

    # # Save individual datasets
    # Data_train_df_1.to_csv(train_csv_path, index=False)
    # Data_test_df_1.to_csv(test_csv_path, index=False)
    # print(f"Saved individual train and test datasets to {train_csv_path} and {test_csv_path}")
    # # print(Data_train_df, Data_test_df)

    # # Compute Mutual Information (MI) Scores for each feature in train data only
    # X_train_mi = Data_train_df.iloc[:, :-1].values
    # y_train_mi = Data_train_df.iloc[:, -1].values
    # MI_scores = mutual_info_classif(X_train_mi, y_train_mi, random_state=42)
    # for i, score in enumerate(MI_scores):
    #     print(f"Feature {i}: MI Score = {score:.4f}")
    # plt.figure(figsize=(10, 6))
    # plt.bar(range(len(MI_scores)), MI_scores, color='skyblue')
    # plt.xlabel('Feature Index')
    # plt.ylabel('Mutual Information Score')
    # plt.title('Mutual Information Scores for Features')
    # plt.xticks(range(len(MI_scores)), df.columns[:-1], rotation=45, ha='right')
    # plt.tight_layout()
    # plt.savefig("images/mi_scores.png")
    # plt.show()
    
    # # Run HPO otpimization with RS sampler and MedianPruner
    # print(f"\nStarting RS optimization...\n")
    # sampler = optuna.samplers.RandomSampler(seed=42)  # Use RandomSampler for simplicity
    # # pruner = optuna.pruners.MedianPruner(n_warmup_steps=1, n_startup_trials=10)
    # pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=80, reduction_factor=3)
    # best_trial_rs = run_optimization(sampler, pruner, train_csv_path, n_trials=40)

    # Run HPO otpimization with TPE sampler and HyperbandPruner
    print(f"\nStarting TPE optimization...\n")
    n_startup_trials = 100
    n_trials = 200
    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials, seed=42) #(n_startup_trials=10, seed=31) # Here tried to add some startup trials
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=80, reduction_factor=3)
    best_trial_tpe = run_optimization(sampler, pruner, train_csv_path, n_trials=n_trials, n_startup_trials=n_startup_trials)

    # Retrain the best models
    _, _, _, _ = train_and_save_best_model(params_tpe=best_trial_tpe.params, params_rs=best_params_RS_global, epochs=200, csv_path_train=train_csv_path, csv_path_test=test_csv_path)

    # Print total time taken
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
