# Reg_MLP_IM.py
#
# Regression using MLPs for weight prediction for Injection Molding (IM) data.
#
# This code is part of a machine learning project for regression using MLPs with hyperparameter optimization (HPO) and pruning techniques.
# It includes model definition, training, evaluation, and visualization of results.

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import matplotlib.pyplot as plt
import optuna
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # project root (works from any subfolder)

# Global variables
best_mae_global = float('inf')
best_model_global = None
best_mae_RS_global = float('inf')
best_model_RS_global = None
best_params_RS_global = None

batch_size = 32
# dropout = 0.2
# lr = 2.5e-3
# n_layers = 1
# weight_decay = 5e-5

test_csv_path = str(BASE_DIR / 'data/IM_Data_Test.csv')  # Path to the test dataset
train_csv_path = str(BASE_DIR / 'data/IM_Data_Train.csv')  # Path to the training dataset


# === Model Definition ===
class MLPRegression(nn.Module):
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


# === Weight Initialization for Regression ===
def init_weights_with_prior(model: nn.Module, pos_prior: float | None = None, method: str = "kaiming"):
    """
    Initialize Linear layers for regression:
      - Hidden: Kaiming (He) for ReLU
      - Output: Xavier for weights; bias set to target mean if provided
    
    Parameters:
    - pos_prior: For regression, this should be the mean of the target values
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
                # Output layer: Xavier initialization
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    if pos_prior is not None:
                        # For regression: initialize bias to mean of target values
                        m.bias.data.fill_(float(pos_prior))
                    else:
                        nn.init.zeros_(m.bias)
            else:
                # Hidden layers: Kaiming initialization for ReLU
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
    val_r2_scores = []

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
        epoch_val_loss = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                epoch_val_loss += criterion(outputs, yb).item()
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(yb.cpu().numpy().flatten())

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        r2 = r2_score(all_targets, all_preds)
        val_r2_scores.append(r2)

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
        ax1.plot(train_losses, label='Train MAE Loss', color='blue')
        ax1.plot(val_losses, label='Val MAE Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MAE Loss')
        ax1.set_title('{} - Training and Validation Loss Fold {}'.format(sampler, fold + 1))
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # R² plot
        ax2.plot(val_r2_scores, label='Val R² Score', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('R² Score')
        ax2.set_title('{} - Validation R² Score Fold {}'.format(sampler, fold + 1))
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(str(BASE_DIR / 'outputs/Reg/images/{}_training_curves_fold_{}.png'.format(sampler, fold + 1)))
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
        epoch_val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                epoch_val_loss += criterion(outputs, yb).item()

        avg_val_loss = epoch_val_loss / len(val_loader)

        early_stopping(-avg_val_loss, model)
        if early_stopping.early_stop:
            break

    model.load_state_dict(early_stopping.best_model_state)

    return model


# === Evaluate Model ===
def evaluate_model(model, loader, device, metric='mae'):
    if metric not in ['mae', 'rmse', 'r2']:
        raise ValueError("Metric must be 'mae' or 'rmse' or 'r2'")
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    if metric == 'mae':
        mae = mean_absolute_error(all_labels, all_preds)
        return mae
    elif metric == 'rmse':
        rmse = mean_squared_error(all_labels, all_preds, squared=False)
        return rmse
    elif metric == 'r2':
        r2 = r2_score(all_labels, all_preds)
        return r2
    return None


# === Save Best Overall Model ===
def save_best_overall_model(model, model_name, mae, rmse, r2, mape, max_error, X_train, y_train, X_test, y_test, params):
    """
    Save the best overall model with its metadata, data, and hyperparameters.
    Only saves if the current model is better than the previously saved one.
    
    Parameters:
    - model: trained PyTorch model
    - model_name: 'TPE' or 'RS'
    - mae: Mean Absolute Error
    - rmse: Root Mean Squared Error
    - r2: R² score
    - mape: Mean Absolute Percentage Error
    - max_error: Maximum error
    - X_train, y_train: training data
    - X_test, y_test: test data
    - params: hyperparameters dictionary
    
    Returns:
    - saved: True if model was saved (better than previous), False otherwise
    """
    best_model_dir = str(BASE_DIR / 'outputs/Reg/models/best_model_overall')
    metadata_file = os.path.join(best_model_dir, 'metadata.json')
    
    # Check if there's a previous best model
    should_save = True
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            prev_metadata = json.load(f)
        
        prev_mae = prev_metadata['mae']
        
        print(f"\n=== Comparing with previous best model ===")
        print(f"Previous best: {prev_metadata['model_name']} - MAE: {prev_mae:.4f}, RMSE: {prev_metadata['rmse']:.4f}, R²: {prev_metadata['r2']:.4f}")
        print(f"Current model: {model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        if mae >= prev_mae:
            print(f"Current model is not better than previous best. Not saving.")
            should_save = False
        else:
            print(f"Current model is BETTER! Overwriting previous best model.")
    else:
        print(f"\n=== No previous best model found. Saving current model as best. ===")
        print(f"Current model: {model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    if should_save:
        # Create directory if it doesn't exist (or clear it)
        if os.path.exists(best_model_dir):
            shutil.rmtree(best_model_dir)
        os.makedirs(best_model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(best_model_dir, f'best_model_{model_name}.pt')
        torch.save(model.state_dict(), model_path)
        
        # Save training data
        train_data = pd.DataFrame(X_train)
        train_data['Product Weight g'] = y_train
        train_data.to_csv(os.path.join(best_model_dir, 'train_data.csv'), index=False)
        
        # Save test data
        test_data = pd.DataFrame(X_test)
        test_data['Product Weight g'] = y_test
        test_data.to_csv(os.path.join(best_model_dir, 'test_data.csv'), index=False)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
            'max_error': float(max_error),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'hyperparameters': params,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': X_train.shape[1]
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Copy visualization files for the best model
        scatter_plot_src = str(BASE_DIR / f'outputs/Reg/images/scatter_plot_{model_name}.png')
        scatter_plot_dst = os.path.join(best_model_dir, 'scatter_plot.png')
        if os.path.exists(scatter_plot_src):
            shutil.copy2(scatter_plot_src, scatter_plot_dst)
        
        residual_plot_src = str(BASE_DIR / f'outputs/Reg/images/residual_plot_{model_name}.png')
        residual_plot_dst = os.path.join(best_model_dir, 'residual_plot.png')
        if os.path.exists(residual_plot_src):
            shutil.copy2(residual_plot_src, residual_plot_dst)
        
        # Copy comparison plots
        comparison_src = str(BASE_DIR / 'outputs/Reg/images/metrics_comparison.png')
        comparison_dst = os.path.join(best_model_dir, 'metrics_comparison.png')
        if os.path.exists(comparison_src):
            shutil.copy2(comparison_src, comparison_dst)
        
        print(f"\n✅ Best overall model saved to {best_model_dir}/")
        print(f"   - Model: best_model_{model_name}.pt")
        print(f"   - Train data: train_data.csv ({len(X_train)} samples)")
        print(f"   - Test data: test_data.csv ({len(X_test)} samples)")
        print(f"   - Metadata: metadata.json")
        print(f"   - Scatter plot: scatter_plot.png")
        print(f"   - Residual plot: residual_plot.png")
        print(f"   - Metrics comparison: metrics_comparison.png")
        
    return should_save

# === Evaluate Model and plot results ===
def evaluate_and_plot_results(model_tp, model_rs, X_test, y_test, device, save_path=str(BASE_DIR / "outputs/Reg/images/test_results.png")):
    """
    Evaluate regression models and create visualizations.
    
    Parameters:
    - model_tp: TPE optimized model
    - model_rs: Random search optimized model
    - X_test: test features
    - y_test: test targets
    - device: torch device
    - save_path: path to save comparison plot
    
    Returns:
    - Dictionary with metrics for both models
    """
    # Evaluate model_tp
    model_tp.eval()
    with torch.no_grad():
        y_pred_tp = model_tp(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy().flatten()

    mae_tp = mean_absolute_error(y_test, y_pred_tp)
    rmse_tp = np.sqrt(mean_squared_error(y_test, y_pred_tp))
    r2_tp = r2_score(y_test, y_pred_tp)
    mape_tp = np.mean(np.abs((y_test - y_pred_tp) / y_test)) * 100
    max_error_tp = np.max(np.abs(y_test - y_pred_tp))

    print(f"\n=== TPE Model Test Results ===")
    print(f"MAE: {mae_tp:.4f}")
    print(f"RMSE: {rmse_tp:.4f}")
    print(f"R²: {r2_tp:.4f}")
    print(f"MAPE: {mape_tp:.2f}%")
    print(f"Max Error: {max_error_tp:.4f}")

    # Evaluate model_rs
    model_rs.eval()
    with torch.no_grad():
        y_pred_rs = model_rs(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy().flatten()

    mae_rs = mean_absolute_error(y_test, y_pred_rs)
    rmse_rs = np.sqrt(mean_squared_error(y_test, y_pred_rs))
    r2_rs = r2_score(y_test, y_pred_rs)
    mape_rs = np.mean(np.abs((y_test - y_pred_rs) / y_test)) * 100
    max_error_rs = np.max(np.abs(y_test - y_pred_rs))

    print(f"\n=== RS Model Test Results ===")
    print(f"MAE: {mae_rs:.4f}")
    print(f"RMSE: {rmse_rs:.4f}")
    print(f"R²: {r2_rs:.4f}")
    print(f"MAPE: {mape_rs:.2f}%")
    print(f"Max Error: {max_error_rs:.4f}")

    # Create scatter plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # TPE scatter plot
    ax1.scatter(y_test, y_pred_tp, alpha=0.5, s=30)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect prediction')
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predictions')
    ax1.set_title(f'TPE Model: Predicted vs True\nMAE={mae_tp:.4f}, R²={r2_tp:.4f}')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # RS scatter plot
    ax2.scatter(y_test, y_pred_rs, alpha=0.5, s=30, color='green')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect prediction')
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Predictions')
    ax2.set_title(f'RS Model: Predicted vs True\nMAE={mae_rs:.4f}, R²={r2_rs:.4f}')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(BASE_DIR / 'outputs/Reg/images/scatter_plots_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save individual scatter plots
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_tp, alpha=0.5, s=30)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'TPE Model: Predicted vs True\nMAE={mae_tp:.4f}, R²={r2_tp:.4f}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(str(BASE_DIR / 'outputs/Reg/images/scatter_plot_TPE.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_rs, alpha=0.5, s=30, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'RS Model: Predicted vs True\nMAE={mae_rs:.4f}, R²={r2_rs:.4f}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(str(BASE_DIR / 'outputs/Reg/images/scatter_plot_RS.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create residual plots
    residuals_tp = y_test - y_pred_tp
    residuals_rs = y_test - y_pred_rs
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # TPE residual plot
    ax1.scatter(y_pred_tp, residuals_tp, alpha=0.5, s=30)
    ax1.axhline(y=0, color='r', linestyle='--', lw=2)
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title(f'TPE Model: Residual Plot\nMAE={mae_tp:.4f}')
    ax1.grid(alpha=0.3)
    
    # RS residual plot
    ax2.scatter(y_pred_rs, residuals_rs, alpha=0.5, s=30, color='green')
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title(f'RS Model: Residual Plot\nMAE={mae_rs:.4f}')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(BASE_DIR / 'outputs/Reg/images/residual_plots_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save individual residual plots
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred_tp, residuals_tp, alpha=0.5, s=30)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'TPE Model: Residual Plot\nMAE={mae_tp:.4f}')
    plt.grid(alpha=0.3)
    plt.savefig(str(BASE_DIR / 'outputs/Reg/images/residual_plot_TPE.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred_rs, residuals_rs, alpha=0.5, s=30, color='green')
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'RS Model: Residual Plot\nMAE={mae_rs:.4f}')
    plt.grid(alpha=0.3)
    plt.savefig(str(BASE_DIR / 'outputs/Reg/images/residual_plot_RS.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create metrics comparison bar chart
    metrics_names = ['MAE', 'RMSE', 'R²', 'MAPE (%)', 'Max Error']
    tp_values = [mae_tp, rmse_tp, r2_tp, mape_tp, max_error_tp]
    rs_values = [mae_rs, rmse_rs, r2_rs, mape_rs, max_error_rs]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, tp_values, width, label='TPE Model', alpha=0.8)
    bars2 = ax.bar(x + width/2, rs_values, width, label='RS Model', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Regression Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(str(BASE_DIR / 'outputs/Reg/images/metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved:")
    print(f"  - outputs/Reg/images/scatter_plot_TPE.png")
    print(f"  - outputs/Reg/images/scatter_plot_RS.png")
    print(f"  - outputs/Reg/images/scatter_plots_comparison.png")
    print(f"  - outputs/Reg/images/residual_plot_TPE.png")
    print(f"  - outputs/Reg/images/residual_plot_RS.png")
    print(f"  - outputs/Reg/images/residual_plots_comparison.png")
    print(f"  - outputs/Reg/images/metrics_comparison.png")
    
    # Return metrics for both models
    return {
        'tp': {
            'mae': mae_tp,
            'rmse': rmse_tp,
            'r2': r2_tp,
            'mape': mape_tp,
            'max_error': max_error_tp
        },
        'rs': {
            'mae': mae_rs,
            'rmse': rmse_rs,
            'r2': r2_rs,
            'mape': mape_rs,
            'max_error': max_error_rs
        }
    }


# === Objective Function ===
def objective(trial, csv_path=str(BASE_DIR / 'data/DATA_ABS_&_PP_Binary.csv'), n_startup_trials=10, sampler="RandomSampler"):
    global best_mae_global
    global best_model_global
    global best_mae_RS_global
    global best_model_RS_global
    global best_params_RS_global
    global batch_size
    # global dropout
    # global n_layers
    # global lr
    # global weight_decay

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y, groups = load_dataset(csv_path, return_groups=True)
    
    # Use GroupKFold if groups are available, otherwise use KFold
    if groups is not None:
        skf = GroupKFold(n_splits=5)
    else:
        skf = KFold(n_splits=5, shuffle=True, random_state=42)

    neuron_min_limit = X.shape[1] #int(1/6 * X.shape[1])
    neuron_max_limit = 20 * X.shape[1]

    # Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    # batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    weight_decay = trial.suggest_float("weight_decay", 0, 1e-2) #, log=True)
    n_layers = trial.suggest_int("n_layers", 2, 4)
    layers_dim = []
    for i in range(n_layers):
        size_layer = trial.suggest_int("size_layer{}".format(i), neuron_min_limit, neuron_max_limit, log=True)  # TODO Here I can start from 4 or mmaybe from X.shape[1] ??
        neuron_max_limit = size_layer  # Ensure next layer has less or equal neurons
        neuron_min_limit = 1
        layers_dim.append(size_layer)

    mae_values = []
    best_mae = float('inf')
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

        model = MLPRegression(input_size=X.shape[1], layers_dim=layers_dim, dropout=dropout).to(device)
        # Initialize using training fold positive prior
        init_weights_with_prior(model, pos_prior=float(y_train.mean()))
        criterion = nn.L1Loss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Train
        model = train_one_fold_hpo(model, train_loader, val_loader, device, criterion, optimizer)

        # Evaluate
        mae_value = evaluate_model(model, val_loader, device, 'mae')
        mae_values.append(mae_value)

        if mae_value < best_mae:
            best_mae = mae_value
            best_model = model
        
        trial.report(np.mean(mae_values), fold)

        if trial.should_prune():
            raise optuna.TrialPruned()

    mae_mean = np.mean(mae_values)

    if best_model is not None:
        if best_model_global is None or mae_mean < best_mae_global:
            best_mae_global = mae_mean
            best_model_global = best_model
            torch.save(best_model_global.state_dict(), str(BASE_DIR / "outputs/Reg/models/best_model_MAE_global.pt"))
            if (sampler == "TPESampler") and (trial.number < n_startup_trials):
                    best_mae_RS_global = best_mae_global
                    best_model_RS_global = best_model_global
                    best_params_RS_global = trial.params  # Store the TRIAL parameters (hyperparameters), not model weights
                    torch.save(best_model_RS_global.state_dict(), str(BASE_DIR / "outputs/Reg/models/best_model_MAE_RS.pt"))

    return mae_mean


# === Retrain Final Model ===
def train_and_save_best_model(params_tpe, params_rs, epochs=100, csv_path_train=str(BASE_DIR / 'data/IM_Data_Train.csv'), csv_path_test=str(BASE_DIR / 'data/IM_Data_Test.csv')):
    global batch_size
    # global dropout
    # global lr
    # global n_layers
    # global weight_decay

    print(f"\nTraining the best model TPE and RS...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train data
    X_train, y_train, groups = load_dataset(csv_path_train, return_groups=True)
    print(f"The Data has {X_train.shape[0]} samples and {X_train.shape[1]} features.")

    # Use GroupKFold if groups are available, otherwise use KFold
    if groups is not None:
        skf = GroupKFold(n_splits=5)
        print(f"Using GroupKFold to keep shots together in folds")
    else:
        skf = KFold(n_splits=5, shuffle=True, random_state=42)
        print(f"Using KFold (no shot grouping available)")

    mae_values_tp = []
    best_mae_tp = float('inf')
    best_model_tp = None
    best_val_loader_tp = None

    mae_values_rs = []
    best_mae_rs = float('inf')
    best_model_rs = None
    best_val_loader_rs = None

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
        model_tp = MLPRegression(input_size=X_train.shape[1], layers_dim=[params_tpe["size_layer{}".format(i)] for i in range(params_tpe["n_layers"])], dropout=params_tpe["dropout"]).to(device)
        criterion_tp = nn.L1Loss()
        optimizer_tp = torch.optim.AdamW(model_tp.parameters(), lr=params_tpe["lr"], weight_decay=params_tpe["weight_decay"])
        init_weights_with_prior(model_tp, pos_prior=float(y_train.mean()))
        #RS
        model_rs = MLPRegression(input_size=X_train.shape[1], layers_dim=[params_rs["size_layer{}".format(i)] for i in range(params_rs["n_layers"])], dropout=params_rs["dropout"]).to(device)
        criterion_rs = nn.L1Loss()
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
        
        # train_loader_tpe = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        # val_loader_tpe = DataLoader(val_ds, batch_size=batch_size)

        # train_loader_rs = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        # val_loader_rs = DataLoader(val_ds, batch_size=batch_size)

        # Train TP and RS
        model_tp = train_one_fold_test(model_tp, train_loader, val_loader, device, criterion_tp, optimizer_tp, early_stopping_patience, num_epochs, plot_metrics=True, print_early_stopping=True, fold=fold, sampler="TPE")
        model_rs = train_one_fold_test(model_rs, train_loader, val_loader, device, criterion_rs, optimizer_rs, early_stopping_patience, num_epochs, plot_metrics=True, print_early_stopping=True, fold=fold, sampler="RS")

        # Evaluate TP
        mae_tp = evaluate_model(model_tp, val_loader, device, 'mae')
        mae_values_tp.append(mae_tp)
        if mae_tp < best_mae_tp:
            best_mae_tp = mae_tp
            best_model_tp = model_tp
            best_val_loader_tp = val_loader

        # Evaluate RS
        mae_rs = evaluate_model(model_rs, val_loader, device, 'mae')
        mae_values_rs.append(mae_rs)
        if mae_rs < best_mae_rs:
            best_mae_rs = mae_rs
            best_model_rs = model_rs
            best_val_loader_rs = val_loader

    print(f"\nTP: Best MAE across folds: {best_mae_tp:.4f} and mean MAE: {np.mean(mae_values_tp):.4f}, after {len(mae_values_tp)} folds.")
    print(f"RS: Best MAE across folds: {best_mae_rs:.4f} and mean MAE: {np.mean(mae_values_rs):.4f}, after {len(mae_values_rs)} folds.")

    torch.save(best_model_tp.state_dict(), str(BASE_DIR / "outputs/Reg/models/best_model_MAE_TP.pt"))
    torch.save(best_model_rs.state_dict(), str(BASE_DIR / "outputs/Reg/models/best_model_MAE_RS.pt"))

    # Model evaluation
    print(f"\n=== Final Test Set Evaluation ===")
    # Load test data (with return_groups=True to remove 'shot' column)
    X_test, y_test, _ = load_dataset(csv_path_test, return_groups=True)
    metrics = evaluate_and_plot_results(best_model_tp, best_model_rs, X_test, y_test, device=device)
    
    # Determine which model is better (based on MAE - lower is better)
    mae_tp = metrics['tp']['mae']
    mae_rs = metrics['rs']['mae']
    
    print(f"\n=== Final Model Comparison ===")
    print(f"TPE Model - MAE: {mae_tp:.4f}, RMSE: {metrics['tp']['rmse']:.4f}, R²: {metrics['tp']['r2']:.4f}")
    print(f"RS Model  - MAE: {mae_rs:.4f}, RMSE: {metrics['rs']['rmse']:.4f}, R²: {metrics['rs']['r2']:.4f}")
    
    # Save the better model as best overall
    if mae_tp <= mae_rs:
        print(f"\nTPE model performs better (lower MAE). Checking if it should be saved as best overall...")
        save_best_overall_model(
            model=best_model_tp,
            model_name='TPE',
            mae=metrics['tp']['mae'],
            rmse=metrics['tp']['rmse'],
            r2=metrics['tp']['r2'],
            mape=metrics['tp']['mape'],
            max_error=metrics['tp']['max_error'],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            params=params_tpe
        )
    else:
        print(f"\nRS model performs better (lower MAE). Checking if it should be saved as best overall...")
        save_best_overall_model(
            model=best_model_rs,
            model_name='RS',
            mae=metrics['rs']['mae'],
            rmse=metrics['rs']['rmse'],
            r2=metrics['rs']['r2'],
            mape=metrics['rs']['mape'],
            max_error=metrics['rs']['max_error'],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            params=params_rs
        )


# === Run Optuna Optimization ===
def run_optimization(sampler, pruner, csv_path=str(BASE_DIR / 'data/DATA_ABS_&_PP_Binary.csv'), n_trials=100, n_startup_trials=10):
    global best_mae_RS_global
    global best_model_RS_global
    global best_params_RS_global
    
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(lambda trial: objective(trial, csv_path=csv_path, n_startup_trials=n_startup_trials, sampler=sampler.__class__.__name__), n_trials=n_trials, timeout=3600)

    # Best trial overall (hoping is TPE)
    if sampler.__class__.__name__ == "TPESampler":
        print("\n=== Best model TPE - after initial {} RS and {} TPE trials ===".format(n_startup_trials, (n_trials - n_startup_trials)))
    elif sampler.__class__.__name__ == "RandomSampler":
        print("\n=== RandomSampler ===")
    print("Best trial:")
    trial = study.best_trial
    print(f"  MAE: {trial.value:.4f}")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    # If TPE optimizer, Best trial RS
    if (sampler.__class__.__name__ == "TPESampler") and (n_startup_trials > 0) and (best_params_RS_global is not None):
            print("\n=== Best model RS - found with initial {} RS trials ===".format(n_startup_trials))
            print(f"  MAE: {best_mae_RS_global:.4f}")
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
    csv_path_1 = str(BASE_DIR / 'data/DATA_ABS_P1_W.csv')
    csv_path_2 = str(BASE_DIR / 'data/DATA_ABS_P2_W.csv')

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

    # Remove outliers from both datasets
    df_1 = df_1[~outliers_p1].reset_index(drop=True)
    df_2 = df_2[~outliers_p2].reset_index(drop=True)

    
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
    # plt.savefig("outputs/Reg/images/mi_scores.png")
    # plt.show()
    
    # # Run HPO otpimization with RS sampler and MedianPruner
    # print(f"\nStarting RS optimization...\n")
    # sampler = optuna.samplers.RandomSampler(seed=42)  # Use RandomSampler for simplicity
    # # pruner = optuna.pruners.MedianPruner(n_warmup_steps=1, n_startup_trials=10)
    # pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=80, reduction_factor=3)
    # best_trial_rs = run_optimization(sampler, pruner, train_csv_path, n_trials=40)

    # Run HPO otpimization with TPE sampler and HyperbandPruner
    print(f"\nStarting TPE optimization...\n")
    n_startup_trials = 60
    n_trials = 120
    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials, seed=42) #(n_startup_trials=10, seed=31) # Here tried to add some startup trials
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=80, reduction_factor=3)
    best_trial_tpe = run_optimization(sampler, pruner, train_csv_path, n_trials=n_trials, n_startup_trials=n_startup_trials)

    # Retrain the best models
    train_and_save_best_model(params_tpe=best_trial_tpe.params, params_rs=best_params_RS_global, epochs=200, csv_path_train=train_csv_path, csv_path_test=test_csv_path)

    # Print total time taken
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
