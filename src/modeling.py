import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


SEQUENCE_LENGTH = 60
HORIZONS        = [1, 5, 10]
HIDDEN_SIZE     = 128
NUM_LAYERS      = 2
DROPOUT         = 0.2
BATCH_SIZE      = 32
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4
MAX_EPOCHS      = 200
PATIENCE        = 20
SCHED_PATIENCE  = 8
HORIZON_WEIGHTS = torch.tensor([0.2, 0.6, 0.2], dtype=torch.float32)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VIXDataset(Dataset):
    def __init__(self, dataframe, feature_cols, target_col='vix_log',
                 sequence_length=60, horizons=None):
        if horizons is None:
            horizons = [5]
        self.seq_len  = sequence_length
        self.horizons = horizons
        self.max_h    = max(horizons)
        self.X        = dataframe[feature_cols].values.astype(np.float32)
        self.target   = dataframe[target_col].values.astype(np.float32)
        self.indices  = list(range(sequence_length, len(dataframe) - self.max_h))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        x   = self.X[idx - self.seq_len : idx]
        y   = np.array([self.target[idx + h] for h in self.horizons], dtype=np.float32)
        return torch.tensor(x), torch.tensor(y), idx


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn    = nn.Linear(hidden_size, hidden_size)
        self.context = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_output):
        energy  = self.context(torch.tanh(self.attn(lstm_output)))
        weights = torch.softmax(energy, dim=1)
        context = (weights * lstm_output).sum(dim=1)
        return context, weights.squeeze(-1)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 dropout=0.2, n_horizons=3):
        super().__init__()
        self.lstm      = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = Attention(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size, n_horizons)

    def forward(self, x):
        lstm_out, _     = self.lstm(x)
        context, attn_w = self.attention(lstm_out)
        out             = self.fc(self.dropout(context))
        return out, attn_w


def build_dataloaders(train_df, val_df, test_df, feature_cols,
                      sequence_length=SEQUENCE_LENGTH,
                      horizons=HORIZONS,
                      batch_size=BATCH_SIZE):
    train_ds = VIXDataset(train_df, feature_cols, horizons=horizons, sequence_length=sequence_length)
    val_ds   = VIXDataset(val_df,   feature_cols, horizons=horizons, sequence_length=sequence_length)
    test_ds  = VIXDataset(test_df,  feature_cols, horizons=horizons, sequence_length=sequence_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


def weighted_mse_loss(preds, targets, weights=None):
    if weights is None:
        weights = HORIZON_WEIGHTS
    hw        = weights.to(preds.device)
    sq_errors = (preds - targets) ** 2
    return (sq_errors * hw.unsqueeze(0)).mean()


def compute_metrics(preds_vix, true_vix, current_vix):
    mae     = float(np.mean(np.abs(preds_vix - true_vix)))
    rmse    = float(np.sqrt(np.mean((preds_vix - true_vix) ** 2)))
    mape    = float(np.mean(np.abs((preds_vix - true_vix) / true_vix)) * 100)
    dir_acc = float(np.mean(
        np.sign(preds_vix - current_vix) == np.sign(true_vix - current_vix)
    ) * 100)
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'dir_acc': dir_acc}


def run_epoch(model, loader, optimizer=None, training=True):
    model.train() if training else model.eval()
    total_loss  = 0.0
    total_gnorm = 0.0
    n_batches   = 0

    with torch.set_grad_enabled(training):
        for xb, yb, _ in loader:
            xb, yb   = xb.to(DEVICE), yb.to(DEVICE)
            preds, _ = model(xb)
            loss     = weighted_mse_loss(preds, yb)

            if training and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                gnorm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                total_gnorm += gnorm.item()
                optimizer.step()

            total_loss += loss.item() * len(xb)
            n_batches  += 1

    mean_loss  = total_loss / len(loader.dataset)
    mean_gnorm = total_gnorm / n_batches if training else 0.0
    return mean_loss, mean_gnorm


def train_model(model, train_loader, val_loader,
                max_epochs=MAX_EPOCHS,
                patience=PATIENCE,
                sched_patience=SCHED_PATIENCE,
                save_path='models/lstm_model.pt',
                config_path='models/lstm_config.json',
                extra_config=None):

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=sched_patience
    )

    train_losses, val_losses, grad_norms = [], [], []
    best_val_loss    = float('inf')
    best_epoch       = 0
    patience_counter = 0
    best_state       = None

    for epoch in range(1, max_epochs + 1):
        tr_loss, gnorm = run_epoch(model, train_loader, optimizer, training=True)
        val_loss, _    = run_epoch(model, val_loader,   training=False)
        scheduler.step(val_loss)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        grad_norms.append(gnorm)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_epoch       = epoch
            patience_counter = 0
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch:>3} | Train: {tr_loss:.5f} | Val: {val_loss:.5f} | '
                  f'Best: {best_val_loss:.5f} @ ep {best_epoch} | '
                  f'LR: {lr_now:.2e} | GNorm: {gnorm:.4f} | Pat: {patience_counter}/{patience}')

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    model.load_state_dict(best_state)
    model.eval()

    gap = val_losses[best_epoch - 1] - train_losses[best_epoch - 1]
    if best_epoch < 15:
        diagnosis = 'UNDERFITTING'
    elif gap > 0.06:
        diagnosis = 'OVERFITTING'
    elif gap > 0.03:
        diagnosis = 'MILD OVERFITTING'
    else:
        diagnosis = 'GOOD FIT'

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    torch.save(model.state_dict(), save_path)

    config = {
        'input_size'      : next(model.parameters()).shape[-1] if extra_config is None else extra_config.get('input_size'),
        'hidden_size'     : model.fc.in_features,
        'num_layers'      : model.lstm.num_layers,
        'dropout'         : DROPOUT,
        'n_horizons'      : model.fc.out_features,
        'horizons'        : HORIZONS,
        'horizon_weights' : HORIZON_WEIGHTS.tolist(),
        'sequence_length' : SEQUENCE_LENGTH,
        'best_epoch'      : best_epoch,
        'best_val_loss'   : best_val_loss,
        'total_epochs'    : len(train_losses),
        'mean_grad_norm'  : float(np.mean(grad_norms)),
        'diagnosis'       : diagnosis,
        **(extra_config or {})
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f'\nBest epoch    : {best_epoch}')
    print(f'Best val loss : {best_val_loss:.6f}')
    print(f'Diagnosis     : {diagnosis}')
    print(f'Saved         : {save_path}')

    return model, train_losses, val_losses, grad_norms, config


def load_model(config_path='models/lstm_config.json',
               weights_path='models/lstm_model.pt'):
    with open(config_path) as f:
        config = json.load(f)

    model = LSTMModel(
        input_size  = config['input_size'],
        hidden_size = config['hidden_size'],
        num_layers  = config['num_layers'],
        dropout     = config['dropout'],
        n_horizons  = config['n_horizons']
    ).to(DEVICE)

    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model, config


def predict(model, loader):
    model.eval()
    all_preds, all_targets, all_attn = [], [], []

    with torch.no_grad():
        for xb, yb, _ in loader:
            xb          = xb.to(DEVICE)
            out, attn_w = model(xb)
            all_preds.append(out.cpu().numpy())
            all_targets.append(yb.numpy())
            all_attn.append(attn_w.cpu().numpy())

    preds_log    = np.vstack(all_preds)
    targets_log  = np.vstack(all_targets)
    attn_weights = np.vstack(all_attn)

    return np.exp(preds_log), np.exp(targets_log), attn_weights
