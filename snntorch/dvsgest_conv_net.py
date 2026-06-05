#!/usr/bin/env python3
#%%
"""
Convolutional SNN for DVSGesture (32x32) using snnTorch + Tonic.
Usage:
  python conv_snn_dvsgesture.py --mode train   # train with default params
  python conv_snn_dvsgesture.py --mode optuna  # hyperparameter search
"""

import os
import json
import argparse
import shutil
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
import tonic
import tonic.transforms as transforms
from tonic import DiskCachedDataset
import snntorch as snn
from snntorch import surrogate
import optuna
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPATIAL_FACTOR = 0.25          # 128 -> 32
SENSOR_SIZE    = (32, 32, 2)   # (W, H, polarities) after downsampling
N_TIME_BINS    = 100
N_CLASSES      = 11
DATA_DIR       = "./data"
CACHE_DIR      = f"./data/cache/{SENSOR_SIZE[0]}x{SENSOR_SIZE[1]}_t{N_TIME_BINS}"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------
def save_run(model, params: dict, acc: float, tag: str = "") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{tag}" if tag else ts
    out_dir = os.path.join("results", name)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "weights.pth"))
    meta = {"accuracy": acc, "params": params}
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump(meta, f, indent=2)
    shutil.copy(__file__, os.path.join(out_dir, "script.py"))
    print(f"Saved run to {out_dir}/")
    return out_dir


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def get_dataloaders(batch_size: int = 32, num_workers: int = 4):

    # Figshare's bot-protection blocks the default URL; the backend S3 endpoint works fine.
    tonic.datasets.DVSGesture.train_url = "https://ndownloader.figshare.com/files/38022171"
    tonic.datasets.DVSGesture.test_url  = "https://ndownloader.figshare.com/files/38020584"

    transform = transforms.Compose([
        transforms.Downsample(spatial_factor=SPATIAL_FACTOR),
        transforms.ToFrame(sensor_size=SENSOR_SIZE, n_time_bins=N_TIME_BINS),
    ])

    train_ds = tonic.datasets.DVSGesture(save_to=DATA_DIR, train=True,  transform=transform)
    test_ds  = tonic.datasets.DVSGesture(save_to=DATA_DIR, train=False, transform=transform)
    cached_train = DiskCachedDataset(train_ds, cache_path=os.path.join(CACHE_DIR, "train"))
    cached_test = DiskCachedDataset(test_ds,  cache_path=os.path.join(CACHE_DIR, "test"))

    train_loader = DataLoader(
        cached_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        cached_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ConvSNN(nn.Module):
    """
    Three conv blocks (each conv -> BN -> AvgPool -> LIF) followed by two
    fully-connected LIF layers.  Spike counts across all time steps are
    summed and used as logits.

    Input:  (B, T, 2, H, W)     T = N_TIME_BINS, H/W from SENSOR_SIZE
    Output: (B, N_CLASSES)      unnormalized spike counts
    """

    def __init__(
        self,
        beta: float        = 0.9,
        threshold: float   = 1.0,
        n_filters_1: int   = 32,
        n_filters_2: int   = 64,
        n_filters_3: int   = 128,
        fc_size: int       = 512,
        dropout: float     = 0.4,
    ):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # --- conv block 1:  32x32 -> 16x16 ---
        self.conv1 = nn.Conv2d(2, n_filters_1, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(n_filters_1)
        self.pool1 = nn.AvgPool2d(2)
        self.lif1  = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)

        # --- conv block 2:  16x16 -> 8x8 ---
        self.conv2 = nn.Conv2d(n_filters_1, n_filters_2, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(n_filters_2)
        self.pool2 = nn.AvgPool2d(2)
        self.lif2  = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)

        # --- conv block 3:  8x8 -> 4x4 ---
        self.conv3 = nn.Conv2d(n_filters_2, n_filters_3, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(n_filters_3)
        self.pool3 = nn.AvgPool2d(2)
        self.lif3  = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)

        h_out = SENSOR_SIZE[1] // (2 ** 3)
        w_out = SENSOR_SIZE[0] // (2 ** 3)
        fc_in = n_filters_3 * h_out * w_out

        # --- FC block 1 ---
        self.fc1     = nn.Linear(fc_in, fc_size)
        self.drop    = nn.Dropout(p=dropout)
        self.lif4    = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)

        # --- FC block 2 (output) ---
        self.fc2  = nn.Linear(fc_size, N_CLASSES)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 2, 32, 32)
        B, T = x.shape[0], x.shape[1]

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()

        spk_acc = torch.zeros(B, N_CLASSES, device=x.device)

        for t in range(T):
            xt = x[:, t].float()           # (B, 2, 32, 32)

            # block 1
            c1 = self.pool1(self.bn1(self.conv1(xt)))
            spk1, mem1 = self.lif1(c1, mem1)

            # block 2
            c2 = self.pool2(self.bn2(self.conv2(spk1)))
            spk2, mem2 = self.lif2(c2, mem2)

            # block 3
            c3 = self.pool3(self.bn3(self.conv3(spk2)))
            spk3, mem3 = self.lif3(c3, mem3)

            # FC 1
            h = self.drop(self.fc1(spk3.view(B, -1)))
            spk4, mem4 = self.lif4(h, mem4)

            # FC 2 (output)
            spk5, mem5 = self.lif5(self.fc2(spk4), mem5)
            spk_acc += spk5

        return spk_acc  # spike counts  -> cross-entropy logits


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = correct = total = 0
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        out  = model(data)
        loss = criterion(out, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(1) == targets).sum().item()
        total      += targets.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        out  = model(data)
        loss = criterion(out, targets)
        total_loss += loss.item()
        correct    += (out.argmax(1) == targets).sum().item()
        total      += targets.size(0)
    return total_loss / len(loader), correct / total


def train_model(params: dict, train_loader, test_loader, epochs: int = 30,
                device=DEVICE, verbose: bool = True):
    model = ConvSNN(
        beta        = params["beta"],
        threshold   = params["threshold"],
        n_filters_1 = params["n_filters_1"],
        n_filters_2 = params["n_filters_2"],
        n_filters_3 = params["n_filters_3"],
        fc_size     = params["fc_size"],
        dropout     = params.get("dropout", 0.4),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = params["lr"],
        weight_decay = params.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = eval_epoch(model, test_loader, criterion, device)
        scheduler.step()
        if verbose:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train loss {tr_loss:.4f}  acc {tr_acc:.4f} | "
                f"val loss {va_loss:.4f}  acc {va_acc:.4f}"
            )
        best_acc = max(best_acc, va_acc)

    return best_acc, model


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = dict(
    lr          = 1e-3,
    beta        = 0.9,
    threshold   = 1.0,
    n_filters_1 = 32,
    n_filters_2 = 64,
    n_filters_3 = 128,
    fc_size     = 512,
    dropout     = 0.4,
    weight_decay= 1e-4,
)

# Best params from first Optuna search (82.2 % val acc, 20 trials × 10 epochs)
BEST_PARAMS = dict(
    lr          = 0.0011206853479171495,
    beta        = 0.7508406817245454,
    threshold   = 1.177052397178093,
    n_filters_1 = 32,
    n_filters_2 = 64,
    n_filters_3 = 128,
    fc_size     = 256,
    dropout     = 0.3888123311378158,
    weight_decay= 0.00018324963565002856,
)

PARAM_SETS = {"default": DEFAULT_PARAMS, "best": BEST_PARAMS}


def objective(trial, train_loader, test_loader, epochs: int = 8):
    params = dict(
        lr          = trial.suggest_float("lr",          1e-4,  5e-3, log=True),
        beta        = trial.suggest_float("beta",        0.75,  0.98),
        threshold   = trial.suggest_float("threshold",   0.5,   2.0),
        n_filters_1 = trial.suggest_categorical("n_filters_1", [16, 32, 64]),
        n_filters_2 = trial.suggest_categorical("n_filters_2", [32, 64, 128]),
        n_filters_3 = trial.suggest_categorical("n_filters_3", [64, 128, 256]),
        fc_size     = trial.suggest_categorical("fc_size",     [256, 512, 1024]),
        dropout     = trial.suggest_float("dropout",     0.2,   0.6),
        weight_decay= trial.suggest_float("weight_decay",1e-5,  1e-3, log=True),
    )
    acc, _ = train_model(params, train_loader, test_loader,
                         epochs=epochs, verbose=False)
    return acc


def run_optuna(n_trials: int = 20, search_epochs: int = 10,
               final_epochs: int = 30, batch_size: int = 32):
    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    study = optuna.create_study(
        direction  = "maximize",
        study_name = "conv_snn_dvsgesture",
        storage    = "sqlite:///optuna_snn.db",
        load_if_exists = True,
    )
    study.optimize(
        partial(objective, train_loader=train_loader,
                test_loader=test_loader, epochs=search_epochs),
        n_trials   = n_trials,
        show_progress_bar = True,
    )

    print("\n=== Optuna results ===")
    print(f"Best validation accuracy : {study.best_value:.4f}")
    print(f"Best params              : {study.best_params}")

    print("\nRetraining best config for full epochs …")
    best_acc, model = train_model(
        study.best_params, train_loader, test_loader,
        epochs=final_epochs, verbose=True,
    )
    save_run(model, study.best_params, best_acc, tag="optuna_best")
    print(f"\nFinal accuracy: {best_acc:.4f}")
    return study

#%%
get_dataloaders()  # test dataset loading and caching

#%%
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ConvSNN on DVSGesture")
    parser.add_argument(
        "--mode", choices=["train", "optuna"], default="train",
        help="'train' uses DEFAULT_PARAMS; 'optuna' runs hyperparameter search",
    )
    parser.add_argument("--epochs",       type=int, default=50, help="Training epochs (train mode)")
    parser.add_argument("--batch-size",   type=int, default=32)
    parser.add_argument("--n-trials",     type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--search-epochs",type=int, default=20, help="Epochs per Optuna trial (keep low for speed)")
    parser.add_argument("--params",       choices=["default", "best"], default="best",
                        help="Parameter set for train mode: 'default' or 'best' (Optuna result)")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    if args.mode == "optuna":
        run_optuna(
            n_trials     = args.n_trials,
            search_epochs= args.search_epochs,
            final_epochs = args.epochs,
            batch_size   = args.batch_size,
        )
    else:
        params = PARAM_SETS[args.params]
        print(f"Using param set: {args.params}")
        train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)
        best_acc, model = train_model(
            params, train_loader, test_loader,
            epochs=args.epochs, verbose=True,
        )
        save_run(model, params, best_acc, tag=args.params)
        print(f"\nBest accuracy: {best_acc:.4f}")

