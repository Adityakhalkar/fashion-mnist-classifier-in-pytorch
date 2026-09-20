"""
Fashion-MNIST Classifier in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_fashion_mnist
import os
import gzip
import tempfile
import urllib.request
import numpy as np
import torch

def load_fashion_mnist(n_train=10000, n_test=2000):
    base = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    names = {"xtr": "train-images-idx3-ubyte.gz", "ytr": "train-labels-idx1-ubyte.gz",
             "xte": "t10k-images-idx3-ubyte.gz", "yte": "t10k-labels-idx1-ubyte.gz"}
    raw = {}
    for key, name in names.items():
        path = os.path.join(tempfile.gettempdir(), "fashion_" + name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(base + name, path)
        with gzip.open(path, "rb") as f:
            buf = f.read()
        if "images" in name:
            raw[key] = np.frombuffer(buf, dtype=np.uint8, offset=16).reshape(-1, 28, 28)
        else:
            raw[key] = np.frombuffer(buf, dtype=np.uint8, offset=8)
    to_x = lambda a, n: torch.tensor(a[:n], dtype=torch.float32) / 255.0
    to_y = lambda a, n: torch.tensor(a[:n].astype(np.int64), dtype=torch.int64)
    return {"X_train": to_x(raw["xtr"], n_train), "y_train": to_y(raw["ytr"], n_train),
            "X_test": to_x(raw["xte"], n_test), "y_test": to_y(raw["yte"], n_test)}

# Step 2 - FashionDataset
from torch.utils.data import Dataset
class FashionDataset(Dataset):
    def __init__(self, X, y, mean=0.2860, std=0.3530):
        self.X = X
        self.y = y
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (self.X[i] - self.mean) / self.std, self.y[i]

# Step 3 - make_loaders
import torch
from torch.utils.data import DataLoader
def make_loaders(data, batch_size=64, val_size=2000, seed=42):
    n = len(data["X_train"]) - val_size
    train_ds = FashionDataset(data["X_train"][:n], data["y_train"][:n])
    val_ds = FashionDataset(data["X_train"][n:], data["y_train"][n:])
    test_ds = FashionDataset(data["X_test"], data["y_test"])
    g = torch.Generator().manual_seed(seed)
    return {"train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g),
            "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
            "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
            "sizes": (len(train_ds), len(val_ds), len(test_ds))}

# Step 4 - MLP
import torch.nn as nn
import torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self, hidden1=300, hidden2=100, n_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, n_classes)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

# Step 5 - train_one_epoch
def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return total / len(loader)

# Step 6 - evaluate
import torch
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            total_loss += float(loss_fn(logits, yb).item()) * len(yb)
            correct += int((logits.argmax(1) == yb).sum().item())
            n += len(yb)
    return total_loss / n, correct / n

# Step 7 - fit (not yet solved)
# TODO: implement

# Step 8 - lr_range_test (not yet solved)
# TODO: implement

# Step 9 - random_search (not yet solved)
# TODO: implement

# Step 10 - test_accuracy (not yet solved)
# TODO: implement

# Step 11 - save_model (not yet solved)
# TODO: implement

# Step 12 - predict_classes (not yet solved)
# TODO: implement

