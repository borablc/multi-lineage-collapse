"""One-time training script for the judge network (Fashion-MNIST)."""
import random
import subprocess
import json
from datetime import datetime

import torch
from torch import nn
import numpy as np

from lab.data import load_fmnist, preprocess
from lab.judge import JudgeCNN, load_judge
from lab.paths import CHECKPOINT_DIR, PROJECT_ROOT

SEED = 42
EPOCHS = 15
BATCH_SIZE = 256
LR = 1e-3


def set_seed(seed: int) -> None:
    """Seed every random number generator we use, for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_info() -> dict:
    """Returns the current commit hash and whether there are uncommitted changes."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"],
                            capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"],
                           capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip() != ""
    return {"commit": commit, "dirty": dirty}


@torch.no_grad()
def evaluate(model: JudgeCNN, x_uint8: torch.Tensor, y: torch.Tensor, batch_size: int = 1000) -> float:
    """Returns classification accuracy in [0, 1]."""
    model.eval()
    correct = 0
    for i in range(0, len(x_uint8), batch_size):
        xb = x_uint8[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(preprocess(xb))
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
    return correct / len(x_uint8)


def train_one_epoch(model: JudgeCNN, optimizer: torch.optim.Optimizer,
                    x_uint8: torch.Tensor, y: torch.Tensor, batch_size: int) -> float:
    """Runs one pass over the training set in random order. Returns the mean loss."""
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    perm = torch.randperm(len(x_uint8), device=x_uint8.device)  # new random order every epoch
    total_loss = 0.0
    for i in range(0, len(x_uint8), batch_size):
        idx = perm[i:i + batch_size]
        xb = preprocess(x_uint8[idx])
        yb = y[idx]
        logits = model(xb)
        loss = loss_fn(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(idx)
    return total_loss / len(x_uint8)


def main() -> None:
    set_seed(SEED)
    device = "cuda"
    d = load_fmnist(device=device)
    model = JudgeCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, optimizer, d["x_train"], d["y_train"], BATCH_SIZE)
        scheduler.step()
        train_acc = evaluate(model, d["x_train"], d["y_train"])
        print(f"epoch {epoch + 1:2d}  loss {loss:.4f}  train acc {train_acc:.4f}")

    test_acc = evaluate(model, d["x_test"], d["y_test"])
    print(f"TEST ACCURACY: {test_acc:.4f}")
    # --- save ---
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "judge_fmnist.pt"
    torch.save(model.state_dict(), ckpt_path)

    meta = {
        "dataset": "fashion-mnist",
        "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
        "train_acc": train_acc, "test_acc": test_acc,
        "git": git_info(),
        "torch_version": torch.__version__,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    with open(CHECKPOINT_DIR / "judge_fmnist.json", "w") as f:
        json.dump(meta, f, indent=2)
    if meta["git"]["dirty"]:
        print("WARNING: the code has uncommitted changes")

    # --- sanity check: reload and re-evaluate ---
    reloaded = load_judge(ckpt_path, device)
    print("reloaded:", evaluate(reloaded, d["x_test"], d["y_test"]))


if __name__ == "__main__":
    main()