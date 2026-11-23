import random
import subprocess

import numpy as np
import torch

from lab.paths import PROJECT_ROOT


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


def check(condition: bool, description: str) -> None:
    """Assert a sanity condition and report it by name."""
    assert condition, f"FAILED: {description}"
    print(f"[PASSED] {description}")
