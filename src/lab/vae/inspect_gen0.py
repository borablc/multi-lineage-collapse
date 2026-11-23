"""First look at a trained VAE: sample grids and one metric row against reference A."""
import torch
from torchvision.utils import save_image

from lab.data import load_fmnist, preprocess, quantize, NUM_CLASSES
from lab.judge.model import extract_features, load_judge
from lab.paths import CHECKPOINT_DIR, RUNS_DIR
from lab.metrics.core import evaluate
from lab.vae.model import load_vae

N_SAMPLES = 3000  # same size as reference A, so k-NN metrics and FID are comparable
SAMPLE_SEED = 3110
GRID_SIZE = 64
GRID_SEED = 541

ARCH = "conv"  # "mlp" or "conv", see lab.vae.ARCHITECTURES


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = load_vae(CHECKPOINT_DIR / f"vae_fmnist_{ARCH}_seed0.pt", ARCH, device=device)
    data = load_fmnist(device)
    judge = load_judge(CHECKPOINT_DIR / "judge_fmnist.pt", device)
    ref = torch.load(CHECKPOINT_DIR / "reference_fmnist.pt")

    generator = torch.Generator(device=device).manual_seed(GRID_SEED)
    x_fake_grid = vae.sample(GRID_SIZE, generator=generator)
    print(f"x_fake_grid shape: {x_fake_grid.shape}")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    save_image(x_fake_grid, RUNS_DIR / f"grid_vae_{ARCH}.png", nrow=8)

    x_real_grid = preprocess(data["x_test"][:GRID_SIZE])
    print(f"x_real_grid shape: {x_real_grid.shape}")
    save_image(x_real_grid, RUNS_DIR / "grid_real.png", nrow=8)

    generator = torch.Generator(device=device).manual_seed(SAMPLE_SEED)
    x_fake = quantize(vae.sample(N_SAMPLES, generator=generator))
    feat_fake = extract_features(judge, x_fake)
    labels_fake = judge.labels(feat_fake.to(device))
    print(evaluate(ref["feat_a"], feat_fake, labels_fake))
    print(torch.bincount(labels_fake, minlength=NUM_CLASSES).tolist())

    # Reconstructions of B: same real images, only blurred by the VAE
    x_b = data["x_test"][ref["idx_b"].to(device)]
    mu, _ = vae.encode(preprocess(x_b))  # (no sampling: the best guess)
    x_recon = quantize(vae.decode(mu).sigmoid())
    feat_recon = extract_features(judge, x_recon)
    labels_recon = judge.labels(feat_recon.to(device))
    print(evaluate(ref["feat_a"], feat_recon, labels_recon))
    print(torch.bincount(labels_recon, minlength=NUM_CLASSES).tolist())


if __name__ == "__main__":
    main()
