import torch
import torch.nn.functional as F

from dataset import SyntheticClusterDataset
from model import TwoLayerReLU

# Figure 2(a) config
CONFIG = dict(
    k=10, d=3072, m=5, n=1000, alpha=1, sigma=1,
    eta=0.001, sigma_w=1e-5, sigma_b=1e-5, T=100,
    data_seed=7, model_seed=0,
)


def logistic_loss(out, y):
    """ell(q) = log(1 + e^{-q}) with q = y * f(x); softplus(-q) computes it stably."""
    return F.softplus(-y * out).mean()


def clean_accuracy(out, y):
    """Fraction of points with sgn(f(x)) == y (sgn convention: f(x) >= 0 -> +1)."""
    pred = torch.where(out >= 0, 1.0, -1.0)
    return (pred == y).float().mean()


def train(model, x, y, eta, T, log_every=10):
    """Full-batch gradient descent: theta <- theta - eta * grad L."""
    # SGD with the whole dataset as one batch and momentum=0 is exactly GD.
    opt = torch.optim.SGD(model.parameters(), lr=eta)
    for t in range(1, T + 1):
        opt.zero_grad()
        out = model(x)
        loss = logistic_loss(out, y)
        loss.backward()
        opt.step()
        if t == 1 or t % log_every == 0:
            with torch.no_grad():
                acc = clean_accuracy(model(x), y)
            print(f"iter {t:4d} | loss {loss.item():.4f} | clean acc {acc.item():.4f}")
    return model


def main(cfg=CONFIG):
    ds = SyntheticClusterDataset(cfg["k"], cfg["d"], cfg["n"],
                                 cfg["alpha"], cfg["sigma"], seed=cfg["data_seed"])
    model = TwoLayerReLU(d=cfg["d"], m=cfg["m"],
                         sigma_w=cfg["sigma_w"], sigma_b=cfg["sigma_b"],
                         seed=cfg["model_seed"])

    train(model, ds.x, ds.y, eta=cfg["eta"], T=cfg["T"])

    # save everything analysis.py needs to reproduce Figure 2(a)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg,
            "mean_vectors": torch.from_numpy(ds.mean_vectors).float(),  # cluster features mu_i
            "class_labels": torch.from_numpy(ds.class_labels),          # +1/-1 per cluster
        },
        "checkpoints/model_2class.pt",
    )
    print("saved checkpoints/model_2class.pt")
    return model, ds


if __name__ == "__main__":
    main()
