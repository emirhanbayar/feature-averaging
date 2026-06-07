import torch
import torch.nn.functional as F

from dataset import SyntheticClusterDataset
from model import MultiClassReLU

# Figure 2(b) config
CONFIG = dict(
    k=10, d=3072, h=1, n=1000, alpha=1, sigma=1,
    eta=0.001, sigma_w=1e-5, T=100,
    data_seed=7, model_seed=0,
)


def accuracy(out, y):
    """Fraction of points with argmax_j f_j(x) == y."""
    return (out.argmax(dim=1) == y).float().mean()


def train(model, x, y, eta, T, log_every=10):
    """Full-batch gradient descent: theta <- theta - eta * grad L_CE."""
    opt = torch.optim.SGD(model.parameters(), lr=eta)
    for t in range(1, T + 1):
        opt.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        opt.step()
        if t == 1 or t % log_every == 0:
            with torch.no_grad():
                acc = accuracy(model(x), y)
            print(f"iter {t:4d} | loss {loss.item():.4f} | acc {acc.item():.4f}")
    return model


def main(cfg=CONFIG):
    ds = SyntheticClusterDataset(cfg["k"], cfg["d"], cfg["n"],
                                 cfg["alpha"], cfg["sigma"], seed=cfg["data_seed"])
    model = MultiClassReLU(d=cfg["d"], k=cfg["k"], h=cfg["h"],
                           sigma_w=cfg["sigma_w"], seed=cfg["model_seed"])

    # labels are cluster indices in [k], not the +/-1 binary labels
    y = torch.from_numpy(ds.j).long()

    train(model, ds.x, y, eta=cfg["eta"], T=cfg["T"])

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg,
            "mean_vectors": torch.from_numpy(ds.mean_vectors).float(),
            "class_labels": torch.from_numpy(ds.class_labels),
        },
        "checkpoints/model_10class.pt",
    )
    print("saved checkpoints/model_10class.pt")
    return model, ds


if __name__ == "__main__":
    main()
