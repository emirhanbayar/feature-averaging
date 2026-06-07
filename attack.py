import torch
import torch.nn.functional as F
import numpy as np

from model import TwoLayerReLU, MultiClassReLU


def pgd_l2(forward_fn, x, y, epsilon, alpha, steps):
    """Standard L2 PGD; forward_fn returns a scalar score per sample, y in {-1, +1}."""
    if epsilon == 0:
        return x
    # random init inside the L2 ball of radius epsilon
    delta = torch.randn_like(x)
    delta = delta / delta.norm(dim=1, keepdim=True).clamp(min=1e-12)
    delta = (delta * epsilon * torch.rand(x.size(0), 1)).detach()
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.softplus(-y * forward_fn(x + delta)).sum()
        grad, = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            g_unit = grad / grad.norm(dim=1, keepdim=True).clamp(min=1e-12)
            delta = delta + alpha * g_unit
            # project onto the L2 ball
            d_norm = delta.norm(dim=1, keepdim=True).clamp(min=1e-12)
            delta = delta * (epsilon / d_norm).clamp(max=1.0)
            delta = delta.detach()
    return (x + delta).detach()


def robust_accuracy(forward_fn, x, y, epsilon, steps=40):
    """Fraction of points still correctly classified after a PGD-L2 attack of radius epsilon."""
    alpha = epsilon * 2.5 / steps if epsilon > 0 else 0.0
    x_adv = pgd_l2(forward_fn, x, y, epsilon, alpha, steps)
    out = forward_fn(x_adv)
    pred = torch.where(out >= 0, 1.0, -1.0)
    return (pred == y).float().mean().item()


def sample_test_set(mean_vectors, class_labels, n, alpha, sigma, seed):
    """Fresh draw from D_binary reusing the saved cluster features."""
    rng = np.random.default_rng(seed)
    k, d = mean_vectors.shape
    j = rng.integers(0, k, size=n)
    y = class_labels[j]
    x = alpha * mean_vectors[j] + sigma * rng.standard_normal((n, d))
    return torch.from_numpy(x).float(), torch.from_numpy(y).float()


ckpt_bin = torch.load("checkpoints/model_2class.pt", weights_only=False)
ckpt_multi = torch.load("checkpoints/model_10class.pt", weights_only=False)
cfg_bin = ckpt_bin["config"]
cfg_multi = ckpt_multi["config"]

m_bin = TwoLayerReLU(d=cfg_bin["d"], m=cfg_bin["m"])
m_bin.load_state_dict(ckpt_bin["state_dict"])

m_multi = MultiClassReLU(d=cfg_multi["d"], k=cfg_multi["k"], h=cfg_multi["h"])
m_multi.load_state_dict(ckpt_multi["state_dict"])

# binary readout for the multi-class model: sum logits of J+ minus J-
signs = ckpt_multi["class_labels"].float()
def f_multi(x):
    return m_multi(x) @ signs

# test set reuses the saved cluster features but with a fresh seed
mu = ckpt_bin["mean_vectors"].numpy()
labels = ckpt_bin["class_labels"].numpy()
x_test, y_test = sample_test_set(mu, labels, n=1000, alpha=1, sigma=1, seed=42)

epsilons = list(range(0, 65, 5))
acc_bin, acc_multi = [], []
for eps in epsilons:
    a1 = robust_accuracy(m_bin, x_test, y_test, eps)
    a2 = robust_accuracy(f_multi, x_test, y_test, eps)
    acc_bin.append(a1)
    acc_multi.append(a2)
    print(f"eps={eps:5.1f}  2-class={a1:.3f}  10-class={a2:.3f}")

torch.save(
    {
        "epsilons": epsilons,
        "acc_2class": acc_bin,
        "acc_10class": acc_multi,
        "test_seed": 42,
        "test_n": 1000,
        "pgd_steps": 40,
    },
    "checkpoints/robust_results.pt",
)
print("saved checkpoints/robust_results.pt")
