import torch
import matplotlib.pyplot as plt

from model import TwoLayerReLU, MultiClassReLU


def cosine_matrix(features, weights):
    f = features / features.norm(dim=1, keepdim=True)
    w = weights / weights.norm(dim=1, keepdim=True)
    return f @ w.t()


def equivalent_weights_10class(state_dict, k, h, d):
    head = MultiClassReLU(d=d, k=k, h=h)
    head.load_state_dict(state_dict)
    return head.W.detach().reshape(k, h, d).mean(dim=1)


def equivalent_weights_2class(state_dict, m, d):
    """Paper groups 15 positive and 15 negative weights into 5+5 classes of 3."""
    head = TwoLayerReLU(d=d, m=m)
    head.load_state_dict(state_dict)
    W = head.W.detach()
    W_pos = W[:m].reshape(5, 3, d).mean(dim=1)
    W_neg = W[m:].reshape(5, 3, d).mean(dim=1)
    return torch.cat([W_pos, W_neg], dim=0)


ckpt = torch.load("checkpoints/cifar_heads.pt", weights_only=False)
cfg = ckpt["config"]
mu = ckpt["mu"]
d = mu.size(1)

M10_per_seed, M2_per_seed = [], []
for hd in ckpt["heads"]:
    W10 = equivalent_weights_10class(hd["head10_state_dict"], 10, cfg["h"], d)
    W2  = equivalent_weights_2class(hd["head2_state_dict"], cfg["m"], d)
    M10_per_seed.append(cosine_matrix(mu, W10))
    M2_per_seed.append(cosine_matrix(mu, W2))

# average cosine across the seeds (the paper says "average cosine value")
M10 = torch.stack(M10_per_seed).mean(dim=0)
M2  = torch.stack(M2_per_seed).mean(dim=0)


def plot_single(M, title, save_path, vmin, vmax):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels([f"$w_{{{j+1}}}$" for j in range(10)])
    ax.set_yticklabels([f"$\\mu_{{{j+1}}}$" for j in range(10)])
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"saved {save_path}")


plot_single(M2_per_seed[0],  "CIFAR-10: 2-class (seed 0 only)",  "figures/cifar_average_seed0.png",  -0.05, 0.25)
plot_single(M10_per_seed[0], "CIFAR-10: 10-class (seed 0 only)", "figures/cifar_decouple_seed0.png", -0.05, 0.65)

plot_single(M2,  "CIFAR-10: 2-class",  "figures/cifar_average_repro.png",  -0.05, 0.25)
plot_single(M10, "CIFAR-10: 10-class", "figures/cifar_decouple_repro.png", -0.05, 0.65)
