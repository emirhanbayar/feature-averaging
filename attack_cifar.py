import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from train_cifar import CIFARResNet18


def pgd_l2(model, x, y, epsilon, alpha, steps):
    """Standard L2 PGD (Madry et al.): maximize CE(model(x+delta), y) within an L2 ball.

    Attacks each model on its native training loss with its training-time label
    (original 10-class label for the 10-class model, binary label for the 2-class model).
    After each step, perturbed inputs are clipped to the [0, 1] image range.
    """
    if epsilon == 0:
        return x
    delta = torch.randn_like(x)
    norm = delta.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp(min=1e-12)
    delta = delta / norm
    delta = (delta * epsilon * torch.rand(x.size(0), 1, 1, 1, device=x.device)).detach()
    delta = ((x + delta).clamp(0, 1) - x).detach()      # keep x+delta in [0,1]
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.cross_entropy(model(x + delta), y, reduction="sum")
        grad, = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            g_norm = grad.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp(min=1e-12)
            delta = delta + alpha * grad / g_norm
            d_norm = delta.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp(min=1e-12)
            delta = delta * (epsilon / d_norm).clamp(max=1.0)
            delta = (x + delta).clamp(0, 1) - x          # keep x+delta in [0,1]
            delta = delta.detach()
    return (x + delta).detach()


def robust_accuracy_2class(model, loader, device, epsilon, steps=40):
    """Attack on CE(binary), evaluate binary via argmax."""
    alpha = epsilon * 2.5 / steps if epsilon > 0 else 0.0
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y_binary = (y >= 5).long()                  # first 5 -> 0, last 5 -> 1
        x_adv = pgd_l2(model, x, y_binary, epsilon, alpha, steps)
        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1)
        correct += (pred == y_binary).sum().item()
        total += y_binary.size(0)
    return correct / total


def robust_accuracy_10class(model, loader, device, epsilon, signs, steps=40):
    """Attack on CE(10-class), evaluate binary via sgn(sum_signs · logits)."""
    alpha = epsilon * 2.5 / steps if epsilon > 0 else 0.0
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = pgd_l2(model, x, y, epsilon, alpha, steps)
        with torch.no_grad():
            f_bin = model(x_adv) @ signs
        pred_binary = (f_bin >= 0).long()
        y_binary = (y >= 5).long()
        correct += (pred_binary == y_binary).sum().item()
        total += y.size(0)
    return correct / total


def make_test_loader(batch_size=128, n=1000, seed=42):
    test_set = datasets.CIFAR10("./data", train=False, download=True, transform=transforms.ToTensor())
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(test_set), size=n, replace=False).tolist()
    return DataLoader(Subset(test_set, idx), batch_size=batch_size, shuffle=False)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_2 = torch.load("checkpoints/cifar_2class.pt", weights_only=False)
    m2 = CIFARResNet18(num_classes=2).to(device)
    m2.load_state_dict(ckpt_2["state_dict"])
    m2.eval()

    ckpt_10 = torch.load("checkpoints/cifar_10class.pt", weights_only=False)
    m10 = CIFARResNet18(num_classes=10).to(device)
    m10.load_state_dict(ckpt_10["state_dict"])
    m10.eval()

    # signs[i] = -1 for the first 5 classes, +1 for the last 5
    signs = torch.tensor([-1.0]*5 + [1.0]*5, device=device)

    test_loader = make_test_loader()
    epsilons = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    acc_2, acc_10 = [], []
    for eps in epsilons:
        a1 = robust_accuracy_2class(m2, test_loader, device, eps)
        a2 = robust_accuracy_10class(m10, test_loader, device, eps, signs)
        acc_2.append(a1)
        acc_10.append(a2)
        print(f"eps={eps:4.2f}  2-class={a1:.3f}  10-class={a2:.3f}", flush=True)

    torch.save(
        {
            "epsilons": epsilons,
            "acc_2class": acc_2,
            "acc_10class": acc_10,
            "test_seed": 42,
            "test_n": 1000,
            "pgd_steps": 40,
        },
        "checkpoints/cifar_robust_results.pt",
    )
    print("saved checkpoints/cifar_robust_results.pt", flush=True)


if __name__ == "__main__":
    main()
