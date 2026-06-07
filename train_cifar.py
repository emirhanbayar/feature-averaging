import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from resnet import ResNet18

# Figure 3 (right) config -- recipe from coupled-diffusion/cifar (SGD-Nesterov, cosine, 100 epochs)
CONFIG = dict(
    batch_size=128, epochs=100, lr=0.1, momentum=0.9, weight_decay=5e-4, seed=0,
)


class CIFARResNet18(nn.Module):
    """ResNet18 with input normalization built in -- PGD can operate in [0, 1] image space."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.net = ResNet18(in_channels=3, num_classes=num_classes)
        # [0.5, 0.5, 0.5] mean & std: maps [0, 1] -> [-1, 1]; matches the recipe in
        # /arf/scratch/ebayar/coupled-diffusion/cifar/train_coupled.py
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def get_loaders(batch_size):
    train_tfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_tfm = transforms.ToTensor()
    train_set = datasets.CIFAR10("./data", train=True, download=True, transform=train_tfm)
    test_set = datasets.CIFAR10("./data", train=False, download=True, transform=test_tfm)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2),
    )


def evaluate(model, loader, device, label_fn):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = label_fn(y.to(device))
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total


def train_one(num_classes, label_fn, ckpt_path, cfg=CONFIG):
    """Train ResNet18 on CIFAR-10 with the given (label_fn, num_classes) head."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])
    train_loader, test_loader = get_loaders(cfg["batch_size"])
    model = CIFARResNet18(num_classes=num_classes).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"],
                          momentum=cfg["momentum"], weight_decay=cfg["weight_decay"],
                          nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    for epoch in range(1, cfg["epochs"] + 1):
        for x, y in train_loader:
            x = x.to(device)
            y = label_fn(y.to(device))
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
        scheduler.step()
        acc = evaluate(model, test_loader, device, label_fn)
        print(f"[{num_classes}-class] epoch {epoch:2d}/{cfg['epochs']} | test acc {acc:.4f}", flush=True)

    torch.save(
        {"state_dict": model.state_dict(), "config": cfg, "num_classes": num_classes},
        ckpt_path,
    )
    print(f"saved {ckpt_path}", flush=True)


def main():
    # 10-class: normal CIFAR-10 classification
    train_one(10, label_fn=lambda y: y, ckpt_path="checkpoints/cifar_10class.pt")
    # 2-class: first 5 classes vs last 5 (Experiment.tex:58)
    train_one(2, label_fn=lambda y: (y >= 5).long(), ckpt_path="checkpoints/cifar_2class.pt")


if __name__ == "__main__":
    main()
