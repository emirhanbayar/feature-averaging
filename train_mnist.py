import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from resnet import ResNet18

# Figure 3 (middle) config
CONFIG = dict(
    batch_size=128, epochs=10, lr=0.1, momentum=0.9, weight_decay=5e-4, seed=0,
)


def get_loaders(batch_size):
    tfm = transforms.ToTensor()
    train_set = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    test_set = datasets.MNIST("./data", train=False, download=True, transform=tfm)
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
    """Train ResNet18 on MNIST with the given (label_fn, num_classes) head."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])
    train_loader, test_loader = get_loaders(cfg["batch_size"])
    model = ResNet18(in_channels=1, num_classes=num_classes).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"],
                          momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
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
        print(f"[{num_classes}-class] epoch {epoch:2d}/{cfg['epochs']} | test acc {acc:.4f}")

    torch.save(
        {"state_dict": model.state_dict(), "config": cfg, "num_classes": num_classes},
        ckpt_path,
    )
    print(f"saved {ckpt_path}")


def main():
    # 10-class: normal MNIST classification
    train_one(10, label_fn=lambda y: y, ckpt_path="checkpoints/mnist_10class.pt")
    # 2-class: parity (odd -> 1, even -> 0)
    train_one(2, label_fn=lambda y: y % 2, ckpt_path="checkpoints/mnist_2class.pt")


if __name__ == "__main__":
    main()
