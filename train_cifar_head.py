import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import TwoLayerReLU, MultiClassReLU
from train_cifar import CIFARResNet18

# Figure 2(c, d) config -- same total head width 30 (paper section 4.1)
CONFIG = dict(
    m=15, h=3,                                # 2-class: 15+15; 10-class: 10 groups of 3
    eta=0.01, T=300,
    sigma_w=1e-2, sigma_b=1e-2,
    seeds=[0, 1, 2],
)


@torch.no_grad()
def extract_features(backbone, loader, device):
    """Forward through the pretrained ResNet18 up to (but not including) the final linear."""
    backbone.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device)
        z = (x - backbone.mean) / backbone.std
        net = backbone.net
        z = F.relu(net.bn1(net.conv1(z)))
        z = net.layer1(z)
        z = net.layer2(z)
        z = net.layer3(z)
        z = net.layer4(z)
        z = F.adaptive_avg_pool2d(z, 1).flatten(1)
        feats.append(z.cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def train_10class(head, z, y, eta, T, label_prefix):
    opt = torch.optim.SGD(head.parameters(), lr=eta)
    for t in range(1, T + 1):
        opt.zero_grad()
        loss = F.cross_entropy(head(z), y)
        loss.backward()
        opt.step()
        if t == 1 or t % 100 == 0:
            with torch.no_grad():
                acc = (head(z).argmax(dim=1) == y).float().mean()
            print(f"  [{label_prefix} 10-class] iter {t:3d}/{T}  loss {loss.item():.4f}  acc {acc.item():.4f}", flush=True)


def train_2class(head, z, y_bin, eta, T, label_prefix):
    opt = torch.optim.SGD(head.parameters(), lr=eta)
    for t in range(1, T + 1):
        opt.zero_grad()
        loss = F.softplus(-y_bin * head(z)).mean()
        loss.backward()
        opt.step()
        if t == 1 or t % 100 == 0:
            with torch.no_grad():
                pred = torch.where(head(z) >= 0, 1.0, -1.0)
                acc = (pred == y_bin).float().mean()
            print(f"  [{label_prefix}  2-class] iter {t:3d}/{T}  loss {loss.item():.4f}  acc {acc.item():.4f}", flush=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = CONFIG

    # frozen ResNet18 backbone (the 10-class CIFAR model we already trained)
    ckpt = torch.load("checkpoints/cifar_10class.pt", weights_only=False)
    backbone = CIFARResNet18(num_classes=10).to(device)
    backbone.load_state_dict(ckpt["state_dict"])

    tfm = transforms.ToTensor()
    train_set = datasets.CIFAR10("./data", train=True, download=True, transform=tfm)
    train_loader = DataLoader(train_set, batch_size=512, shuffle=False, num_workers=2)

    print("extracting features...", flush=True)
    z_train, y_train = extract_features(backbone, train_loader, device)
    print(f"  features {tuple(z_train.shape)}", flush=True)

    # mu_i = average penultimate feature per CIFAR class (train set)
    mu = torch.stack([z_train[y_train == i].mean(dim=0) for i in range(10)])

    z = z_train.to(device)
    y10 = y_train.to(device)
    y2 = torch.where(y10 < 5, 1.0, -1.0)             # first 5 -> +1, last 5 -> -1

    heads = []
    for seed in cfg["seeds"]:
        tag = f"seed={seed}"
        print(f"=== {tag} ===", flush=True)
        torch.manual_seed(seed)

        head10 = MultiClassReLU(d=z.size(1), k=10, h=cfg["h"],
                                sigma_w=cfg["sigma_w"], seed=seed).to(device)
        train_10class(head10, z, y10, cfg["eta"], cfg["T"], tag)

        head2 = TwoLayerReLU(d=z.size(1), m=cfg["m"],
                             sigma_w=cfg["sigma_w"], sigma_b=cfg["sigma_b"],
                             seed=seed).to(device)
        train_2class(head2, z, y2, cfg["eta"], cfg["T"], tag)

        heads.append({
            "seed": seed,
            "head10_state_dict": head10.state_dict(),
            "head2_state_dict":  head2.state_dict(),
        })

    torch.save(
        {"config": cfg, "mu": mu.cpu(), "heads": heads},
        "checkpoints/cifar_heads.pt",
    )
    print("saved checkpoints/cifar_heads.pt", flush=True)


if __name__ == "__main__":
    main()
