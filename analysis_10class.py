import torch
import matplotlib.pyplot as plt

from model import MultiClassReLU


def cosine_matrix(features, weights):
    f = features / features.norm(dim=1, keepdim=True)
    w = weights / weights.norm(dim=1, keepdim=True)
    return f @ w.t()


ckpt = torch.load("checkpoints/model_10class.pt", weights_only=False)
cfg = ckpt["config"]

model = MultiClassReLU(d=cfg["d"], k=cfg["k"], h=cfg["h"])
model.load_state_dict(ckpt["state_dict"])

mu = ckpt["mean_vectors"]
# equivalent weight per sub-network: (1/h) sum over r
weights = model.W.detach().reshape(cfg["k"], cfg["h"], cfg["d"]).mean(dim=1)

M = cosine_matrix(mu, weights)

plt.imshow(M)
plt.colorbar()
plt.xlabel("neuron weight")
plt.ylabel("cluster feature")
plt.title("cos(mu_i, w_j)")
plt.savefig("figures/syn_decouple_repro.png")
