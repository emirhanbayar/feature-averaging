import torch
import matplotlib.pyplot as plt

from model import TwoLayerReLU


def cosine_matrix(features, weights):
    f = features / features.norm(dim=1, keepdim=True)
    w = weights / weights.norm(dim=1, keepdim=True)
    return f @ w.t()


ckpt = torch.load("model_2class.pt", weights_only=False)
cfg = ckpt["config"]

model = TwoLayerReLU(d=cfg["d"], m=cfg["m"])
model.load_state_dict(ckpt["state_dict"])

mu = ckpt["mean_vectors"]
weights = model.W.detach()

M = cosine_matrix(mu, weights)

plt.imshow(M)
plt.colorbar()
plt.xlabel("neuron weight")
plt.ylabel("cluster feature")
plt.title("cos(mu_i, w_j)")
plt.savefig("syn_average_repro.png")
