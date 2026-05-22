import torch
import torch.nn as nn
from torch.nn.functional import relu

class TwoLayerReLU(nn.Module):

    def __init__(self, d, m=5, sigma_w=1e-5, sigma_b=1e-5, seed=None):
        super().__init__()
        self.d, self.m = d, m
        M = 2 * m  # total hidden neurons

        gen = torch.Generator().manual_seed(seed) if seed is not None else None

        # first layer -- trainable
        self.W = nn.Parameter(torch.empty(M, d).normal_(0.0, sigma_w, generator=gen))
        self.b = nn.Parameter(torch.empty(M).normal_(0.0, sigma_b, generator=gen))

        # second layer -- fixed
        a = torch.cat([torch.full((m,), 1.0 / m), torch.full((m,), -1.0 / m)])
        self.register_buffer("a", a)

    def forward(self, x):
        """x: (batch, d) -> scalar network output f(x): (batch,)."""
        pre = x @ self.W.t() + self.b          # (batch, 2m) pre-activations
        return relu(pre) @ self.a              # (batch,)

    @torch.no_grad()
    def predict(self, x):
        """Binary prediction sgn(f(x)) in {-1, +1} (clean-accuracy convention)."""
        out = self.forward(x)
        return torch.where(out >= 0, 1.0, -1.0)

    @property
    def w_pos(self):
        """Positive-neuron weights, shape (m, d)."""
        return self.W[: self.m]

    @property
    def w_neg(self):
        """Negative-neuron weights, shape (m, d)."""
        return self.W[self.m :]


if __name__ == "__main__":
    # smoke test against the synthetic dataset
    from dataset import SyntheticClusterDataset

    K, d, n = 10, 3072, 1000
    ds = SyntheticClusterDataset(K, d, n, alpha=1, sigma=1)
    model = TwoLayerReLU(d=d, m=5, sigma_w=1e-5, sigma_b=1e-5, seed=0)

    out = model(ds.x)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"f(x) shape: {tuple(out.shape)} | trainable params: {n_params}")
    print(f"hidden width 2m: {2 * model.m} | w_pos: {tuple(model.w_pos.shape)}")
    print(f"clean accuracy at init: {(model.predict(ds.x) == ds.y).float().mean():.3f}")
