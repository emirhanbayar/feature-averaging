import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
from torch.utils.data import Dataset

class SyntheticClusterDataset(Dataset):
    def __init__(self, K, d, n, alpha, sigma, seed=7):
        self.K, self.d, self.n = K, d, n
        self.alpha, self.sigma = alpha, sigma
        rng = np.random.default_rng(seed)
        # Cluster features (Assumption 1)
        A = rng.standard_normal((d, K))
        Q, _ = np.linalg.qr(A)
        self.mean_vectors = Q.T * np.sqrt(d)  # K x d
        # Partition J_+/J_- (Assumption 2)
        self.class_labels = np.where(np.arange(K) < K // 2, 1, -1)  # first K/2 positive, last K/2 negative
        # Sample the dataset (Definition 3.1)
        j = rng.integers(0, K, size=n)  # step 1: j ~ Unif([K])
        y = self.class_labels[j]  # step 2: label by partition
        x = alpha * self.mean_vectors[j] + sigma * rng.standard_normal((n, d))  # step 3: x = a*mu_j + xi
        self.j = j  # cluster index of each sample
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def visualize_tsne(self, save_path="synthetic_dataset_tsne.png", seed=0):
        embedding = TSNE(n_components=2, perplexity=30, init="pca",
                         random_state=seed).fit_transform(self.x.numpy())
        plt.figure(figsize=(7, 6))
        label_colors = {1: "tab:red", -1: "tab:blue"}
        for label in (1, -1):
            mask = self.y.numpy() == label
            plt.scatter(embedding[mask, 0], embedding[mask, 1],
                        color=label_colors[label], s=14,
                        label=f"y={'+1' if label == 1 else '-1'}")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title(f"Multi-Cluster Data Distribution (k={self.K}, d={self.d}, n={self.n}) — t-SNE")
        plt.legend(fontsize=8, markerscale=1.5)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"saved {save_path}")

if __name__ == "__main__":
    K, d, n, alpha, sigma = 10, 3072, 1000, 1, 1
    dataset = SyntheticClusterDataset(K, d, n, alpha, sigma)
    dataset.visualize_tsne()