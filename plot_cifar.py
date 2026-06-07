import torch
import matplotlib.pyplot as plt


results = torch.load("checkpoints/cifar_robust_results.pt", weights_only=False)

plt.plot(results["epsilons"], results["acc_2class"], marker="o", label="2-class")
plt.plot(results["epsilons"], results["acc_10class"], marker="o", label="10-class")
plt.xlabel("L2 perturbation radius")
plt.ylabel("robust accuracy")
plt.legend()
plt.savefig("figures/cifar_robust_repro.png")
