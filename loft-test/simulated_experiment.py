import torch
import math
from torch.optim import AdamW
import loralib as lora
import torch.nn as nn
import random
torch.manual_seed(0)


m, n, r = 1024, 512, 8
steps = 300
lr = 1e-5
lr_full = lr
lr_lora = lr
lr_loft = lr

class SimpleModel(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(SimpleModel, self).__init__()
        self.linear = lora.Linear(in_features, out_features, r=rank)
        # set W = 0
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class DenseModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)

A = torch.randn(m, r) @ torch.randn(r, n)

# def generate_hard_rank_r_matrix(
#     m, n, r,
#     decay=6.0,        # larger = harder
#     device=None,
#     dtype=torch.float32,
# ):
#     U, _ = torch.linalg.qr(torch.randn(m, r, device=device, dtype=dtype))
#     V, _ = torch.linalg.qr(torch.randn(n, r, device=device, dtype=dtype))

#     # exponentially decaying singular values
#     s = torch.exp(-decay * torch.arange(r, device=device, dtype=dtype) / r)
#     S = torch.diag(s)

#     A = U @ S @ V.T
#     return A, s

# m, n, r = 1024, 512, 8
# A, s = generate_hard_rank_r_matrix(m, n, r)

# def generate_lora_unfriendly_matrix(m, n, r, device=None):
#     U, _ = torch.linalg.qr(torch.randn(m, r, device=device))
#     V, _ = torch.linalg.qr(torch.randn(n, r, device=device))

#     # one dominant direction, rest tiny but nonzero
#     s = torch.tensor([1.0] + [1e-3] * (r - 1), device=device)
#     A = U @ torch.diag(s) @ V.T
#     return A, s

# A, s = generate_lora_unfriendly_matrix(1024, 512, 8)

# def generate_lora_unfriendly_matrix(m, n, r, device=None):
#     U, _ = torch.linalg.qr(torch.randn(m, r, device=device))
#     V, _ = torch.linalg.qr(torch.randn(n, r, device=device))

#     # one dominant direction, rest tiny but nonzero
#     s = torch.tensor([1.0] + [1e-3] * (r - 1), device=device)
#     A = U @ torch.diag(s) @ V.T
#     return A, s

# A, s = generate_lora_unfriendly_matrix(1024, 512, 8)



def loss_fn(output):
    #return ((output - A) ** 2).mean()
    return torch.sum((output - A) ** 2)

def run_lora():
    model = SimpleModel(in_features=n, out_features=m, rank=r)
    lora.mark_only_lora_as_trainable(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_lora)
    loss_lora = []
    # confirming trainable parameters
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.requires_grad}")
    for i in range(steps):
        optimizer.zero_grad()
        x = torch.eye(n)
        output = model(x)          # shape (n, m)
        output = output.T          # shape (m, n)
        loss = loss_fn(output)
        loss.backward()
        optimizer.step()
        loss_lora.append(loss.item())
    return loss_lora

def run_full_ft():
    model_full = SimpleModel(in_features=n, out_features=m, rank=r)
    # make all parameters trainable
    for param in model_full.parameters():
        param.requires_grad = True

    optimizer_full = torch.optim.AdamW(model_full.parameters(), lr=lr_full)
    loss_full = []
    for i in range(steps):
        optimizer_full.zero_grad()
        x = torch.eye(n)
        output = model_full(x)          # shape (n, m)
        output = output.T                # shape (m, n)
        loss = loss_fn(output)
        loss.backward()
        optimizer_full.step()
        # print(f"Step {i+1}/{steps}, Loss: {loss.item()}")
        loss_full.append(loss.item())
    return loss_full

def run_loft():
    from loft import LoFTLinear, LoFTAdamW

    model_loft = DenseModel(in_features=n, out_features=m)
    # replace linear layer with LoFTLinear
    model_loft.linear = LoFTLinear(model_loft.linear, rank=r)
    

    # make only U and V trainable
    for name, param in model_loft.named_parameters():
        print(name, param)
        if ".U" in name or ".V" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    loft_groups = []
    for module in model_loft.modules():
        if isinstance(module, LoFTLinear):
            loft_groups.append({"params": [module.U, module.V]})
    # print paremeters values
    # for name, param in model_loft.named_parameters():
    #     # print values of parameters
    #     print(f"{name}: {param.requires_grad} {param.data}")

    optimizer_loft = LoFTAdamW(loft_groups, lr=lr_loft)

    loss_loft = []
    for i in range(steps):
        optimizer_loft.zero_grad()
        x = torch.eye(n)
        output = model_loft(x)          # shape (n, m)
        output = output.T                # shape (m, n)
        loss = loss_fn(output)
        loss.backward()
        optimizer_loft.step()
        loss_loft.append(loss.item())
    return loss_loft

import matplotlib.pyplot as plt

loss_full = run_full_ft()
loss_lora = run_lora()
loss_loft = run_loft()

plt.figure(figsize=(6, 4))
# plt.yscale("log")
plt.plot(loss_full, label="Full Fine-Tuning")
plt.plot(loss_lora, label="LoRA")
plt.plot(loss_loft, label="LoFT")

# print lists to losses.txt, indicating which is which
# with open("losses.txt", "w") as f:
#     f.write("Full Fine-Tuning:\n")
#     f.write(", ".join([str(x) for x in loss_full]) + "\n")
#     f.write("LoRA:\n")
#     f.write(", ".join([str(x) for x in loss_lora]) + "\n")
#     f.write("LoFT:\n")
#     f.write(", ".join([str(x) for x in loss_loft]) + "\n")

plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
# save the figure
plt.savefig("ft_vs_lora.png")
