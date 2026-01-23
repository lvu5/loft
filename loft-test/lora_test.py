# ===== Before =====
# layer = nn.Linear(in_features, out_features)

# ===== After ======
import loralib as lora
import torch

m = 1024
n = 512
r = 8
in_features = n
out_features = m

W = torch.zeros((out_features, in_features))
layer = lora.Linear(in_features, out_features, r=r)

import loralib as lora
# i only want a single layer model
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(SimpleModel, self).__init__()
        self.linear = lora.Linear(in_features, out_features, r=rank)

    def forward(self, x):
        return self.linear(x)
    
model = SimpleModel(in_features=n, out_features=m, rank=r)
lora.mark_only_lora_as_trainable(model)

# loss fn is W - A where A is low rank matrix = r
A = torch.randn(m, r) @ torch.randn(r, n)
def loss_fn(output):
    return ((output - A) ** 2).sum()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
# print trainable parameters
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
# print all parameters
for name, param in model.named_parameters():
    print(f"{name}: {param.requires_grad}")
    
loss_lora = []
for i in range(300):
    optimizer.zero_grad()
    # output = model(torch.zeros((m, n)))
    x = torch.eye(n)
    output = model(x)          # shape (n, m)
    output = output.T          # shape (m, n)

    loss = loss_fn(output)
    loss.backward()
    optimizer.step()
    loss_lora.append(loss.item())
    
# do full fine-tuning for comparison
model_full = SimpleModel(in_features=n, out_features=m, rank=r)
# make all parameters trainable
for param in model_full.parameters():
    param.requires_grad = True
optimizer_full = torch.optim.AdamW(model_full.parameters(), lr=1e-2)
loss_full = []
for i in range(300):
    optimizer_full.zero_grad()
    x = torch.eye(n)
    output = model_full(x)          # shape (n, m)
    output = output.T                # shape (m, n)

    loss = loss_fn(output)
    loss.backward()
    optimizer_full.step()
    loss_full.append(loss.item())
    
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
plt.yscale("log")
plt.plot(loss_lora, label="LoRA")
# plt.plot(loss_loft, label="LoFT")
plt.plot(loss_full, label="Full Fine-Tuning")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
# save the figure
plt.savefig("lora_test.png", dpi=300)
plt.show()
    
