import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required

from transformers import RobertaTokenizer, RobertaForMaskedLM
from tensorly.tenalg import khatri_rao

def safe_pinv_gram(G, eps=1e-6):
    r = G.shape[0]
    return torch.linalg.inv(G + eps * torch.eye(r, device=G.device, dtype=G.dtype))

def loft_pinv_gram(V, eps=1e-6):
    r = V.shape[1]
    return torch.linalg.inv(
        V.T @ V + eps * torch.eye(r, device=V.device, dtype=V.dtype)
    )

class LoFTLinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        self.W0 = nn.Parameter(linear.weight.data.clone(), requires_grad=False)

        self.U = nn.Parameter(0.01 * torch.randn(self.out_features, rank))
        self.V = nn.Parameter(0.01 * torch.randn(self.in_features, rank))

        # freeze W0
        self.W0.requires_grad = False
        
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x):
        W = self.W0 + self.U @ self.V.T
        out = x @ W.T
        if self.bias is not None:
            out = out + self.bias
        return out

def rowwise_khatri_rao(A, B):
    """
    A, B: (m, r)
    returns: (m, r*r)
    """
    # (m, r, 1) ⊗ (m, 1, r) → (m, r, r)
    KR = A.unsqueeze(2) * B.unsqueeze(1)

    # flatten last two dims
    return KR.reshape(A.shape[0], -1)


class LoFTAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=required,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            # Each group must contain exactly one (U, V) pair
            if len(group["params"]) != 2:
                raise ValueError("Each LoFT param group must contain [U, V]")

            U, V = group["params"]

            if U.grad is None or V.grad is None:
                continue

            state = self.state.setdefault(U, {})
            if len(state) == 0: # initialize everything
                m = U.shape[0]
                r = U.shape[1]
                state["step"] = 0
                state["update_U"] = True
                n = V.shape[0]
                # First moments
                state["mU"] = torch.zeros_like(U)
                state["mV"] = torch.zeros_like(V)
                state["mU_prev"] = torch.zeros_like(U)
                state["mV_prev"] = torch.zeros_like(V)
                # second moments stuffs
                state["pU_prev"] = torch.zeros(m, r*r, device=U.device, dtype=U.dtype)
                state["pV_prev"] = torch.zeros(n, r*r, device=U.device, dtype=U.dtype)
                state["pU"] = torch.zeros(m, r*r, device=U.device, dtype=U.dtype)
                state["pV"] = torch.zeros(n, r*r, device=U.device, dtype=U.dtype)
                # saving previous factors
                state["V_k_1"] = V.detach().clone()
                state["U_k_1"] = U.detach().clone()
                
            state["step"] += 1
            k = state["step"]
            # Gradients from autograd
            gU = U.grad
            gV = V.grad
            # projection matrices
            projection_U_k = U.T @ U # (U_k^T U_k) shape (r, r)
            projection_V_k = V.T @ V # (V_k^T V_k) shape (r, r)
            
            projection_U_k_inv = safe_pinv_gram(projection_U_k) # shape (r, r)
            projection_V_k_inv = safe_pinv_gram(projection_V_k) # shape (r, r)

            # Calibration matrices (PLACEHOLDERS)
            C_k_V  = (state["V_k_1"].T @ V) @ (safe_pinv_gram(projection_V_k)) # shape (r, r)
            C_k_U  = (state["U_k_1"].T @ U) @ (safe_pinv_gram(projection_U_k)) # shape (r, r)
            
            # gradient projection
            gU_tilde = gU @ projection_V_k_inv # shape (m, r)
            gV_tilde = gV @ projection_U_k_inv # shape (n, r)
            # First moment calibration
            state["mU"] = beta1 * (state["mU_prev"] @ C_k_V) + (1.0 - beta1) * gU_tilde
            state["mV"] = beta1 * (state["mV_prev"] @ C_k_U) + (1.0 - beta1) * gV_tilde
            
            
            tkr_gu = rowwise_khatri_rao(gU_tilde,  gU_tilde)
            tkr_gv = rowwise_khatri_rao(gV_tilde,  gV_tilde)
            # state pu_prev shape (m, r*r)
            # kron(C_k_V, C_k_V) shape (r*r, r*r)
            # tkr_gu shape (m, r*r)
            # Second moment calibration
            state["pU"] = beta2 * (state["pU_prev"] @ torch.kron(C_k_V, C_k_V)) + (1.0 - beta2) * tkr_gu
            state["pV"] = beta2 * (state["pV_prev"] @ torch.kron(C_k_U, C_k_U)) + (1.0 - beta2) * tkr_gv
            
            state["pV_prev"].copy_(state["pV"])
            state["mV_prev"].copy_(state["mV"])
            state["mU_prev"].copy_(state["mU"])
            state["pU_prev"].copy_(state["pU"])
            # Alternating updates
            if state["update_U"]:
                vU_tilde = state["pU"] @ (khatri_rao([V.T, V.T])) # @ (r^2, n)
                #print("vU_tilde",vU_tilde)
                mU_tilde = state["mU"] @ V.T / (1 - beta1 ** k)
                #print("mU_tilde",mU_tilde)
                vU_tilde = vU_tilde / (1 - beta2 ** k)
                vU_tilde = torch.clamp(vU_tilde, min=eps)
                #print("vU_tilde after bias",vU_tilde)
                delta_U = lr * (mU_tilde)/ torch.sqrt(vU_tilde + eps)
                #print("delta_U before proj",delta_U)
                delta_U = delta_U @ V @ projection_V_k_inv
                state["U_k_1"].copy_(U.detach())
                if wd != 0.0:
                    U.mul_(1.0 - lr * wd)
                U.sub_(delta_U)
                
            else:
                # Reconstruct projected second moment (PLACEHOLDER)
                vV = state["pV"] @ (khatri_rao([U.T, U.T])) # @ (r^2, m)
                mV_tilde = state["mV"] @ U.T / (1 - beta1 ** k)
                vV_tilde = vV / (1 - beta2 ** k)
                vV_tilde = torch.clamp(vV_tilde, min=eps)
                delta_V = lr * (mV_tilde) / torch.sqrt(vV_tilde + eps)
                delta_V = delta_V @ U @ projection_U_k_inv
                state["V_k_1"].copy_(V.detach())
                
                if wd != 0.0:
                    V.mul_(1.0 - lr * wd)
                V.sub_(delta_V)
            state["update_U"] = not state["update_U"]
        return loss



def replace_linear_with_loft(module, rank=4):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, LoFTLinear(child, rank))
        else:
            replace_linear_with_loft(child, rank)

# model = RobertaForMaskedLM.from_pretrained("roberta-base")

# replace_linear_with_loft(model, rank=4)

# for name, param in model.named_parameters():
#     if ".U" in name or ".V" in name:
#         param.requires_grad = True
#     else:
#         param.requires_grad = False

# loft_groups = []

# for module in model.modules():
#     if isinstance(module, LoFTLinear):
#         loft_groups.append({"params": [module.U, module.V]})


# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# inputs = tokenizer(
#     ["hello world", "this is a test"],
#     return_tensors="pt",
#     padding=True,
#     truncation=True
# )

# labels = inputs["input_ids"].clone()

# optimizer = LoFTAdamW(loft_groups, lr=1e-3, weight_decay=1e-2)

# model.train()

# for i in range(20):
#     optimizer.zero_grad()

#     outputs = model(**inputs, labels=labels)
#     loss = outputs.loss

#     loss.backward()
#     # for name, p in model.named_parameters():
#     #     if p.grad is not None and not torch.isfinite(p.grad).all():
#     #         print("Bad grad:", name)
#     optimizer.step()

#     print(f"Iter {i + 1} | Loss: {loss.item()}")



import loralib as lora
m, n, r = 1024, 512, 8

lr = 1e-2 
class SimpleModel(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(SimpleModel, self).__init__()
        self.linear = lora.Linear(in_features, out_features, r=rank)

    def forward(self, x):
        return self.linear(x)

A = torch.randn(m, r) @ torch.randn(r, n)

def loss_fn(output):
    return ((output - A) ** 2).sum()
steps = 50
def run_loft():

    model_loft = SimpleModel(in_features=n, out_features=m, rank=r)
    # replace linear layer with LoFTLinear
    model_loft.linear = LoFTLinear(nn.Linear(n, m), rank=r)

    # make only U and V trainable
    for name, param in model_loft.named_parameters():
        if ".U" in name or ".V" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    loft_groups = []
    for module in model_loft.modules():
        if isinstance(module, LoFTLinear):
            loft_groups.append({"params": [module.U, module.V]})

    optimizer_loft = LoFTAdamW(loft_groups, lr=lr)
    for name, param in model_loft.named_parameters():
        print(f"{name}: {param.requires_grad}")
    loss_loft = []
    for i in range(steps):
        optimizer_loft.zero_grad()
        x = torch.eye(n)
        output = model_loft(x)          # shape (n, m)
        output = output.T                # shape (m, n)
        loss = loss_fn(output)
        print(f"Iter {i + 1} | Loss: {loss.item()}")
        loss.backward()
        optimizer_loft.step()
        loss_loft.append(loss.item())
    return loss_loft


import matplotlib.pyplot as plt
loss_loft = run_loft()
plt.figure(figsize=(6, 4))
# plt.yscale("log")
plt.plot(loss_loft, label="LoFT")

plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
# save the figure
plt.savefig("loft_test.png")
plt.show()
