import torch
from torch.optim.optimizer import Optimizer, required
from .utils import loft_pinv_gram, rowwise_khatri_rao, safe_pinv_gram
from tensorly.tenalg import khatri_rao

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