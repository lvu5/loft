import torch

def rowwise_khatri_rao(A, B):
    KR = A.unsqueeze(2) * B.unsqueeze(1)
    return KR.reshape(A.shape[0], -1)

def loft_pinv_gram(V, eps=1e-6):
    r = V.shape[1]
    return torch.linalg.inv(
        V.T @ V + eps * torch.eye(r, device=V.device, dtype=V.dtype)
    )
    
def safe_pinv_gram(G, eps=1e-6):
    r = G.shape[0]
    return torch.linalg.inv(G + eps * torch.eye(r, device=G.device, dtype=G.dtype))

def khatri_rao_torch(A, B):
    """
    Standard columnwise Khatri–Rao product.
    A: (I, R)
    B: (J, R)
    Returns: (I*J, R)
    """
    I, R = A.shape
    J, Rb = B.shape
    assert R == Rb

    return torch.einsum("ir,jr->ijr", A, B).reshape(I * J, R)
