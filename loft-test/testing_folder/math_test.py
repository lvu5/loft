import torch

def loft_pinv_gram(V, eps=1e-6):
    r = V.shape[1]
    return torch.linalg.inv(
        V.T @ V + eps * torch.eye(r, device=V.device, dtype=V.dtype)
    )
    
# test torch.linalg.inv(X.T @ X) vs loft_pinv_gram, test on random matrices and multiple cases
def test_loft_pinv_gram():
    for _ in range(10):
        print("Test case")
        m = torch.randint(5, 20, (1,)).item()
        r = torch.randint(2, 10, (1,)).item()
        V = torch.randn(m, r)

        pinv_gram_loft = torch.linalg.pinv(V.T @ V)
        pinv_gram_torch = torch.linalg.inv(V.T @ V)

        assert torch.allclose(pinv_gram_loft, pinv_gram_torch, atol=1e-5), f"Failed for m={m}, r={r}\n,LoFT:\n{pinv_gram_loft}\nTorch:\n{pinv_gram_torch}"

print("Testing loft_pinv_gram...")
test_loft_pinv_gram()