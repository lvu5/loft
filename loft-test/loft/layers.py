import math
import torch
import torch.nn as nn

# self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
# self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
# nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
# nn.init.zeros_(self.lora_B)

class LoFTLinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        self.W0 = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        # same initialization values as Lore
        self.U = nn.Parameter(torch.zeros((self.out_features, rank)))
        self.V = nn.Parameter(torch.zeros((self.in_features, rank)))
        nn.init.zeros_(self.U)
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        
        # nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        # nn.init.zeros_(self.V)


        # freeze W0
        self.W0.requires_grad = False
        
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

    def forward(self, x):
        # W = self.W0 + self.U @ self.V.T
        # out = x @ W.T
        delta = (self.U @ self.V.T) # * self.scaling
        out = x @ (self.W0 + delta).T
        if self.bias is not None:
            out = out + self.bias
        return out
