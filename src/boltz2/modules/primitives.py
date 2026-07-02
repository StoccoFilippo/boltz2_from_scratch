import torch
import math
import torch.nn as nn

class Linear(nn.Module):

    def __init__(self, in_features, out_features, init="default", bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if init == "final":
            nn.init.zeros_(self.linear.weight)
            if bias:
                nn.init.zeros_(self.linear.bias)
        elif init == "gating":
            nn.init.zeros_(self.linear.weight)
            if bias:
                nn.init.ones_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


class Transition(nn.Module):

    def __init__(self, in_features, out_features, hid_features):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.up = Linear(in_features, 2 * hid_features)
        self.act = nn.SiLU()
        self.down = Linear(hid_features, out_features, init="final")

    def forward(self, x):
        gate, value = self.up(self.norm(x)).chunk(2, dim=-1)
        return x + self.down(self.act(gate) * value)

class Attention(nn.Module):
    def __init__(self, c, n_heads=4):
        super().__init__()
        assert c % n_heads==0, "c must be divisible by n_heads"
        head_dim = c // n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q = Linear(c, c, init="default")
        self.v = Linear(c, c, init="default")
        self.k = Linear(c, c, init="default")
        self.o = Linear(c, c, init="final")

    def forward(self,x):
        q = self.q(x)
        v = self.v(x)
        k = self.k(x)

        q = q.reshape(q.shape[0], q.shape[1], self.n_heads, -1).transpose(1, 2)
        v = v.reshape(v.shape[0], v.shape[1], self.n_heads, -1).transpose(1, 2)
        k = k.reshape(k.shape[0], k.shape[1], self.n_heads, -1).transpose(1, 2)

        scores = (q @ k.transpose(-1, -2))/ (self.head_dim**0.5)
        weights = torch.softmax(scores, dim=-1)
        value = weights @ v

        return self.o(value.transpose(1,2).reshape(x.shape[0], x.shape[1], -1))
