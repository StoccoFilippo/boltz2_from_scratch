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
