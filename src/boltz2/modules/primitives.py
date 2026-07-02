"""Shared building blocks: Linear (with named inits), LayerNorm, Attention, Transition.

Write these once, well. If you find yourself writing multi-head attention a
second time, generalize the first one instead.
"""

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

# here the def forward is not necessary as we are importing from nn.Module that already dose it by default.
