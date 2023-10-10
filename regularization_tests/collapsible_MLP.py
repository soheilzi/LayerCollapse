""" MLP module w/ dropout layer

Hacked together by / Copyright 2020 Ross Wightman
Changed by Zibakhsh Shabgahi to add the collapsible MLP
"""
from functools import partial

import torch
from torch import nn as nn

from itertools import repeat
import collections.abc


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)


class CollapsibleMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            batch_norm=False,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = nn.PReLU(num_parameters=1, init=0.1)
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = nn.BatchNorm1d(hidden_features) if batch_norm else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
    def linear_loss(self):
        return (self.act.weight - 1)**2
    
    def collapse(self):
        if self.batch_norm:
            W1 = self.fc1.weight.data
            B1 = self.fc1.bias.data
            gamma = self.norm.weight.data
            beta = self.norm.bias.data
            mean = self.norm.running_mean
            var = self.norm.running_var
            eps = self.norm.eps
            W2 = self.fc2.weight.data
            B2 = self.fc2.bias.data

            new_W = W2 @ torch.diag(gamma / torch.sqrt(var + eps)) @ W1
            new_B = W2 @ (gamma * (B1 - mean) / torch.sqrt(var + eps) + beta) + B2

            self.fc1 = nn.Linear(self.fc1.in_features, self.fc2.out_features)
            self.fc1.weight.data = new_W
            self.fc1.bias.data = new_B
            self.fc2 = nn.Identity()
            self.norm = nn.Identity()
            self.act = nn.Identity()
            self.drop1 = nn.Identity()
        else:
            W1 = self.fc1.weight.data
            B1 = self.fc1.bias.data
            W2 = self.fc2.weight.data
            B2 = self.fc2.bias.data

            new_W = W2 @ W1
            new_B = W2 @ B1 + B2

            self.fc1 = nn.Linear(self.fc1.in_features, self.fc2.out_features)
            self.fc1.weight.data = new_W
            self.fc1.bias.data = new_B
            self.fc2 = nn.Identity()
            self.act = nn.Identity()
            self.drop1 = nn.Identity()

