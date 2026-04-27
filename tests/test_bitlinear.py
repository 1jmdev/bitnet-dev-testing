import torch
from torch import nn

from qwen_bitnet_qat.bitlinear import BitLinear, BitLinearConfig


def test_bitlinear_forward_backward_b158():
    torch.manual_seed(0)
    lin = nn.Linear(8, 4, bias=False)
    bit = BitLinear.from_linear(lin, BitLinearConfig(scheme="b1_58"))
    x = torch.randn(2, 3, 8, requires_grad=True)
    y = bit(x).sum()
    y.backward()
    assert bit.weight.grad is not None
    assert x.grad is not None


def test_bitlinear_forward_backward_b1125():
    torch.manual_seed(0)
    lin = nn.Linear(8, 4, bias=False)
    bit = BitLinear.from_linear(lin, BitLinearConfig(scheme="b1_125", group_size=4))
    x = torch.randn(2, 3, 8, requires_grad=True)
    y = bit(x).sum()
    y.backward()
    assert bit.weight.grad is not None
    assert x.grad is not None
