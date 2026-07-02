import pytest
import torch


@pytest.fixture
def assert_grads_flow():
    def _check(module, *inputs):
        out = module(*inputs)
        out.sum().backward()
        for name, p in module.named_parameters():
            assert p.grad is not None, f"no grad for {name}"
            assert not torch.isnan(p.grad).any(), f"nan grad in {name}"
        return out
    return _check
 
	
