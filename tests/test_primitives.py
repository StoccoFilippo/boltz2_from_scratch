import torch
from boltz2.modules.primitives import Linear, Transition



def test_transition_grad(assert_grads_flow):
	x = torch.randn(2, 4)
	block = Transition(in_features=4, out_features=4, hid_features=8)
	assert_grads_flow(block, torch.randn(2,4))

	
