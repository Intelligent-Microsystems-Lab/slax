import pytest
import slax as sl
import flax.linen as nn

class Net(nn.Module):
    @nn.compact
    def __call__(self, x):
        cur1 = nn.Dense(1)(x)
        carry,spike = nn.RNN(sl.LIF(),return_carry=True)(cur1)
        return spike,carry['Vmem']
