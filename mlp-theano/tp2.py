import numpy as np
import theano
import theano.tensor as tt


x = tt.fscalar()
y = tt.fscalar()

z = x + y**2


print z