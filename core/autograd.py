from core.tensor import Tensor
from core.nn.linear import Linear
import numpy as np

model = Linear(10, 1)

x = Tensor(np.random.randn(64, 10), False)
y = Tensor(np.random.randn(64, 1), False)

pred = model(x)
loss = ((pred.data - y.data) ** 2).mean()

loss = Tensor(loss, True)
loss.backward()

# SGD
lr = 1e-3
model.W.data -= lr * model.W.grad
model.b.data -= lr * model.b.grad
