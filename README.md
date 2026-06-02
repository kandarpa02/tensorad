# TensorAutodiff

A lightweight autodiff library, designed to work as a bridge between PyTorch's simplicity and TensorFlow's explicitness!

### Example:
```python
import tensorad as ts

x = ts.tensor(6.)

with ts.Grad() as g2:
    with ts.Grad() as g1:
        y = x ** 3
        dx = g1.gradient(y, [x])
    d2x = g2.gradient(dx, [x])

print('y = ', y)
print('dx = ', dx)
print('d2x = ', d2x)

# y =  Tensor(216.)
# dx =  Tensor(108.)
# d2x =  Tensor(36.)
```
