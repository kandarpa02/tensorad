# TensorAutodiff

TensorAutodiff is a small educational automatic differentiation library built on top of PyTorch tensors.

The goal of this project is not to replace existing machine learning frameworks, but to explore how reverse-mode automatic differentiation works under the hood. It serves as a playground for experimenting with computational graphs, gradient propagation, higher-order derivatives, and autodiff system design.

PyTorch is used only as the tensor backend; all gradient tracking and differentiation logic is implemented by TensorAutodiff itself.

---

## Why This Project Exists

Modern machine learning frameworks provide powerful automatic differentiation systems, but much of their internal machinery remains hidden behind high-level APIs.

TensorAutodiff was created as an attempt to answer questions such as:

* How does reverse-mode autodiff actually work?
* How are computational graphs constructed?
* How are gradients propagated through a graph?
* How can higher-order derivatives be computed?
* What does a minimal autodiff engine look like?

This project is an exploration of those ideas in a compact and approachable codebase.

---

## Features

* Reverse-mode automatic differentiation
* Explicit gradient contexts
* Higher-order derivatives
* Eager execution
* PyTorch tensor backend
* Small and easy-to-read implementation

---

## Example

```python
import tensorad as ts

x = ts.tensor(6.)

with ts.Grad() as g2:
    with ts.Grad() as g1:
        y = x ** 3
        dx = g1.gradient(y, [x])

    d2x = g2.gradient(dx, [x])

print("y =", y)
print("dx =", dx)
print("d2x =", d2x)
```

Output:

```text
y = Tensor(216.)
dx = Tensor(108.)
d2x = Tensor(36.)
```

---

## Design Philosophy

TensorAutodiff prioritizes clarity over completeness.

Many implementation choices are intentionally designed to make the underlying autodiff mechanics easier to understand rather than maximizing performance or feature coverage.

If you are interested in learning how automatic differentiation systems are built, modifying an autodiff engine, or experimenting with gradient computation techniques, this project may be useful as a reference.

---

## Project Status

TensorAutodiff is an experimental and educational project under active development.

The API may change as new ideas are explored and the implementation evolves.

---

## License

Apache-2.0
