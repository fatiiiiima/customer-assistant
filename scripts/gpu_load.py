"""Keeps the Mac's GPU busy (back-to-back 4096 matrix products in MLX) until killed: the worst case of the language model writing a reply."""
import mlx.core as mx
a = mx.random.normal((4096, 4096)); b = mx.random.normal((4096, 4096))
while True:
    mx.eval(a @ b)
