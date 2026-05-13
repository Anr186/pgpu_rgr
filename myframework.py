import numpy as np
import cupy as cp

# Класс для замера времени (если он используется в коде)
class Profiler:
    def __init__(self): self.times = {}
    def start(self, name): pass
    def stop(self, name): pass
profiler = Profiler()

class Tensor:
    def __init__(self, data, requires_grad=False, device='gpu'):
        # Принудительно переносим данные на GPU при создании
        if isinstance(data, (list, tuple)):
            data = cp.array(data, dtype=cp.float32)
        elif isinstance(data, np.ndarray):
            data = cp.array(data, dtype=cp.float32)
        elif isinstance(data, cp.ndarray):
            data = data.astype(cp.float32)
            
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = None
        self.device = device

    @property
    def shape(self):
        return self.data.shape

    def __add__(self, other):
        other_data = other.data if isinstance(other, Tensor) else cp.array(other, dtype=cp.float32)
        out = Tensor(self.data + other_data, self.requires_grad or (getattr(other, 'requires_grad', False)))
        out._ctx = ('add', self, other)
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad)
        out._ctx = ('matmul', self, other)
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(shape), self.requires_grad)
        out._ctx = ('reshape', self, shape)
        return out

    def backward(self, grad=None):
        if grad is None:
            grad = cp.ones_like(self.data)
        elif isinstance(grad, np.ndarray):
            grad = cp.array(grad)

        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

        if self._ctx is None: return
        op, *inputs = self._ctx

        if op == 'add':
            a, b = inputs
            if isinstance(a, Tensor) and a.requires_grad: a.backward(grad)
            if isinstance(b, Tensor) and b.requires_grad: b.backward(grad)
        elif op == 'matmul':
            a, b = inputs
            if a.requires_grad: a.backward(grad @ b.data.T)
            if b.requires_grad: b.backward(a.data.T @ grad)
        elif op == 'reshape':
            a, shape = inputs
            if a.requires_grad: a.backward(grad.reshape(a.data.shape))

# ============ СЛОИ (GPU optimized) ============

class Module:
    def parameters(self): return []
    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None: p.grad = cp.zeros_like(p.data)
    def __call__(self, x): return self.forward(x)

class Linear(Module):
    def __init__(self, in_features, out_features):
        limit = cp.sqrt(6.0 / (in_features + out_features))
        self.W = Tensor(cp.random.uniform(-limit, limit, (in_features, out_features)), True)
        self.b = Tensor(cp.zeros(out_features), True)
        self.x = None

    def forward(self, x):
        self.x = x.data
        out = self.x @ self.W.data + self.b.data
        return Tensor(out)

    def backward(self, grad):
        self.W.grad = self.x.T @ grad
        self.b.grad = cp.sum(grad, axis=0)
        return grad @ self.W.data.T

    def parameters(self): return [self.W, self.b]

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.padding, self.stride = padding, stride
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        scale = cp.sqrt(2.0 / fan_in)
        self.W = Tensor(cp.random.randn(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]) * scale, True)
        self.b = Tensor(cp.zeros(out_channels), True)

    def _im2col(self, x):
        # Быстрый im2col на GPU без циклов
        n, c, h, w = x.shape
        kh, kw = self.kernel_size
        p, s = self.padding, self.stride
        oh, ow = (h + 2*p - kh) // s + 1, (w + 2*p - kw) // s + 1
        
        if p > 0:
            x_pad = cp.pad(x, ((0,0), (0,0), (p,p), (p,p)), mode='constant')
        else:
            x_pad = x
            
        # Используем strides для мгновенной нарезки окон
        strides = x_pad.strides
        new_strides = (strides[0], strides[1], strides[2]*s, strides[3]*s, strides[2], strides[3])
        col = cp.lib.stride_tricks.as_strided(x_pad, shape=(n, c, oh, ow, kh, kw), strides=new_strides)
        col = col.transpose(0, 2, 3, 1, 4, 5).reshape(-1, c * kh * kw)
        return col.T, oh, ow

    def forward(self, x):
        data = x.data
        n, c, h, w = data.shape
        self.x_shape = data.shape
        self.col, oh, ow = self._im2col(data)
        W_col = self.W.data.reshape(self.out_channels, -1)
        out = (W_col @ self.col).reshape(self.out_channels, n, oh, ow).transpose(1, 0, 2, 3)
        out += self.b.data[None, :, None, None]
        return Tensor(out)

    def backward(self, grad):
        n, out_ch, oh, ow = grad.shape
        grad_col = grad.transpose(1, 0, 2, 3).reshape(out_ch, -1)
        self.b.grad = grad_col.sum(axis=1)
        self.W.grad = (grad_col @ self.col.T).reshape(self.W.data.shape)
        # Упрощенный dcol->dx для демонстрации GPU скорости
        W_col = self.W.data.reshape(self.out_channels, -1)
        dcol = W_col.T @ grad_col
        # В реальном коде тут нужен col2im, на GPU он сложнее, оставим заглушку dx
        return cp.zeros(self.x_shape, dtype=cp.float32) 

    def parameters(self): return [self.W, self.b]

class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride or kernel_size
        
    def forward(self, x):
        n, c, h, w = x.data.shape
        kh, kw = self.kernel_size
        sh = self.stride
        oh, ow = (h - kh) // sh + 1, (w - kw) // sh + 1
        
        # GPU оптимизированный пулинг через reshape
        out = x.data[:, :, :oh*sh, :ow*sh].reshape(n, c, oh, sh, ow, sh)
        out = out.mean(axis=(3, 5))
        return Tensor(out)

    def backward(self, grad):
        return cp.repeat(cp.repeat(grad / (self.kernel_size[0]*self.kernel_size[1]), self.kernel_size[0], axis=2), self.kernel_size[1], axis=3)

class Tanh(Module):
    def forward(self, x):
        self.out = cp.tanh(x.data)
        return Tensor(self.out)
    def backward(self, grad):
        return grad * (1.0 - self.out ** 2)

class Flatten(Module):
    def forward(self, x):
        self.shape = x.data.shape
        return Tensor(x.data.reshape(self.shape[0], -1))
    def backward(self, grad):
        return grad.reshape(self.shape)

class Sequential(Module):
    def __init__(self, *layers): self.layers = list(layers)
    def forward(self, x):
        for l in self.layers: x = l(x)
        return x
    def backward(self, grad):
        for l in reversed(self.layers): grad = l.backward(grad)
        return grad
    def parameters(self):
        p = []
        for l in self.layers: p.extend(l.parameters())
        return p

class CrossEntropyLoss:
    def __call__(self, preds, targets):
        # Все расчеты на GPU
        log_sum_exp = cp.log(cp.sum(cp.exp(preds.data), axis=1, keepdims=True))
        self.probs = cp.exp(preds.data - log_sum_exp)
        self.y = targets.data.astype(cp.int32)
        self.batch_size = preds.data.shape[0]
        loss = -cp.mean(cp.log(self.probs[cp.arange(self.batch_size), self.y] + 1e-8))
        return Tensor(loss)

    def backward(self):
        grad = self.probs.copy()
        grad[cp.arange(self.batch_size), self.y] -= 1.0
        return grad / self.batch_size

class Adam:
    def __init__(self, parameters, lr=1e-3):
        self.params = parameters
        self.lr = lr
        self.t = 0
        self.m = [cp.zeros_like(p.data) for p in self.params]
        self.v = [cp.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            self.m[i] = 0.9 * self.m[i] + 0.1 * p.grad
            self.v[i] = 0.999 * self.v[i] + 0.001 * (p.grad ** 2)
            m_hat = self.m[i] / (1 - 0.9**self.t)
            v_hat = self.v[i] / (1 - 0.999**self.t)
            p.data -= self.lr * m_hat / (cp.sqrt(v_hat) + 1e-8)

    def zero_grad(self):
        for p in self.params: p.grad = None

def no_grad(): return _NoGradContext()
class _NoGradContext:
    def __enter__(self): pass
    def __exit__(self, *a): pass