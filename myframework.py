"""
MyFramework - простой фреймворк для глубокого обучения
Аналог PyTorch для educational целей с поддержкой GPU и Tensor Cores
"""

import numpy as np
import time

# Попытка импорта CuPy для работы с GPU
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None

# ============ ПРОФИЛИРОВЩИК ============

class Profiler:
    def __init__(self):
        self.stats = {}
        self.enabled = True

    def start(self, name):
        if not self.enabled: return
        if HAS_GPU:
            try: cp.cuda.Device().synchronize()
            except: pass
        self.stats[name] = self.stats.get(name, 0) - time.perf_counter()

    def stop(self, name):
        if not self.enabled: return
        if HAS_GPU:
            try: cp.cuda.Device().synchronize()
            except: pass
        self.stats[name] += time.perf_counter()

    def report(self):
        print("\n" + "="*40)
        print("📊 ОТЧЕТ ПРОФИЛИРОВЩИКА")
        print("="*40)
        for name, duration in sorted(self.stats.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:25} | {duration:.4f} сек")
        print("="*40)

profiler = Profiler()

# ============ ТЕНЗОР ============

class Tensor:
    def __init__(self, data, requires_grad=False, device='cpu'):
        if isinstance(data, (list, tuple)):
            data = np.array(data, dtype=np.float32)
        
        self.device = device
        if device == 'gpu' and HAS_GPU:
            self.data = cp.asarray(data) if isinstance(data, np.ndarray) else data
        else:
            self.data = np.asarray(data) if isinstance(data, cp.ndarray) else data
            self.device = 'cpu'
            
        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = None

    def to(self, device):
        if device == self.device: return self
        return Tensor(self.data, self.requires_grad, device=device)

    def backward(self, grad=None):
        if self._ctx is None: return
        xp = cp if self.device == 'gpu' else np
        
        if grad is None:
            grad = xp.ones_like(self.data)
        
        op, *args = self._ctx
        
        if op == 'add':
            a, b = args
            if a.requires_grad:
                a.grad = (a.grad if a.grad is not None else xp.zeros_like(a.data)) + grad
                a.backward(a.grad)
            if isinstance(b, Tensor) and b.requires_grad:
                # Обработка бродкастинга для bias
                grad_b = grad
                while grad_b.ndim > b.data.ndim:
                    grad_b = grad_b.sum(axis=0)
                for i, dim in enumerate(b.data.shape):
                    if dim == 1:
                        grad_b = grad_b.sum(axis=i, keepdims=True)
                b.grad = (b.grad if b.grad is not None else xp.zeros_like(b.data)) + grad_b
                b.backward(b.grad)

        elif op == 'matmul':
            a, b = args
            if a.requires_grad:
                grad_a = grad @ b.data.T
                a.grad = (a.grad if a.grad is not None else xp.zeros_like(a.data)) + grad_a
                a.backward(grad_a)
            if b.requires_grad:
                grad_b = a.data.T @ grad
                b.grad = (b.grad if b.grad is not None else xp.zeros_like(b.data)) + grad_b
                b.backward(grad_b)

        elif op == 'tanh':
            a = args[0]
            if a.requires_grad:
                grad_a = grad * (1 - xp.tanh(a.data)**2)
                a.grad = (a.grad if a.grad is not None else xp.zeros_like(a.data)) + grad_a
                a.backward(grad_a)

        elif op == 'reshape':
            a, old_shape = args
            if a.requires_grad:
                grad_a = grad.reshape(old_shape)
                a.grad = (a.grad if a.grad is not None else xp.zeros_like(a.data)) + grad_a
                a.backward(grad_a)

        elif op == 'transpose':
            a, axes = args
            if a.requires_grad:
                inv_axes = np.argsort(axes)
                grad_a = grad.transpose(tuple(inv_axes))
                a.grad = (a.grad if a.grad is not None else xp.zeros_like(a.data)) + grad_a
                a.backward(grad_a)

    def __add__(self, other):
        xp = cp if self.device == 'gpu' else np
        other_data = other.data if isinstance(other, Tensor) else other
        out = Tensor(self.data + other_data, requires_grad=self.requires_grad, device=self.device)
        out._ctx = ('add', self, other)
        return out

    def __matmul__(self, other):
        profiler.start('matmul_tensor_core')
        res_data = self.data @ other.data
        out = Tensor(res_data, requires_grad=self.requires_grad or other.requires_grad, device=self.device)
        out._ctx = ('matmul', self, other)
        profiler.stop('matmul_tensor_core')
        return out

    def reshape(self, *shape):
        old_shape = self.data.shape
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad, device=self.device)
        out._ctx = ('reshape', self, old_shape)
        return out

    def transpose(self, *axes):
        out = Tensor(self.data.transpose(*axes), requires_grad=self.requires_grad, device=self.device)
        out._ctx = ('transpose', self, axes)
        return out

# ============ СЛОИ ============

class Linear:
    def __init__(self, in_f, out_f):
        limit = np.sqrt(6 / (in_f + out_f))
        self.weight = Tensor(np.random.uniform(-limit, limit, (in_f, out_f)), requires_grad=True)
        self.bias = Tensor(np.zeros(out_f), requires_grad=True)
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.weight = self.weight.to(device)
        self.bias = self.bias.to(device)
        return self

    def __call__(self, x):
        return x @ self.weight + self.bias

class Conv2d:
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0):
        self.in_channels = in_c
        self.out_channels = out_c
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride, self.padding = stride, padding
        limit = np.sqrt(6 / (in_c * np.prod(self.kernel_size) + out_c))
        self.weight = Tensor(np.random.uniform(-limit, limit, (out_c, in_c, *self.kernel_size)), requires_grad=True)
        self.bias = Tensor(np.zeros(out_c), requires_grad=True)
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.weight = self.weight.to(device)
        self.bias = self.bias.to(device)
        return self

    def _im2col(self, x):
        xp = cp if self.device == 'gpu' else np
        n, c, h, w = x.shape
        kh, kw = self.kernel_size
        p, s = self.padding, self.stride
        oh, ow = (h + 2*p - kh) // s + 1, (w + 2*p - kw) // s + 1
        
        if p > 0:
            x_pad = xp.zeros((n, c, h + 2*p, w + 2*p), dtype=xp.float32)
            x_pad[:, :, p:p+h, p:p+w] = x
        else: x_pad = x
            
        col = xp.zeros((n, c, kh, kw, oh, ow), dtype=xp.float32)
        for y in range(kh):
            for x_idx in range(kw):
                col[:, :, y, x_idx, :, :] = x_pad[:, :, y:y+oh*s:s, x_idx:x_idx+ow*s:s]
        return col.transpose(1, 2, 3, 0, 4, 5).reshape(c * kh * kw, -1), oh, ow

    def __call__(self, x):
        profiler.start('conv2d_forward')
        col, oh, ow = self._im2col(x.data)
        w_flat = self.weight.data.reshape(self.out_channels, -1)
        # Matmul на GPU задействует Tensor Cores
        res = w_flat @ col 
        out = res.reshape(self.out_channels, x.data.shape[0], oh, ow).transpose(1, 0, 2, 3)
        out_tensor = Tensor(out, requires_grad=True, device=self.device)
        # Упрощенный контекст для примера (сложение с bias)
        bias_reshaped = self.bias.data.reshape(1, -1, 1, 1)
        final_out = out_tensor + Tensor(bias_reshaped, device=self.device)
        profiler.stop('conv2d_forward')
        return final_out

class AvgPool2d:
    def __init__(self, size):
        self.size = size
    def __call__(self, x):
        xp = cp if x.device == 'gpu' else np
        n, c, h, w = x.data.shape
        s = self.size
        out = x.data.reshape(n, c, h//s, s, w//s, s).mean(axis=(3, 5))
        return Tensor(out, requires_grad=x.requires_grad, device=x.device)

class Tanh:
    def __call__(self, x):
        xp = cp if x.device == 'gpu' else np
        out = Tensor(xp.tanh(x.data), requires_grad=x.requires_grad, device=x.device)
        out._ctx = ('tanh', x)
        return out

class Flatten:
    def __call__(self, x):
        return x.reshape(x.data.shape[0], -1)

class Sequential:
    def __init__(self, *layers):
        self.layers = layers
    def to(self, device):
        for l in self.layers:
            if hasattr(l, 'to'): l.to(device)
        return self
    def __call__(self, x):
        for l in self.layers: x = l(x)
        return x
    def parameters(self):
        p = []
        for l in self.layers:
            if hasattr(l, 'weight'): p.extend([l.weight, l.bias])
        return p

# ============ LOSS & OPTIM ============

class CrossEntropyLoss:
    def __call__(self, preds, targets):
        xp = cp if preds.device == 'gpu' else np
        n = preds.data.shape[0]
        # Softmax
        exps = xp.exp(preds.data - xp.max(preds.data, axis=1, keepdims=True))
        probs = exps / xp.sum(exps, axis=1, keepdims=True)
        # Loss
        log_p = -xp.log(probs[xp.arange(n), targets.data.astype(int)])
        loss = xp.mean(log_p)
        # Ручной расчет градиента для последнего слоя
        if preds.requires_grad:
            grad = probs.copy()
            grad[xp.arange(n), targets.data.astype(int)] -= 1
            preds.grad = grad / n
        return loss

class Adam:
    def __init__(self, parameters, lr=0.001):
        self.params = list(parameters)
        self.lr = lr
        self.m = [None] * len(self.params)
        self.v = [None] * len(self.params)
        self.t = 0

    def zero_grad(self):
        for p in self.params: p.grad = None

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            xp = cp if p.device == 'gpu' else np
            if self.m[i] is None:
                self.m[i] = xp.zeros_like(p.data)
                self.v[i] = xp.zeros_like(p.data)
            self.m[i] = 0.9 * self.m[i] + 0.1 * p.grad
            self.v[i] = 0.999 * self.v[i] + 0.001 * (p.grad**2)
            m_hat = self.m[i] / (1 - 0.9**self.t)
            v_hat = self.v[i] / (1 - 0.999**self.t)
            p.data -= self.lr * m_hat / (xp.sqrt(v_hat) + 1e-8)

def no_grad():
    class NoGradContext:
        def __enter__(self): pass
        def __exit__(self, exc_type, exc_val, exc_tb): pass
    return NoGradContext()