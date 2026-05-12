"""
MyFramework - простой фреймворк для глубокого обучения с поддержкой GPU и Tensor Cores
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
            cp.cuda.Device().synchronize()
        self.stats[name] = self.stats.get(name, 0) - time.perf_counter()

    def stop(self, name):
        if not self.enabled: return
        if HAS_GPU:
            cp.cuda.Device().synchronize()
        self.stats[name] += time.perf_counter()

    def report(self):
        print("\n" + "="*30)
        print("📊 ОТЧЕТ ПРОФИЛИРОВЩИКА (GPU/CPU)")
        print("="*30)
        for name, duration in sorted(self.stats.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:20} | {duration:.4f} сек")
        print("="*30)

profiler = Profiler()

# ============ ТЕНЗОР ============

class Tensor:
    """Аналог torch.Tensor с поддержкой устройств"""
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
        if grad is None:
            xp = cp if self.device == 'gpu' else np
            grad = xp.ones_like(self.data)
        
        op, *args = self._ctx
        if op == 'add':
            a, b = args
            if a.requires_grad:
                a.grad = (a.grad if a.grad is not None else 0) + grad
                a.backward(grad)
            if b.requires_grad:
                b.grad = (b.grad if b.grad is not None else 0) + grad
                b.backward(grad)
        elif op == 'matmul':
            a, b = args
            # Считаем градиенты через matmul (используются Tensor Cores)
            if a.requires_grad:
                grad_a = grad @ b.data.T
                a.grad = (a.grad if a.grad is not None else 0) + grad_a
                a.backward(grad_a)
            if b.requires_grad:
                grad_b = a.data.T @ grad
                b.grad = (b.grad if b.grad is not None else 0) + grad_b
                b.backward(grad_b)
        # Другие операции (tanh, reshape и т.д.) по аналогии...

    def __add__(self, other):
        return self._add(self, other)
    
    def __matmul__(self, other):
        return self._matmul(self, other)
    
    @staticmethod
    def _add(a, b):
        xp = cp if a.device == 'gpu' else np
        b_data = b.data if isinstance(b, Tensor) else b
        out = Tensor(a.data + b_data, requires_grad=a.requires_grad, device=a.device)
        out._ctx = ('add', a, b if isinstance(b, Tensor) else Tensor(b_data, device=a.device))
        return out
    
    @staticmethod
    def _matmul(a, b):
        profiler.start('matmul_tensor_core')
        xp = cp if a.device == 'gpu' else np
        # Использование cp.matmul на Ada Lovelace задействует тензорные ядра
        res_data = a.data @ b.data
        out = Tensor(res_data, requires_grad=a.requires_grad or b.requires_grad, device=a.device)
        out._ctx = ('matmul', a, b)
        profiler.stop('matmul_tensor_core')
        return out

# ============ СЛОИ ============

class Linear:
    def __init__(self, in_features, out_features):
        # Инициализация Xavier
        limit = np.sqrt(6 / (in_features + out_features))
        self.weight = Tensor(np.random.uniform(-limit, limit, (in_features, out_features)), requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.weight = self.weight.to(device)
        self.bias = self.bias.to(device)
        return self

    def __call__(self, x):
        return x @ self.weight + self.bias

class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        
        limit = np.sqrt(6 / (in_channels * np.prod(self.kernel_size) + out_channels))
        self.weight = Tensor(np.random.uniform(-limit, limit, (out_channels, in_channels, *self.kernel_size)), requires_grad=True)
        self.bias = Tensor(np.zeros(out_channels), requires_grad=True)
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
        oh = (h + 2*p - kh) // s + 1
        ow = (w + 2*p - kw) // s + 1
        
        # Безопасный паддинг без вызова JIT-ядер
        if p > 0:
            x_pad = xp.zeros((n, c, h + 2*p, w + 2*p), dtype=xp.float32)
            x_pad[:, :, p:p+h, p:p+w] = x
        else:
            x_pad = x
            
        # Формирование колонок через срезы
        col = xp.zeros((n, c, kh, kw, oh, ow), dtype=xp.float32)
        for y in range(kh):
            for x_idx in range(kw):
                col[:, :, y, x_idx, :, :] = x_pad[:, :, y:y+oh*s:s, x_idx:x_idx+ow*s:s]
        
        return col.transpose(1, 2, 3, 0, 4, 5).reshape(c * kh * kw, -1), oh, ow

    def __call__(self, x):
        profiler.start('conv2d_total')
        xp = cp if self.device == 'gpu' else np
        col, oh, ow = self._im2col(x.data)
        
        # Основная операция свертки -> MatMul (Tensor Cores)
        # weight: (out_c, in_c*kh*kw), col: (in_c*kh*kw, n*oh*ow)
        w_flat = self.weight.data.reshape(self.out_channels, -1)
        res = w_flat @ col
        
        out = res.reshape(self.out_channels, x.data.shape[0], oh, ow).transpose(1, 0, 2, 3)
        
        # Добавление bias
        out = out + self.bias.data.reshape(1, -1, 1, 1)
        
        profiler.stop('conv2d_total')
        return Tensor(out, requires_grad=True, device=self.device)

class Tanh:
    def __call__(self, x):
        xp = cp if x.device == 'gpu' else np
        return Tensor(xp.tanh(x.data), requires_grad=x.requires_grad, device=x.device)

class Flatten:
    def __call__(self, x):
        return Tensor(x.data.reshape(x.data.shape[0], -1), requires_grad=x.requires_grad, device=x.device)

class Sequential:
    def __init__(self, *layers):
        self.layers = layers
    def to(self, device):
        for layer in self.layers:
            if hasattr(layer, 'to'): layer.to(device)
        return self
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        params = []
        for l in self.layers:
            if hasattr(l, 'weight'): params.append(l.weight)
            if hasattr(l, 'bias'): params.append(l.bias)
        return params

# ============ ОПТИМИЗАТОР ============

class Adam:
    def __init__(self, parameters, lr=0.001):
        self.params = parameters
        self.lr = lr
        self.m = [0] * len(parameters)
        self.v = [0] * len(parameters)
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            xp = cp if p.device == 'gpu' else np
            if isinstance(self.m[i], int):
                self.m[i] = xp.zeros_like(p.data)
                self.v[i] = xp.zeros_like(p.data)
            
            self.m[i] = 0.9 * self.m[i] + 0.1 * p.grad
            self.v[i] = 0.999 * self.v[i] + 0.001 * (p.grad**2)
            m_hat = self.m[i] / (1 - 0.9**self.t)
            v_hat = self.v[i] / (1 - 0.999**self.t)
            p.data -= self.lr * m_hat / (xp.sqrt(v_hat) + 1e-8)

    def zero_grad(self):
        for p in self.params:
            p.grad = None