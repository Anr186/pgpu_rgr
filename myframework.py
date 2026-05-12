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
        print("\n" + "="*45)
        print("📊 ОТЧЕТ ПРОФИЛИРОВЩИКА (GPU/CPU)")
        print("="*45)
        for name, duration in sorted(self.stats.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:25} | {duration:.4f} сек")
        print("="*45)

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
        
        if isinstance(grad, Tensor): grad = grad.data

        op, *args = self._ctx
        
        if op == 'add':
            a, b = args
            if a.requires_grad:
                a.grad = (a.grad if a.grad is not None else xp.zeros_like(a.data)) + grad
                a.backward(a.grad)
            if isinstance(b, Tensor) and b.requires_grad:
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
                a.grad = (a.grad if a.grad is not None else xp.zeros_like(a.data)) + grad.reshape(old_shape)
                a.backward(a.grad)

    def __add__(self, other):
        other_t = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        out = Tensor(self.data + other_t.data, requires_grad=self.requires_grad or other_t.requires_grad, device=self.device)
        out._ctx = ('add', self, other_t)
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

# ============ СЛОИ ============

class Linear:
    def __init__(self, in_features, out_features):
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
        self.stride, self.padding = stride, padding
        
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
        res = w_flat @ col 
        out = res.reshape(self.out_channels, x.data.shape[0], oh, ow).transpose(1, 0, 2, 3)
        
        out_t = Tensor(out, requires_grad=True, device=self.device)
        bias_t = Tensor(self.bias.data.reshape(1, -1, 1, 1), device=self.device)
        final_out = out_t + bias_t
        profiler.stop('conv2d_forward')
        return final_out

class AvgPool2d:
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        
    def __call__(self, x):
        xp = cp if x.device == 'gpu' else np
        n, c, h, w = x.data.shape
        ks, s = self.kernel_size, self.stride
        oh, ow = (h - ks) // s + 1, (w - ks) // s + 1
        
        # Упрощенная реализация через reshape для пулинга без перекрытий
        if ks == s:
            out = x.data[:, :, :oh*s, :ow*s].reshape(n, c, oh, ks, ow, ks).mean(axis=(3, 5))
        else:
            # Для общего случая
            out = xp.zeros((n, c, oh, ow), dtype=xp.float32)
            for i in range(oh):
                for j in range(ow):
                    out[:,:,i,j] = x.data[:, :, i*s:i*s+ks, j*s:j*s+ks].mean(axis=(2,3))
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
        logits = preds.data
        exps = xp.exp(logits - xp.max(logits, axis=1, keepdims=True))
        probs = exps / xp.sum(exps, axis=1, keepdims=True)
        
        t_idx = targets.data.astype(int)
        loss_val = float(xp.mean(-xp.log(probs[xp.arange(n), t_idx] + 1e-10)))
        
        if preds.requires_grad:
            grad = probs.copy()
            grad[xp.arange(n), t_idx] -= 1
            preds.grad = grad / n
        return Tensor(loss_val, device=preds.device)

class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(parameters)
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
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
            
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad**2)
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            p.data -= self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)

def no_grad():
    class NoGrad:
        def __enter__(self): pass
        def __exit__(self, *args): pass
    return NoGrad()