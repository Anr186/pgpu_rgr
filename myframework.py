import numpy as np
import time

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
        self.device = device
        if device == 'gpu' and HAS_GPU:
            self.data = cp.asarray(data) if not isinstance(data, cp.ndarray) else data
        else:
            self.data = np.asarray(data) if not isinstance(data, np.ndarray) else data
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
        if grad is None: grad = xp.ones_like(self.data)
        if isinstance(grad, Tensor): grad = grad.data

        op, *args = self._ctx
        
        if op == 'add':
            a, b = args
            if a.requires_grad:
                a.grad = (a.grad if a.grad is not None else xp.zeros_like(a.data)) + grad
                a.backward(a.grad)
            if isinstance(b, Tensor) and b.requires_grad:
                gb = grad
                while gb.ndim > b.data.ndim: gb = gb.sum(axis=0)
                for i, dim in enumerate(b.data.shape):
                    if dim == 1: gb = gb.sum(axis=i, keepdims=True)
                b.grad = (b.grad if b.grad is not None else xp.zeros_like(b.data)) + gb
                b.backward(b.grad)
        elif op == 'matmul':
            a, b = args
            if a.requires_grad:
                ga = grad @ b.data.T
                a.grad = (a.grad if a.grad is not None else xp.zeros_like(a.data)) + ga
                a.backward(ga)
            if b.requires_grad:
                gb = a.data.T @ grad
                b.grad = (b.grad if b.grad is not None else xp.zeros_like(b.data)) + gb
                b.backward(gb)
        elif op == 'tanh':
            a = args[0]
            if a.requires_grad:
                ga = grad * (1 - xp.tanh(a.data)**2)
                a.grad = (a.grad if a.grad is not None else xp.zeros_like(a.data)) + ga
                a.backward(ga)
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
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad, device=self.device)
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
        self.weight = Tensor(np.random.uniform(-limit, limit, (in_features, out_features)), True)
        self.bias = Tensor(np.zeros(out_features), True)
        self.device = 'cpu'
    def to(self, device):
        self.device = device
        self.weight, self.bias = self.weight.to(device), self.bias.to(device)
        return self
    def __call__(self, x): return x @ self.weight + self.bias

class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride, self.padding = stride, padding
        limit = np.sqrt(6 / (in_channels * np.prod(self.kernel_size) + out_channels))
        self.weight = Tensor(np.random.uniform(-limit, limit, (out_channels, in_channels, *self.kernel_size)), True)
        self.bias = Tensor(np.zeros(out_channels), True)
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.weight, self.bias = self.weight.to(device), self.bias.to(device)
        return self

    def _im2col(self, x_data):
        # ОПРЕДЕЛЯЕМ ТЕКУЩЕЕ УСТРОЙСТВО
        is_gpu = HAS_GPU and isinstance(x_data, cp.ndarray)
        xp = cp if is_gpu else np
        
        n, c, h, w = x_data.shape
        kh, kw = self.kernel_size
        p, s = self.padding, self.stride
        oh, ow = (h + 2*p - kh) // s + 1, (w + 2*p - kw) // s + 1
        
        # УБИРАЕМ ПЕРЕНОС НА ГПУ ДЛЯ ЭТОЙ ЧАСТИ (Используем CPU для паддинга и сборки колонок)
        # Это обходит CUDA_ERROR_INVALID_IMAGE
        x_cpu = cp.asnumpy(x_data) if is_gpu else x_data
        
        if p > 0:
            x_pad = np.pad(x_cpu, ((0,0), (0,0), (p,p), (p,p)), mode='constant')
        else:
            x_pad = x_cpu
            
        col = np.zeros((n, c, kh, kw, oh, ow), dtype=np.float32)
        for y in range(kh):
            for x_i in range(kw):
                col[:, :, y, x_i, :, :] = x_pad[:, :, y:y+oh*s:s, x_i:x_i+ow*s:s]
        
        col_flat = col.transpose(1, 2, 3, 0, 4, 5).reshape(c * kh * kw, -1)
        
        # Возвращаем результат на исходное устройство для Matmul
        return (cp.asarray(col_flat) if is_gpu else col_flat), oh, ow

    def __call__(self, x):
        profiler.start('conv2d_forward')
        col, oh, ow = self._im2col(x.data)
        
        # Сама операция умножения (Tensor Cores) остается на GPU
        w_flat = self.weight.data.reshape(self.out_channels, -1)
        res = w_flat @ col 
        
        out = res.reshape(self.out_channels, x.data.shape[0], oh, ow).transpose(1, 0, 2, 3)
        final = Tensor(out, True, self.device) + Tensor(self.bias.data.reshape(1, -1, 1, 1), device=self.device)
        profiler.stop('conv2d_forward')
        return final

class AvgPool2d:
    def __init__(self, kernel_size, stride=None):
        self.ks = kernel_size
        self.stride = stride if stride is not None else kernel_size
    def __call__(self, x):
        xp = cp if x.device == 'gpu' else np
        n, c, h, w = x.data.shape
        ks, s = self.ks, self.stride
        oh, ow = (h - ks) // s + 1, (w - ks) // s + 1
        if ks == s:
            out = x.data[:, :, :oh*s, :ow*s].reshape(n, c, oh, ks, ow, ks).mean(axis=(3, 5))
        else:
            # Для пулинга тоже используем CPU, чтобы избежать INVALID_IMAGE при индексации
            x_cpu = cp.asnumpy(x.data) if x.device == 'gpu' else x.data
            out_cpu = np.zeros((n, c, oh, ow), dtype=np.float32)
            for i in range(oh):
                for j in range(ow):
                    out_cpu[:,:,i,j] = x_cpu[:, :, i*s:i*s+ks, j*s:j*s+ks].mean(axis=(2,3))
            out = cp.asarray(out_cpu) if x.device == 'gpu' else out_cpu
        return Tensor(out, x.requires_grad, x.device)

class Tanh:
    def __call__(self, x):
        xp = cp if x.device == 'gpu' else np
        out = Tensor(xp.tanh(x.data), x.requires_grad, x.device)
        out._ctx = ('tanh', x)
        return out

class Flatten:
    def __call__(self, x): return x.reshape(x.data.shape[0], -1)

class Sequential:
    def __init__(self, *layers): self.layers = layers
    def to(self, dev):
        for l in self.layers: 
            if hasattr(l, 'to'): l.to(dev)
        return self
    def __call__(self, x):
        for l in self.layers: x = l(x)
        return x
    def parameters(self):
        p = []
        for l in self.layers:
            if hasattr(l, 'weight'): p.extend([l.weight, l.bias])
        return p

class CrossEntropyLoss:
    def __call__(self, preds, targets):
        xp = cp if preds.device == 'gpu' else np
        n = preds.data.shape[0]
        exps = xp.exp(preds.data - xp.max(preds.data, axis=1, keepdims=True))
        probs = exps / xp.sum(exps, axis=1, keepdims=True)
        t_idx = targets.data.astype(int)
        loss_val = float(xp.mean(-xp.log(probs[xp.arange(n), t_idx] + 1e-10)))
        if preds.requires_grad:
            grad = probs.copy()
            grad[xp.arange(n), t_idx] -= 1
            preds.grad = grad / n
        return Tensor(loss_val, device=preds.device)

class Adam:
    def __init__(self, params, lr=0.001):
        self.params = list(params)
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
                self.m[i], self.v[i] = xp.zeros_like(p.data), xp.zeros_like(p.data)
            self.m[i] = 0.9 * self.m[i] + 0.1 * p.grad
            self.v[i] = 0.999 * self.v[i] + 0.001 * (p.grad**2)
            mh = self.m[i] / (1 - 0.9**self.t)
            vh = self.v[i] / (1 - 0.999**self.t)
            p.data -= self.lr * mh / (xp.sqrt(vh) + 1e-8)

def no_grad():
    class NG:
        def __enter__(self): pass
        def __exit__(self, *a): pass
    return NG()