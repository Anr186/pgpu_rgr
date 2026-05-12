"""
MyFramework - простой фреймворк для глубокого обучения
Аналог PyTorch для educational целей
"""

import numpy as np

# ============ ТЕНЗОР ============

class Tensor:
    """Аналог torch.Tensor"""
    def __init__(self, data, requires_grad=False):
        if isinstance(data, (list, tuple)):
            data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            data = data.astype(np.float32)
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = None
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.full_like(self.data, other))
        return self._add(self, other)
    
    def __matmul__(self, other):
        return self._matmul(self, other)
    
    def reshape(self, *shape):
        return self._reshape(self, shape)
    
    @staticmethod
    def _add(a, b):
        out = Tensor(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad)
        out._ctx = ('add', a, b)
        return out
    
    @staticmethod
    def _matmul(a, b):
        out = Tensor(a.data @ b.data, requires_grad=a.requires_grad or b.requires_grad)
        out._ctx = ('matmul', a, b)
        return out
    
    @staticmethod
    def _reshape(a, shape):
        out = Tensor(a.data.reshape(shape), requires_grad=a.requires_grad)
        out._ctx = ('reshape', a, shape)
        return out
    
    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)
        
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
        
        if self._ctx is None:
            return
        
        op, *inputs = self._ctx
        
        if op == 'add':
            a, b = inputs
            if a.requires_grad:
                a.backward(grad)
            if b.requires_grad:
                b.backward(grad)
        
        elif op == 'matmul':
            a, b = inputs
            if a.requires_grad:
                a.backward(grad @ b.data.T)
            if b.requires_grad:
                b.backward(a.data.T @ grad)
        
        elif op == 'reshape':
            a, shape = inputs
            if a.requires_grad:
                a.backward(grad.reshape(a.data.shape))
    
    def numpy(self):
        return self.data
    
    def to(self, device):
        return self
    
    @property
    def shape(self):
        return self.data.shape
    
    def __repr__(self):
        return f"Tensor({self.data.shape})"


# ============ БАЗОВЫЕ КЛАССЫ ============

class Module:
    """Базовый класс для всех слоёв"""
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, grad):
        raise NotImplementedError
    
    def parameters(self):
        return []
    
    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad = np.zeros_like(p.data)
    
    def __call__(self, x):
        return self.forward(x)


# ============ СЛОИ ============

class Linear(Module):
    """Полносвязный слой"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Инициализация весов (Xavier)
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = Tensor(np.random.uniform(-limit, limit, (in_features, out_features)), requires_grad=True)
        self.b = Tensor(np.zeros(out_features), requires_grad=True)
        self.x = None
    
    def forward(self, x):
        # Всегда работаем с numpy-массивом
        self.x = x.data if isinstance(x, Tensor) else x
        out = self.x @ self.W.data + self.b.data
        return Tensor(out, requires_grad=False)
    
    def backward(self, grad):
        """Обратное распространение для Linear слоя"""
        # grad shape: (batch, out_features)
        # dL/dW = x.T @ grad
        self.W.grad = self.x.T @ grad                  # (in, out)
        # dL/db = sum over batch
        self.b.grad = np.sum(grad, axis=0)             # (out,)
        # dL/dx = grad @ W.T
        return grad @ self.W.data.T                    # (batch, in)
    
    def parameters(self):
        return [self.W, self.b]


class Conv2d(Module):
    """Свёрточный слой"""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.padding = padding
        self.stride = stride
        
        # Инициализация весов (He initialization)
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        scale = np.sqrt(2.0 / fan_in)
        self.W = Tensor(
            np.random.randn(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]) * scale,
            requires_grad=True
        )
        self.b = Tensor(np.zeros(out_channels), requires_grad=True)
        
        # Сохраняем для backward
        self.x_shape = None
        self.col = None      # im2col результат: shape (C*kH*kW, N*oH*oW)
        self.out_h = None
        self.out_w = None
    
    def _im2col(self, x):
        """
        Преобразование изображения в матрицу.
        x: (N, C, H, W)
        Возвращает col: (C*kH*kW, N*oH*oW)
        """
        n, c, h, w = x.shape
        kh, kw = self.kernel_size
        pad = self.padding
        stride = self.stride
        
        out_h = (h + 2*pad - kh) // stride + 1
        out_w = (w + 2*pad - kw) // stride + 1
        
        x_pad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
        
        # Строим col напрямую без промежуточного 6D тензора
        col = np.zeros((n, c, kh, kw, out_h, out_w), dtype=np.float32)
        for i in range(kh):
            for j in range(kw):
                col[:, :, i, j, :, :] = x_pad[
                    :, :,
                    i*stride: i*stride + out_h*stride: stride,
                    j*stride: j*stride + out_w*stride: stride
                ]
        
        # (N, C, kH, kW, oH, oW) -> (C*kH*kW, N*oH*oW)
        col = col.transpose(1, 2, 3, 0, 4, 5)          # (C, kH, kW, N, oH, oW)
        col = col.reshape(c * kh * kw, n * out_h * out_w)
        return col, out_h, out_w
    
    def _col2im(self, dcol, x_shape, out_h, out_w):
        """
        Обратное преобразование: dcol -> dx
        dcol: (C*kH*kW, N*oH*oW)
        Возвращает dx: (N, C, H, W)
        """
        n, c, h, w = x_shape
        kh, kw = self.kernel_size
        pad = self.padding
        stride = self.stride
        
        # (C*kH*kW, N*oH*oW) -> (C, kH, kW, N, oH, oW)
        dcol = dcol.reshape(c, kh, kw, n, out_h, out_w)
        # -> (N, C, kH, kW, oH, oW)
        dcol = dcol.transpose(3, 0, 1, 2, 4, 5)
        
        x_pad = np.zeros((n, c, h + 2*pad, w + 2*pad), dtype=np.float32)
        
        for i in range(kh):
            for j in range(kw):
                x_pad[
                    :, :,
                    i*stride: i*stride + out_h*stride: stride,
                    j*stride: j*stride + out_w*stride: stride
                ] += dcol[:, :, i, j, :, :]
        
        if pad == 0:
            return x_pad
        return x_pad[:, :, pad:-pad, pad:-pad]
    
    def forward(self, x):
        data = x.data if isinstance(x, Tensor) else x
        self.x_shape = data.shape
        n, c, h, w = data.shape
        
        self.col, self.out_h, self.out_w = self._im2col(data)
        # W_col: (out_ch, C*kH*kW)
        W_col = self.W.data.reshape(self.out_channels, -1)
        
        # out: (out_ch, N*oH*oW)
        out = W_col @ self.col
        # -> (N, out_ch, oH, oW)
        out = out.reshape(self.out_channels, n, self.out_h, self.out_w).transpose(1, 0, 2, 3)
        # Добавляем bias
        out += self.b.data[np.newaxis, :, np.newaxis, np.newaxis]
        
        return Tensor(out.astype(np.float32), requires_grad=False)
    
    def backward(self, grad):
        """
        grad: (N, out_ch, oH, oW)
        """
        n, out_ch, out_h, out_w = grad.shape
        
        # grad -> (out_ch, N*oH*oW)
        grad_col = grad.transpose(1, 0, 2, 3).reshape(out_ch, -1)
        
        # Градиент для bias: сумма по N, oH, oW
        self.b.grad = grad_col.sum(axis=1)             # (out_ch,)
        
        # Градиент для весов: (out_ch, C*kH*kW) = grad_col @ col.T
        W_col = self.W.data.reshape(self.out_channels, -1)
        self.W.grad = (grad_col @ self.col.T).reshape(self.W.data.shape)  # (out_ch, C, kH, kW)
        
        # Градиент для входа: (C*kH*kW, N*oH*oW) = W_col.T @ grad_col
        dcol = W_col.T @ grad_col
        dx = self._col2im(dcol, self.x_shape, out_h, out_w)
        
        return dx
    
    def parameters(self):
        return [self.W, self.b]


class AvgPool2d(Module):
    """Средний пулинг"""
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else kernel_size
        self.x_shape = None
    
    def forward(self, x):
        data = x.data if isinstance(x, Tensor) else x
        self.x_shape = data.shape
        n, c, h, w = data.shape
        kh, kw = self.kernel_size
        sh = self.stride if isinstance(self.stride, int) else self.stride[0]
        sw = sh
        
        out_h = (h - kh) // sh + 1
        out_w = (w - kw) // sw + 1
        
        # Используем reshape-trick для скорости вместо циклов
        out = np.zeros((n, c, out_h, out_w), dtype=np.float32)
        for i in range(out_h):
            for j in range(out_w):
                out[:, :, i, j] = data[
                    :, :,
                    i*sh: i*sh + kh,
                    j*sw: j*sw + kw
                ].mean(axis=(2, 3))
        
        return Tensor(out, requires_grad=False)
    
    def backward(self, grad):
        """
        Каждый градиент равномерно распределяется по окну (kH*kW).
        grad: (N, C, oH, oW)
        """
        n, c, out_h, out_w = grad.shape
        kh, kw = self.kernel_size
        sh = self.stride if isinstance(self.stride, int) else self.stride[0]
        sw = sh
        pool_size = kh * kw
        
        dx = np.zeros(self.x_shape, dtype=np.float32)
        
        for i in range(out_h):
            for j in range(out_w):
                # Распределяем градиент равномерно по окну
                dx[
                    :, :,
                    i*sh: i*sh + kh,
                    j*sw: j*sw + kw
                ] += grad[:, :, i:i+1, j:j+1] / pool_size
        
        return dx


class Tanh(Module):
    """Гиперболический тангенс"""
    def __init__(self):
        super().__init__()
        self.out = None
    
    def forward(self, x):
        data = x.data if isinstance(x, Tensor) else x
        self.out = np.tanh(data)
        return Tensor(self.out.astype(np.float32), requires_grad=False)
    
    def backward(self, grad):
        """d(tanh)/dx = 1 - tanh(x)^2"""
        return grad * (1.0 - self.out ** 2)


class Flatten(Module):
    """Преобразование в одномерный вектор"""
    def __init__(self):
        super().__init__()
        self.orig_shape = None
    
    def forward(self, x):
        data = x.data if isinstance(x, Tensor) else x
        self.orig_shape = data.shape
        return Tensor(data.reshape(data.shape[0], -1).astype(np.float32), requires_grad=False)
    
    def backward(self, grad):
        return grad.reshape(self.orig_shape)


# ============ ПОСЛЕДОВАТЕЛЬНАЯ МОДЕЛЬ ============

class Sequential(Module):
    """Контейнер для последовательного соединения слоёв"""
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backward(self, grad):
        """Обратное распространение через все слои"""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def __call__(self, x):
        return self.forward(x)


# ============ ФУНКЦИИ ПОТЕРЬ ============

class CrossEntropyLoss:
    """Cross entropy loss with softmax"""
    def __init__(self):
        self.probs = None
        self.y = None
        self.batch_size = None
    
    def __call__(self, preds, targets):
        data = preds.data if isinstance(preds, Tensor) else preds
        
        # Softmax с численной стабильностью
        shifted = data - np.max(data, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)
        
        # Метки — принимаем Tensor, np.ndarray или list
        if isinstance(targets, Tensor):
            raw = targets.data
        elif isinstance(targets, np.ndarray):
            raw = targets
        else:
            raw = np.array(targets)
        self.y = np.asarray(raw).flatten().astype(np.int64)
        
        self.batch_size = data.shape[0]
        
        # Cross entropy
        correct_probs = self.probs[np.arange(self.batch_size), self.y]
        loss = -np.mean(np.log(correct_probs + 1e-8))
        
        return Tensor(np.array([loss], dtype=np.float32))
    
    def backward(self):
        """
        Градиент softmax + cross-entropy:
        dL/dz_i = p_i - 1(i == y)
        делим на batch_size, т.к. loss = mean(...)
        """
        grad = self.probs.copy()
        grad[np.arange(self.batch_size), self.y] -= 1.0
        grad /= self.batch_size          # согласование с np.mean в forward
        return grad


# ============ ОПТИМИЗАТОРЫ ============

class Adam:
    """Adam оптимизатор"""
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
    
    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)
    
    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            
            # Клиппинг для стабильности
            g = np.clip(p.grad, -1.0, 1.0)
            
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ============ КОНТЕКСТНЫЙ МЕНЕДЖЕР ============

class _NoGradContext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

def no_grad():
    """Отключение градиентов"""
    return _NoGradContext()
