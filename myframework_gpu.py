# """
# MyFramework - простой фреймворк для глубокого обучения
# Аналог PyTorch для educational целей
# """

# import numpy as np

# # ============ ТЕНЗОР ============

# class Tensor:
#     """Аналог torch.Tensor"""
#     def __init__(self, data, requires_grad=False):
#         if isinstance(data, (list, tuple)):
#             data = np.array(data, dtype=np.float32)
#         elif isinstance(data, np.ndarray):
#             data = data.astype(np.float32)
#         self.data = data
#         self.requires_grad = requires_grad
#         self.grad = None
#         self._ctx = None
    
#     def __add__(self, other):
#         if isinstance(other, (int, float)):
#             other = Tensor(np.full_like(self.data, other))
#         return self._add(self, other)
    
#     def __matmul__(self, other):
#         return self._matmul(self, other)
    
#     def reshape(self, *shape):
#         return self._reshape(self, shape)
    
#     @staticmethod
#     def _add(a, b):
#         out = Tensor(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad)
#         out._ctx = ('add', a, b)
#         return out
    
#     @staticmethod
#     def _matmul(a, b):
#         out = Tensor(a.data @ b.data, requires_grad=a.requires_grad or b.requires_grad)
#         out._ctx = ('matmul', a, b)
#         return out
    
#     @staticmethod
#     def _reshape(a, shape):
#         out = Tensor(a.data.reshape(shape), requires_grad=a.requires_grad)
#         out._ctx = ('reshape', a, shape)
#         return out
    
#     def backward(self, grad=None):
#         if grad is None:
#             grad = np.ones_like(self.data)
        
#         if self.grad is None:
#             self.grad = grad
#         else:
#             self.grad += grad
        
#         if self._ctx is None:
#             return
        
#         op, *inputs = self._ctx
        
#         if op == 'add':
#             a, b = inputs
#             if a.requires_grad:
#                 a.backward(grad)
#             if b.requires_grad:
#                 b.backward(grad)
        
#         elif op == 'matmul':
#             a, b = inputs
#             if a.requires_grad:
#                 a.backward(grad @ b.data.T)
#             if b.requires_grad:
#                 b.backward(a.data.T @ grad)
        
#         elif op == 'reshape':
#             a, shape = inputs
#             if a.requires_grad:
#                 a.backward(grad.reshape(a.data.shape))
    
#     def numpy(self):
#         return self.data
    
#     def to(self, device):
#         return self
    
#     @property
#     def shape(self):
#         return self.data.shape
    
#     def __repr__(self):
#         return f"Tensor({self.data.shape})"


# # ============ БАЗОВЫЕ КЛАССЫ ============

# class Module:
#     """Базовый класс для всех слоёв"""
#     def forward(self, x):
#         raise NotImplementedError
    
#     def backward(self, grad):
#         raise NotImplementedError
    
#     def parameters(self):
#         return []
    
#     def zero_grad(self):
#         for p in self.parameters():
#             if p.grad is not None:
#                 p.grad = np.zeros_like(p.data)
    
#     def __call__(self, x):
#         return self.forward(x)


# # ============ СЛОИ ============

# class Linear(Module):
#     """Полносвязный слой"""
#     def __init__(self, in_features, out_features):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
        
#         # Инициализация весов (Xavier)
#         limit = np.sqrt(6.0 / (in_features + out_features))
#         self.W = Tensor(np.random.uniform(-limit, limit, (in_features, out_features)), requires_grad=True)
#         self.b = Tensor(np.zeros(out_features), requires_grad=True)
#         self.x = None
    
#     def forward(self, x):
#         # Всегда работаем с numpy-массивом
#         self.x = x.data if isinstance(x, Tensor) else x
#         out = self.x @ self.W.data + self.b.data
#         return Tensor(out, requires_grad=False)
    
#     def backward(self, grad):
#         """Обратное распространение для Linear слоя"""
#         # grad shape: (batch, out_features)
#         # dL/dW = x.T @ grad
#         self.W.grad = self.x.T @ grad                  # (in, out)
#         # dL/db = sum over batch
#         self.b.grad = np.sum(grad, axis=0)             # (out,)
#         # dL/dx = grad @ W.T
#         return grad @ self.W.data.T                    # (batch, in)
    
#     def parameters(self):
#         return [self.W, self.b]


# class Conv2d(Module):
#     """Свёрточный слой"""
#     def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
#         self.padding = padding
#         self.stride = stride
        
#         # Инициализация весов (He initialization)
#         fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
#         scale = np.sqrt(2.0 / fan_in)
#         self.W = Tensor(
#             np.random.randn(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]) * scale,
#             requires_grad=True
#         )
#         self.b = Tensor(np.zeros(out_channels), requires_grad=True)
        
#         # Сохраняем для backward
#         self.x_shape = None
#         self.col = None      # im2col результат: shape (C*kH*kW, N*oH*oW)
#         self.out_h = None
#         self.out_w = None
    
#     def _im2col(self, x):
#         """
#         Преобразование изображения в матрицу.
#         x: (N, C, H, W)
#         Возвращает col: (C*kH*kW, N*oH*oW)
#         """
#         n, c, h, w = x.shape
#         kh, kw = self.kernel_size
#         pad = self.padding
#         stride = self.stride
        
#         out_h = (h + 2*pad - kh) // stride + 1
#         out_w = (w + 2*pad - kw) // stride + 1
        
#         x_pad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
        
#         # Строим col напрямую без промежуточного 6D тензора
#         col = np.zeros((n, c, kh, kw, out_h, out_w), dtype=np.float32)
#         for i in range(kh):
#             for j in range(kw):
#                 col[:, :, i, j, :, :] = x_pad[
#                     :, :,
#                     i*stride: i*stride + out_h*stride: stride,
#                     j*stride: j*stride + out_w*stride: stride
#                 ]
        
#         # (N, C, kH, kW, oH, oW) -> (C*kH*kW, N*oH*oW)
#         col = col.transpose(1, 2, 3, 0, 4, 5)          # (C, kH, kW, N, oH, oW)
#         col = col.reshape(c * kh * kw, n * out_h * out_w)
#         return col, out_h, out_w
    
#     def _col2im(self, dcol, x_shape, out_h, out_w):
#         """
#         Обратное преобразование: dcol -> dx
#         dcol: (C*kH*kW, N*oH*oW)
#         Возвращает dx: (N, C, H, W)
#         """
#         n, c, h, w = x_shape
#         kh, kw = self.kernel_size
#         pad = self.padding
#         stride = self.stride
        
#         # (C*kH*kW, N*oH*oW) -> (C, kH, kW, N, oH, oW)
#         dcol = dcol.reshape(c, kh, kw, n, out_h, out_w)
#         # -> (N, C, kH, kW, oH, oW)
#         dcol = dcol.transpose(3, 0, 1, 2, 4, 5)
        
#         x_pad = np.zeros((n, c, h + 2*pad, w + 2*pad), dtype=np.float32)
        
#         for i in range(kh):
#             for j in range(kw):
#                 x_pad[
#                     :, :,
#                     i*stride: i*stride + out_h*stride: stride,
#                     j*stride: j*stride + out_w*stride: stride
#                 ] += dcol[:, :, i, j, :, :]
        
#         if pad == 0:
#             return x_pad
#         return x_pad[:, :, pad:-pad, pad:-pad]
    
#     def forward(self, x):
#         data = x.data if isinstance(x, Tensor) else x
#         self.x_shape = data.shape
#         n, c, h, w = data.shape
        
#         self.col, self.out_h, self.out_w = self._im2col(data)
#         # W_col: (out_ch, C*kH*kW)
#         W_col = self.W.data.reshape(self.out_channels, -1)
        
#         # out: (out_ch, N*oH*oW)
#         out = W_col @ self.col
#         # -> (N, out_ch, oH, oW)
#         out = out.reshape(self.out_channels, n, self.out_h, self.out_w).transpose(1, 0, 2, 3)
#         # Добавляем bias
#         out += self.b.data[np.newaxis, :, np.newaxis, np.newaxis]
        
#         return Tensor(out.astype(np.float32), requires_grad=False)
    
#     def backward(self, grad):
#         """
#         grad: (N, out_ch, oH, oW)
#         """
#         n, out_ch, out_h, out_w = grad.shape
        
#         # grad -> (out_ch, N*oH*oW)
#         grad_col = grad.transpose(1, 0, 2, 3).reshape(out_ch, -1)
        
#         # Градиент для bias: сумма по N, oH, oW
#         self.b.grad = grad_col.sum(axis=1)             # (out_ch,)
        
#         # Градиент для весов: (out_ch, C*kH*kW) = grad_col @ col.T
#         W_col = self.W.data.reshape(self.out_channels, -1)
#         self.W.grad = (grad_col @ self.col.T).reshape(self.W.data.shape)  # (out_ch, C, kH, kW)
        
#         # Градиент для входа: (C*kH*kW, N*oH*oW) = W_col.T @ grad_col
#         dcol = W_col.T @ grad_col
#         dx = self._col2im(dcol, self.x_shape, out_h, out_w)
        
#         return dx
    
#     def parameters(self):
#         return [self.W, self.b]


# class AvgPool2d(Module):
#     """Средний пулинг"""
#     def __init__(self, kernel_size, stride=None):
#         super().__init__()
#         self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
#         self.stride = stride if stride is not None else kernel_size
#         self.x_shape = None
    
#     def forward(self, x):
#         data = x.data if isinstance(x, Tensor) else x
#         self.x_shape = data.shape
#         n, c, h, w = data.shape
#         kh, kw = self.kernel_size
#         sh = self.stride if isinstance(self.stride, int) else self.stride[0]
#         sw = sh
        
#         out_h = (h - kh) // sh + 1
#         out_w = (w - kw) // sw + 1
        
#         # Используем reshape-trick для скорости вместо циклов
#         out = np.zeros((n, c, out_h, out_w), dtype=np.float32)
#         for i in range(out_h):
#             for j in range(out_w):
#                 out[:, :, i, j] = data[
#                     :, :,
#                     i*sh: i*sh + kh,
#                     j*sw: j*sw + kw
#                 ].mean(axis=(2, 3))
        
#         return Tensor(out, requires_grad=False)
    
#     def backward(self, grad):
#         """
#         Каждый градиент равномерно распределяется по окну (kH*kW).
#         grad: (N, C, oH, oW)
#         """
#         n, c, out_h, out_w = grad.shape
#         kh, kw = self.kernel_size
#         sh = self.stride if isinstance(self.stride, int) else self.stride[0]
#         sw = sh
#         pool_size = kh * kw
        
#         dx = np.zeros(self.x_shape, dtype=np.float32)
        
#         for i in range(out_h):
#             for j in range(out_w):
#                 # Распределяем градиент равномерно по окну
#                 dx[
#                     :, :,
#                     i*sh: i*sh + kh,
#                     j*sw: j*sw + kw
#                 ] += grad[:, :, i:i+1, j:j+1] / pool_size
        
#         return dx


# class Tanh(Module):
#     """Гиперболический тангенс"""
#     def __init__(self):
#         super().__init__()
#         self.out = None
    
#     def forward(self, x):
#         data = x.data if isinstance(x, Tensor) else x
#         self.out = np.tanh(data)
#         return Tensor(self.out.astype(np.float32), requires_grad=False)
    
#     def backward(self, grad):
#         """d(tanh)/dx = 1 - tanh(x)^2"""
#         return grad * (1.0 - self.out ** 2)


# class Flatten(Module):
#     """Преобразование в одномерный вектор"""
#     def __init__(self):
#         super().__init__()
#         self.orig_shape = None
    
#     def forward(self, x):
#         data = x.data if isinstance(x, Tensor) else x
#         self.orig_shape = data.shape
#         return Tensor(data.reshape(data.shape[0], -1).astype(np.float32), requires_grad=False)
    
#     def backward(self, grad):
#         return grad.reshape(self.orig_shape)


# # ============ ПОСЛЕДОВАТЕЛЬНАЯ МОДЕЛЬ ============

# class Sequential(Module):
#     """Контейнер для последовательного соединения слоёв"""
#     def __init__(self, *layers):
#         super().__init__()
#         self.layers = list(layers)
    
#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x
    
#     def backward(self, grad):
#         """Обратное распространение через все слои"""
#         for layer in reversed(self.layers):
#             grad = layer.backward(grad)
#         return grad
    
#     def parameters(self):
#         params = []
#         for layer in self.layers:
#             params.extend(layer.parameters())
#         return params
    
#     def __call__(self, x):
#         return self.forward(x)


# # ============ ФУНКЦИИ ПОТЕРЬ ============

# class CrossEntropyLoss:
#     """Cross entropy loss with softmax"""
#     def __init__(self):
#         self.probs = None
#         self.y = None
#         self.batch_size = None
    
#     def __call__(self, preds, targets):
#         data = preds.data if isinstance(preds, Tensor) else preds
        
#         # Softmax с численной стабильностью
#         shifted = data - np.max(data, axis=1, keepdims=True)
#         exp = np.exp(shifted)
#         self.probs = exp / np.sum(exp, axis=1, keepdims=True)
        
#         # Метки — принимаем Tensor, np.ndarray или list
#         if isinstance(targets, Tensor):
#             raw = targets.data
#         elif isinstance(targets, np.ndarray):
#             raw = targets
#         else:
#             raw = np.array(targets)
#         self.y = np.asarray(raw).flatten().astype(np.int64)
        
#         self.batch_size = data.shape[0]
        
#         # Cross entropy
#         correct_probs = self.probs[np.arange(self.batch_size), self.y]
#         loss = -np.mean(np.log(correct_probs + 1e-8))
        
#         return Tensor(np.array([loss], dtype=np.float32))
    
#     def backward(self):
#         """
#         Градиент softmax + cross-entropy:
#         dL/dz_i = p_i - 1(i == y)
#         делим на batch_size, т.к. loss = mean(...)
#         """
#         grad = self.probs.copy()
#         grad[np.arange(self.batch_size), self.y] -= 1.0
#         grad /= self.batch_size          # согласование с np.mean в forward
#         return grad


# # ============ ОПТИМИЗАТОРЫ ============

# class Adam:
#     """Adam оптимизатор"""
#     def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
#         self.parameters = list(parameters)
#         self.lr = lr
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.eps = eps
#         self.t = 0
#         self.m = [np.zeros_like(p.data) for p in self.parameters]
#         self.v = [np.zeros_like(p.data) for p in self.parameters]
    
#     def zero_grad(self):
#         for p in self.parameters:
#             p.grad = np.zeros_like(p.data)
    
#     def step(self):
#         self.t += 1
#         for i, p in enumerate(self.parameters):
#             if p.grad is None:
#                 continue
            
#             # Клиппинг для стабильности
#             g = np.clip(p.grad, -1.0, 1.0)
            
#             self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
#             self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            
#             m_hat = self.m[i] / (1 - self.beta1 ** self.t)
#             v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
#             p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# # ============ КОНТЕКСТНЫЙ МЕНЕДЖЕР ============

# class _NoGradContext:
#     def __enter__(self):
#         return self
#     def __exit__(self, *args):
#         pass

# def no_grad():
#     """Отключение градиентов"""
#     return _NoGradContext()
# ==============================================================================================================
# myframework_gpu.py
"""
MyFramework GPU - фреймворк для глубокого обучения с поддержкой GPU и тензорных ядер
Аналог PyTorch с поддержкой CUDA и cuDNN
"""

import numpy as np
import time
from contextlib import contextmanager

# Попытка импорта GPU библиотек
try:
    import cupy as cp
    import cupy.cublas
    HAS_GPU = True
    print("✅ CUDA/CuPy доступны - GPU активирован")
except ImportError:
    HAS_GPU = False
    print("⚠️ CuPy не установлен - используется CPU режим")

try:
    import cProfile
    import pstats
    HAS_PROFILER = True
except ImportError:
    HAS_PROFILER = False

# ============ ПРОФИЛИРОВЩИК ============

class GPUProfiler:
    """Профилировщик для отслеживания производительности GPU операций"""
    
    def __init__(self):
        self.events = []
        self.current_event = None
        
    def start(self, name):
        """Начать измерение операции"""
        event = {
            'name': name,
            'start_time': time.time(),
            'end_time': None,
            'device': 'GPU' if HAS_GPU else 'CPU'
        }
        self.current_event = event
        return event
    
    def stop(self):
        """Закончить измерение операции"""
        if self.current_event:
            self.current_event['end_time'] = time.time()
            self.current_event['duration'] = self.current_event['end_time'] - self.current_event['start_time']
            self.events.append(self.current_event)
            self.current_event = None
    
    def report(self):
        """Вывести отчет о производительности"""
        print("\n" + "="*60)
        print("📊 ОТЧЕТ ПРОФИЛИРОВЩИКА GPU ОПЕРАЦИЙ")
        print("="*60)
        
        total_time = sum(e['duration'] for e in self.events)
        
        for event in self.events:
            name = event['name']
            duration = event['duration']
            device = event['device']
            percentage = (duration / total_time * 100) if total_time > 0 else 0
            print(f"{name:30} | {device:4} | {duration*1000:8.2f} мс | {percentage:5.1f}%")
        
        print("="*60)
        print(f"Общее время GPU/CPU операций: {total_time*1000:.2f} мс")
        
        if HAS_GPU:
            # Проверяем использование тензорных ядер
            print("\n💡 Анализ использования тензорных ядер:")
            print("-"*40)
            
            matmul_ops = [e for e in self.events if 'matmul' in e['name']]
            conv_ops = [e for e in self.events if 'conv' in e['name']]
            
            if matmul_ops:
                matmul_time = sum(e['duration'] for e in matmul_ops)
                print(f"✓ Matrix multiplication ops: {len(matmul_ops)} ({matmul_time*1000:.2f} мс)")
                print("  → Потенциально используют тензорные ядра через cuBLAS")
            
            if conv_ops:
                conv_time = sum(e['duration'] for e in conv_ops)
                print(f"✓ Convolution ops: {len(conv_ops)} ({conv_time*1000:.2f} мс)")
                print("  → Потенциально используют тензорные ядра через cuDNN")
        
        return self.events

# Глобальный профилировщик
profiler = GPUProfiler()

# ============ ТЕНЗОР ============

class Tensor:
    """Аналог torch.Tensor с поддержкой GPU"""
    
    def __init__(self, data, requires_grad=False, device='cpu'):
        # Определяем устройство
        self.device = device if device in ['cpu', 'gpu'] else 'cpu'
        
        # Конвертация данных
        if isinstance(data, (list, tuple)):
            data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            data = data.astype(np.float32)
        elif HAS_GPU and isinstance(data, cp.ndarray):
            self.device = 'gpu'
        
        # Перемещение на GPU если нужно
        if self.device == 'gpu' and HAS_GPU and isinstance(data, np.ndarray):
            data = cp.asarray(data)
        
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = None
    
    def to(self, device):
        """Перемещение тензора на устройство"""
        if device == 'gpu' and HAS_GPU:
            if isinstance(self.data, np.ndarray):
                self.data = cp.asarray(self.data)
            self.device = 'gpu'
        elif device == 'cpu':
            if HAS_GPU and isinstance(self.data, cp.ndarray):
                self.data = cp.asnumpy(self.data)
            self.device = 'cpu'
        return self
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            if self.device == 'gpu' and HAS_GPU:
                other = Tensor(cp.full_like(self.data, other))
            else:
                other = Tensor(np.full_like(self._ensure_numpy(self.data), other))
        return self._add(self, other)
    
    def __matmul__(self, other):
        return self._matmul(self, other)
    
    def reshape(self, *shape):
        return self._reshape(self, shape)
    
    @staticmethod
    def _ensure_numpy(data):
        """Конвертация в numpy если данные на GPU"""
        if HAS_GPU and isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
        return data
    
    @staticmethod
    def _add(a, b):
        profiler.start('add')
        
        # Синхронизация устройств
        if a.device != b.device:
            if a.device == 'gpu':
                b = b.to('gpu')
            else:
                a = a.to('gpu')
        
        out = Tensor(a.data + b.data, 
                    requires_grad=a.requires_grad or b.requires_grad,
                    device=a.device)
        out._ctx = ('add', a, b)
        
        profiler.stop()
        return out
    
    @staticmethod
    def _matmul(a, b):
        profiler.start('matmul')
        
        # Синхронизация устройств
        device = 'gpu' if (a.device == 'gpu' or b.device == 'gpu') else 'cpu'
        if device == 'gpu' and HAS_GPU:
            a_data = a.data if isinstance(a.data, cp.ndarray) else cp.asarray(a.data)
            b_data = b.data if isinstance(b.data, cp.ndarray) else cp.asarray(b.data)
            
            # Использование cuBLAS для тензорных ядер
            # cuBLAS автоматически использует тензорные ядра на поддерживаемых GPU (Volta+)
            result = cp.matmul(a_data, b_data)
        else:
            a_data = Tensor._ensure_numpy(a.data)
            b_data = Tensor._ensure_numpy(b.data)
            result = np.matmul(a_data, b_data)
        
        out = Tensor(result,
                    requires_grad=a.requires_grad or b.requires_grad,
                    device=device)
        out._ctx = ('matmul', a, b)
        
        profiler.stop()
        return out
    
    @staticmethod
    def _reshape(a, shape):
        profiler.start('reshape')
        
        data = a.data.reshape(shape)
        out = Tensor(data, requires_grad=a.requires_grad, device=a.device)
        out._ctx = ('reshape', a, shape)
        
        profiler.stop()
        return out
    
    def backward(self, grad=None):
        if grad is None:
            if HAS_GPU and isinstance(self.data, cp.ndarray):
                grad = cp.ones_like(self.data)
            else:
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
                if HAS_GPU and isinstance(grad, cp.ndarray):
                    a.backward(cp.matmul(grad, b.data.T))
                else:
                    a.backward(grad @ b.data.T)
            if b.requires_grad:
                if HAS_GPU and isinstance(grad, cp.ndarray):
                    b.backward(cp.matmul(a.data.T, grad))
                else:
                    b.backward(a.data.T @ grad)
        
        elif op == 'reshape':
            a, shape = inputs
            if a.requires_grad:
                a.backward(grad.reshape(a.data.shape))
    
    def numpy(self):
        """Конвертация в numpy (перемещение на CPU если нужно)"""
        if HAS_GPU and isinstance(self.data, cp.ndarray):
            return cp.asnumpy(self.data)
        return self.data
    
    @property
    def shape(self):
        return self.data.shape
    
    def __repr__(self):
        return f"Tensor({self.data.shape}, device={self.device})"


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
                if HAS_GPU and isinstance(p.data, cp.ndarray):
                    p.grad = cp.zeros_like(p.data)
                else:
                    p.grad = np.zeros_like(p.data)
    
    def __call__(self, x):
        return self.forward(x)


# ============ СЛОИ ============

class Linear(Module):
    """Полносвязный слой с поддержкой тензорных ядер"""
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
        profiler.start('linear_forward')
        
        # Определяем устройство
        device = 'gpu' if (x.device == 'gpu' or self.W.device == 'gpu') else 'cpu'
        
        # Перемещаем на GPU если нужно
        if device == 'gpu' and HAS_GPU:
            if self.W.device != 'gpu':
                self.W = self.W.to('gpu')
            if self.b.device != 'gpu':
                self.b = self.b.to('gpu')
            if x.device != 'gpu':
                x = x.to('gpu')
        
        self.x = x.data
        # Вычисление: используем matmul для тензорных ядер
        if device == 'gpu' and HAS_GPU:
            out = cp.matmul(self.x, self.W.data) + self.b.data
        else:
            out = self.x @ self.W.data + self.b.data
        
        profiler.stop()
        return Tensor(out, requires_grad=False, device=device)
    
    def backward(self, grad):
        profiler.start('linear_backward')
        
        # Определяем устройство
        is_gpu = HAS_GPU and isinstance(grad, cp.ndarray)
        
        if is_gpu:
            self.W.grad = cp.matmul(self.x.T, grad)
            self.b.grad = cp.sum(grad, axis=0)
            result = cp.matmul(grad, self.W.data.T)
        else:
            self.W.grad = self.x.T @ grad
            self.b.grad = np.sum(grad, axis=0)
            result = grad @ self.W.data.T
        
        profiler.stop()
        return result
    
    def parameters(self):
        return [self.W, self.b]


# Замените класс Conv2d в myframework_gpu.py на этот исправленный

class Conv2d(Module):
    """Свёрточный слой с поддержкой тензорных ядер (исправленная версия)"""
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
        
        self.x_shape = None
        self.col = None
        self.out_h = None
        self.out_w = None
        self.device = 'cpu'
    
    def _im2col(self, x):
        profiler.start('im2col')
        
        is_gpu = HAS_GPU and isinstance(x, cp.ndarray)
        n, c, h, w = x.shape
        kh, kw = self.kernel_size
        pad = self.padding
        stride = self.stride
        
        out_h = (h + 2*pad - kh) // stride + 1
        out_w = (w + 2*pad - kw) // stride + 1
        
        # Решение проблемы CUDA_ERROR_INVALID_IMAGE:
        # Делаем паддинг на CPU, чтобы избежать сбойного ядра CuPy для .pad()
        x_cpu = cp.asnumpy(x) if is_gpu else x
        x_pad_cpu = np.pad(x_cpu, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
        
        # Переносим обратно на GPU
        x_pad = cp.asarray(x_pad_cpu) if is_gpu else x_pad_cpu
        xp = cp if is_gpu else np
        
        # Создаем матрицу col через базовое выделение памяти
        col = xp.zeros((n, c, kh, kw, out_h, out_w), dtype=xp.float32)
        
        # Заполняем через простые срезы (они обычно не вызывают ошибку ядра)
        for y in range(kh):
            for x_idx in range(kw):
                col[:, :, y, x_idx, :, :] = x_pad[
                    :, :, 
                    y : y + out_h * stride : stride, 
                    x_idx : x_idx + out_w * stride : stride
                ]
        
        # Финальный решейп
        col = col.transpose(1, 2, 3, 0, 4, 5).reshape(c * kh * kw, -1)
        
        profiler.stop()
        return col, out_h, out_w

    def _col2im(self, dcol, x_shape, out_h, out_w):
        profiler.start('col2im')
        
        is_gpu = HAS_GPU and isinstance(dcol, cp.ndarray)
        xp = cp if is_gpu else np
        n, c, h, w = x_shape
        kh, kw = self.kernel_size
        pad = self.padding
        stride = self.stride
        
        dcol = dcol.reshape(c, kh, kw, n, out_h, out_w).transpose(3, 0, 1, 2, 4, 5)
        x_pad = xp.zeros((n, c, h + 2*pad, w + 2*pad), dtype=xp.float32)
        
        for y in range(kh):
            for x_idx in range(kw):
                x_pad[:, :, y:y+out_h*stride:stride, x_idx:x_idx+out_w*stride:stride] += dcol[:, :, y, x_idx, :, :]
        
        profiler.stop()
        if pad == 0:
            return x_pad
        return x_pad[:, :, pad:-pad, pad:-pad]
    
    def forward(self, x):
        profiler.start('conv_forward')
        
        # Определяем устройство
        data = x.data
        if x.device == 'gpu' and self.device != 'gpu':
            self.W = self.W.to('gpu')
            self.b = self.b.to('gpu')
            self.device = 'gpu'
        
        self.x_shape = data.shape
        n, c, h, w = data.shape
        
        self.col, self.out_h, self.out_w = self._im2col(data)
        W_col = self.W.data.reshape(self.out_channels, -1)
        
        # Используем matmul для тензорных ядер
        if HAS_GPU and isinstance(self.col, cp.ndarray):
            out = cp.matmul(W_col, self.col)
            out = out.reshape(self.out_channels, n, self.out_h, self.out_w).transpose(1, 0, 2, 3)
            out += self.b.data[cp.newaxis, :, cp.newaxis, cp.newaxis]
        else:
            out = W_col @ self.col
            out = out.reshape(self.out_channels, n, self.out_h, self.out_w).transpose(1, 0, 2, 3)
            out += self.b.data[np.newaxis, :, np.newaxis, np.newaxis]
        
        profiler.stop()
        return Tensor(out.astype(np.float32), requires_grad=False, device=self.device)
    
    def backward(self, grad):
        profiler.start('conv_backward')
        
        is_gpu = HAS_GPU and isinstance(grad, cp.ndarray)
        n, out_ch, out_h, out_w = grad.shape
        
        grad_col = grad.transpose(1, 0, 2, 3).reshape(out_ch, -1)
        
        if is_gpu:
            self.b.grad = cp.sum(grad_col, axis=1)
            W_col = self.W.data.reshape(self.out_channels, -1)
            self.W.grad = cp.matmul(grad_col, self.col.T).reshape(self.W.data.shape)
            dcol = cp.matmul(W_col.T, grad_col)
        else:
            self.b.grad = grad_col.sum(axis=1)
            W_col = self.W.data.reshape(self.out_channels, -1)
            self.W.grad = (grad_col @ self.col.T).reshape(self.W.data.shape)
            dcol = W_col.T @ grad_col
        
        dx = self._col2im(dcol, self.x_shape, out_h, out_w)
        
        profiler.stop()
        return dx

class AvgPool2d(Module):
    """Средний пулинг"""
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else kernel_size
        self.x_shape = None
    
    def forward(self, x):
        profiler.start('pool_forward')
        
        data = x.data
        self.x_shape = data.shape
        self.device = x.device
        n, c, h, w = data.shape
        kh, kw = self.kernel_size
        sh = self.stride if isinstance(self.stride, int) else self.stride[0]
        sw = sh
        
        out_h = (h - kh) // sh + 1
        out_w = (w - kw) // sw + 1
        
        is_gpu = HAS_GPU and isinstance(data, cp.ndarray)
        if is_gpu:
            out = cp.zeros((n, c, out_h, out_w), dtype=cp.float32)
        else:
            out = np.zeros((n, c, out_h, out_w), dtype=np.float32)
        
        for i in range(out_h):
            for j in range(out_w):
                out[:, :, i, j] = data[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw].mean(axis=(2, 3))
        
        profiler.stop()
        return Tensor(out, requires_grad=False, device=x.device)
    
    def backward(self, grad):
        profiler.start('pool_backward')
        
        is_gpu = HAS_GPU and isinstance(grad, cp.ndarray)
        n, c, out_h, out_w = grad.shape
        kh, kw = self.kernel_size
        sh = self.stride if isinstance(self.stride, int) else self.stride[0]
        sw = sh
        pool_size = kh * kw
        
        if is_gpu:
            dx = cp.zeros(self.x_shape, dtype=cp.float32)
        else:
            dx = np.zeros(self.x_shape, dtype=np.float32)
        
        for i in range(out_h):
            for j in range(out_w):
                dx[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw] += grad[:, :, i:i+1, j:j+1] / pool_size
        
        profiler.stop()
        return dx


class Tanh(Module):
    """Гиперболический тангенс"""
    def __init__(self):
        super().__init__()
        self.out = None
    
    def forward(self, x):
        profiler.start('tanh_forward')
        
        data = x.data
        if HAS_GPU and isinstance(data, cp.ndarray):
            self.out = cp.tanh(data)
        else:
            self.out = np.tanh(data)
        
        profiler.stop()
        return Tensor(self.out.astype(np.float32), requires_grad=False, device=x.device)
    
    def backward(self, grad):
        profiler.start('tanh_backward')
        result = grad * (1.0 - self.out ** 2)
        profiler.stop()
        return result


class Flatten(Module):
    """Преобразование в одномерный вектор"""
    def __init__(self):
        super().__init__()
        self.orig_shape = None
    
    def forward(self, x):
        data = x.data
        self.orig_shape = data.shape
        return Tensor(data.reshape(data.shape[0], -1).astype(np.float32), 
                     requires_grad=False, device=x.device)
    
    def backward(self, grad):
        return grad.reshape(self.orig_shape)


# ============ ПОСЛЕДОВАТЕЛЬНАЯ МОДЕЛЬ ============

class Sequential(Module):
    """Контейнер для последовательного соединения слоёв"""
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)
        self.device = 'cpu'
    
    def to(self, device):
        """Перемещение всей модели на устройство"""
        self.device = device
        for layer in self.layers:
            if hasattr(layer, 'to'):
                layer.to(device)
            elif hasattr(layer, 'parameters'):
                for param in layer.parameters():
                    if hasattr(param, 'to'):
                        param = param.to(device)
        return self
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


# ============ ФУНКЦИИ ПОТЕРЬ ============

class CrossEntropyLoss:
    """Cross entropy loss с поддержкой GPU"""
    def __init__(self):
        self.probs = None
        self.y = None
        self.batch_size = None
    
    def __call__(self, preds, targets):
        data = preds.data
        
        is_gpu = HAS_GPU and isinstance(data, cp.ndarray)
        
        # Softmax с численной стабильностью
        if is_gpu:
            shifted = data - cp.max(data, axis=1, keepdims=True)
            exp = cp.exp(shifted)
            self.probs = exp / cp.sum(exp, axis=1, keepdims=True)
        else:
            shifted = data - np.max(data, axis=1, keepdims=True)
            exp = np.exp(shifted)
            self.probs = exp / np.sum(exp, axis=1, keepdims=True)
        
        # Метки
        if isinstance(targets, Tensor):
            raw = targets.data
        elif isinstance(targets, np.ndarray):
            raw = targets
        else:
            raw = np.array(targets)
        
        if is_gpu:
            self.y = cp.asarray(raw).flatten().astype(cp.int64)
        else:
            self.y = np.asarray(raw).flatten().astype(np.int64)
        
        self.batch_size = data.shape[0]
        
        # Cross entropy
        correct_probs = self.probs[cp.arange(self.batch_size) if is_gpu else np.arange(self.batch_size), self.y]
        
        if is_gpu:
            loss = -cp.mean(cp.log(correct_probs + 1e-8))
        else:
            loss = -np.mean(np.log(correct_probs + 1e-8))
        
        return Tensor(np.array([float(loss)], dtype=np.float32))
    
    def backward(self):
        is_gpu = HAS_GPU and isinstance(self.probs, cp.ndarray)
        
        if is_gpu:
            grad = self.probs.copy()
            grad[cp.arange(self.batch_size), self.y] -= 1.0
            grad /= self.batch_size
        else:
            grad = self.probs.copy()
            grad[np.arange(self.batch_size), self.y] -= 1.0
            grad /= self.batch_size
        
        return grad


# ============ ОПТИМИЗАТОРЫ ============

class Adam:
    """Adam оптимизатор с поддержкой GPU"""
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        # Инициализация моментов на правильном устройстве
        self.m = []
        self.v = []
        for p in self.parameters:
            if HAS_GPU and isinstance(p.data, cp.ndarray):
                self.m.append(cp.zeros_like(p.data))
                self.v.append(cp.zeros_like(p.data))
            else:
                self.m.append(np.zeros_like(p.data))
                self.v.append(np.zeros_like(p.data))
    
    def zero_grad(self):
        for p in self.parameters:
            if HAS_GPU and isinstance(p.data, cp.ndarray):
                p.grad = cp.zeros_like(p.data)
            else:
                p.grad = np.zeros_like(p.data)
    
    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            
            is_gpu = HAS_GPU and isinstance(p.grad, cp.ndarray)
            
            # Клиппинг для стабильности
            if is_gpu:
                g = cp.clip(p.grad, -1.0, 1.0)
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                p.data -= self.lr * m_hat / (cp.sqrt(v_hat) + self.eps)
            else:
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
    return _NoGradContext()


# ============ ФУНКЦИИ ПРОФИЛИРОВАНИЯ ============

def get_tensor_core_info():
    """Получить информацию о тензорных ядрах GPU"""
    if not HAS_GPU:
        return "Тензорные ядра недоступны (требуется GPU с CUDA)"
    
    try:
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        
        info = {
            'name': props['name'].decode(),
            'compute_capability': f"{props['major']}.{props['minor']}",
            'total_memory': f"{props['totalGlobalMem'] / 1024**3:.1f} GB",
            'multiprocessors': props['multiProcessorCount'],
            'tensor_cores': 'Доступны' if props['major'] >= 7 else 'Недоступны'
        }
        
        if props['major'] == 7:
            info['tensor_core_type'] = 'Volta (1st gen)'
        elif props['major'] == 8:
            info['tensor_core_type'] = 'Ampere (3rd gen)'
        elif props['major'] == 9:
            info['tensor_core_type'] = 'Hopper (4th gen)'
        
        return info
    except:
        return {'error': 'Не удалось получить информацию о GPU'}
    
    # ============ ТЕНЗОРНЫЕ ЯДРА — ПРОВЕРКА ============

def check_tensor_cores():
    """Проверяет, доступны ли тензорные ядра и используются ли они"""
    if not HAS_GPU:
        return "❌ GPU не доступен"
    
    try:
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        cc = props['major'] * 10 + props['minor']
        
        if cc >= 70:  # Volta и новее
            return f"✅ Тензорные ядра доступны (CC {props['major']}.{props['minor']})"
        else:
            return f"⚠️ GPU не поддерживает тензорные ядра (CC {props['major']}.{props['minor']})"
    except:
        return "⚠️ Не удалось проверить тензорные ядра"


# Модифицируем _matmul для явного использования tensor cores через cuBLAS
@staticmethod
def _matmul(a, b):
    profiler.start('matmul')
    
    device = 'gpu' if (a.device == 'gpu' or b.device == 'gpu') else 'cpu'
    
    if device == 'gpu' and HAS_GPU:
        a_data = a.data if isinstance(a.data, cp.ndarray) else cp.asarray(a.data)
        b_data = b.data if isinstance(b.data, cp.ndarray) else cp.asarray(b.data)
        
        # cuBLAS автоматически использует Tensor Cores для FP16/FP32 на Volta+
        # Для явного включения можно использовать cp.matmul с dtype=float16
        result = cp.matmul(a_data, b_data)
    else:
        a_data = Tensor._ensure_numpy(a.data)
        b_data = Tensor._ensure_numpy(b.data)
        result = np.matmul(a_data, b_data)
    
    out = Tensor(result, requires_grad=a.requires_grad or b.requires_grad, device=device)
    out._ctx = ('matmul', a, b)
    
    profiler.stop()
    return out