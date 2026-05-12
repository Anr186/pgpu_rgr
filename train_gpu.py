# train_gpu.py
import numpy as np
from myframework_gpu import (
    Tensor, Sequential, Linear, Conv2d, AvgPool2d, 
    Flatten, Tanh, CrossEntropyLoss, Adam, profiler, check_tensor_cores
)

print(check_tensor_cores())  # Проверка тензорных ядер

# Данные (пример: классификация изображений 32x32)
np.random.seed(42)
X_train = np.random.randn(1000, 3, 32, 32).astype(np.float32)
y_train = np.random.randint(0, 10, 1000)

# Модель
model = Sequential(
    Conv2d(3, 32, kernel_size=3, padding=1),
    Tanh(),
    AvgPool2d(2),
    Conv2d(32, 64, kernel_size=3, padding=1),
    Tanh(),
    AvgPool2d(2),
    Flatten(),
    Linear(64*8*8, 128),
    Tanh(),
    Linear(128, 10)
).to('gpu')  # ← Перенос на GPU

optimizer = Adam(model.parameters(), lr=1e-3)
criterion = CrossEntropyLoss()

# Обучение
for epoch in range(10):
    # Forward
    preds = model(Tensor(X_train, device='gpu'))
    loss = criterion(preds, y_train)
    
    # Backward
    model.zero_grad()
    grad = criterion.backward()
    model.backward(grad)
    
    # Step
    optimizer.step()
    
    if epoch % 2 == 0:
        print(f"Epoch {epoch}: loss={loss.numpy()[0]:.4f}")

# Отчёт профилировщика
profiler.report()