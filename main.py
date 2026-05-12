import numpy as np
import torchvision.datasets
from myframework import (Tensor, Sequential, Conv2d, AvgPool2d, Tanh, 
                         Flatten, Linear, CrossEntropyLoss, Adam, no_grad, profiler)

# Загрузка MNIST
print("Загрузка MNIST...")
MNIST_train = torchvision.datasets.MNIST('./', download=True, train=True)
X_train = MNIST_train.data.numpy().astype(np.float32)[:, None, :, :] / 255.0
y_train = MNIST_train.targets.numpy()

# Модель LeNet-5
model = Sequential(
    Conv2d(1, 6, 5, padding=2), Tanh(), AvgPool2d(2),
    Conv2d(6, 16, 5), Tanh(), AvgPool2d(2),
    Flatten(),
    Linear(16*5*5, 120), Tanh(),
    Linear(120, 84), Tanh(),
    Linear(84, 10)
).to('gpu')

optimizer = Adam(model.parameters(), lr=1e-3)
criterion = CrossEntropyLoss()

batch_size = 128
n_epochs = 3
n_samples = 2000 # Для быстрой проверки

print("\nНачинаем обучение (Гибридный режим: индексация на CPU, умножение на GPU)...")
print("-" * 60)
for epoch in range(n_epochs):
    order = np.random.permutation(n_samples)
    epoch_loss, correct = 0, 0
    
    for i in range(0, n_samples, batch_size):
        optimizer.zero_grad()
        idx = order[i:i+batch_size]
        
        # Данные передаем на GPU
        X_batch = Tensor(X_train[idx], device='gpu')
        y_batch = Tensor(y_train[idx], device='gpu')
        
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += float(loss.data)
        correct += (preds.data.argmax(axis=1) == y_batch.data).sum()
        
    print(f"Epoch {epoch+1} | Loss: {epoch_loss/(n_samples/batch_size):.4f} | Acc: {float(correct)/n_samples:.2%}")

# Вывод отчета по тензорным ядрам
profiler.report()