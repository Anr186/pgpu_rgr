"""
Обучение LeNet-5 с использованием MyFramework (с обратным распространением)
"""

import random
import numpy as np
import torchvision.datasets
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import cProfile

# Импортируем профилировщик из фреймворка
from myframework import Tensor, Sequential, Conv2d, AvgPool2d, Tanh, Flatten, Linear, CrossEntropyLoss, Adam, no_grad, profiler

random.seed(0)
np.random.seed(0)

# ============ ЗАГРУЗКА ДАННЫХ ============
print("Загрузка MNIST...")
MNIST_train = torchvision.datasets.MNIST('./', download=True, train=True)
MNIST_test = torchvision.datasets.MNIST('./', download=True, train=False)

X_train = MNIST_train.data.numpy().astype(np.float32)
y_train = MNIST_train.targets.numpy()
X_test = MNIST_test.data.numpy().astype(np.float32)
y_test = MNIST_test.targets.numpy()

# Нормализация и добавление размерности канала (N, C, H, W)
X_train = X_train[:, None, :, :] / 255.0
X_test = X_test[:, None, :, :] / 255.0

print(f"Размер обучающей выборки: {len(y_train)}, размер тестовой: {len(y_test)}")
print(f"Форма данных: {X_test.shape}")

# ============ ОПРЕДЕЛЕНИЕ МОДЕЛИ ============

def LeNet5():
    return Sequential(
        Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
        Tanh(),
        AvgPool2d(kernel_size=2, stride=2),
        Conv2d(in_channels=6, out_channels=16, kernel_size=5),
        Tanh(),
        AvgPool2d(kernel_size=2, stride=2),
        Flatten(),
        Linear(in_features=16*5*5, out_features=120),
        Tanh(),
        Linear(in_features=120, out_features=84),
        Tanh(),
        Linear(in_features=84, out_features=10)
    )

lenet5 = LeNet5()

# Пытаемся перенести на GPU
try:
    lenet5.to('gpu')
    print("✅ Модель перенесена на GPU")
except Exception as e:
    print(f"⚠️ Ошибка при переносе на GPU: {e}. Работаем на CPU.")

print(f"Количество параметров: {sum(p.data.size for p in lenet5.parameters())}")

optimizer = Adam(lenet5.parameters(), lr=1e-3)
criterion = CrossEntropyLoss()

# ============ ЦИКЛ ОБУЧЕНИЯ ============

batch_size = 128
n_epochs = 5 # Можно увеличить
n_samples = 2000 # Для быстрой проверки, уберите ограничение для полного обучения

train_loss_history = []
train_accuracy_history = []
test_loss_history = []
test_accuracy_history = []

print("\nНачинаем обучение...")
print("-" * 70)
print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Acc':>9} | {'Test Loss':>10}")
print("-" * 70)

for epoch in range(n_epochs):
    lenet5.to('gpu') # Убеждаемся, что мы на GPU
    
    order = np.random.permutation(min(len(X_train), n_samples))
    epoch_loss = 0
    correct = 0
    
    for i in range(0, len(order), batch_size):
        optimizer.zero_grad()
        
        batch_indices = order[i:i+batch_size]
        X_batch = Tensor(X_train[batch_indices], device='gpu')
        y_batch = Tensor(y_train[batch_indices], device='gpu')
        
        # Forward pass
        preds = lenet5(X_batch)
        loss = criterion(preds, y_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Статистика (ИСПРАВЛЕНО: используем float() для обхода memoryview)
        epoch_loss += float(loss.data)
        
        # Accuracy
        pred_labels = preds.data.argmax(axis=1)
        correct += (pred_labels == y_batch.data).sum()

    train_loss = epoch_loss / (len(order) / batch_size)
    train_acc = float(correct) / len(order)
    
    # ============ ТЕСТИРОВАНИЕ ============
    test_loss = 0
    test_correct = 0
    
    with no_grad():
        # Тестируем по батчам, чтобы не забить память GPU
        test_indices = np.arange(min(len(X_test), 1000)) 
        for j in range(0, len(test_indices), batch_size):
            idx = test_indices[j:j+batch_size]
            X_t = Tensor(X_test[idx], device='gpu')
            y_t = Tensor(y_test[idx], device='gpu')
            
            t_preds = lenet5(X_t)
            t_loss = criterion(t_preds, y_t)
            
            test_loss += float(t_loss.data)
            test_correct += (t_preds.data.argmax(axis=1) == y_t.data).sum()
            
    test_loss /= (len(test_indices) / batch_size)
    test_acc = float(test_correct) / len(test_indices)
    
    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_acc)
    test_loss_history.append(test_loss)
    test_accuracy_history.append(test_acc)
    
    print(f"{epoch+1:6} | {train_loss:10.4f} | {train_acc:9.2%} | {test_acc:9.2%} | {test_loss:10.4f}")

# ============ ВЫВОД ОТЧЕТОВ ============

# 1. Отчет по тензорным ядрам и времени GPU
profiler.report()

# 2. Графики
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(test_loss_history, label='Test Loss')
plt.title('Loss History')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracy_history, label='Train Acc')
plt.plot(test_accuracy_history, label='Test Acc')
plt.title('Accuracy History')
plt.legend()
plt.savefig('training_results.png')
print("\n📊 Графики сохранены в 'training_results.png'")

# 3. Финальная статистика
print("\n" + "="*50)
print("ФИНАЛЬНАЯ СТАТИСТИКА")
print("="*50)
print(f"Лучшая точность на тесте: {max(test_accuracy_history)*100:.2f}%")
print(f"Время работы макс. операции: {max(profiler.stats.values()):.4f} сек")
print("="*50)