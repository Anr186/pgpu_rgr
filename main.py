"""
Обучение LeNet-5 с использованием MyFramework (с обратным распространением)
"""

import random
import numpy as np
import torchvision.datasets
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import cupy as cp

from myframework import Tensor, Sequential, Conv2d, AvgPool2d, Tanh, Flatten, Linear, CrossEntropyLoss, Adam, no_grad

random.seed(0)
np.random.seed(0)

# ============ ЗАГРУЗКА ДАННЫХ ============
print("Загрузка MNIST...")
MNIST_train = torchvision.datasets.MNIST('./', download=True, train=True)
MNIST_test = torchvision.datasets.MNIST('./', download=True, train=False)

X_train = MNIST_train.data.numpy()
y_train = MNIST_train.targets.numpy()
X_test = MNIST_test.data.numpy()
y_test = MNIST_test.targets.numpy()



print(f"Размер обучающей выборки: {len(y_train)}, размер тестовой: {len(y_test)}")

# Показываем пример
plt.figure(figsize=(6, 4))
plt.imshow(X_train[0, :, :], cmap='gray')
plt.title(f"Пример цифры: {y_train[0]}")
plt.savefig('sample_image.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("📸 Пример цифры сохранён как 'sample_image.png'")

print(f"Метка первого примера: {y_train[0]}")

# Нормализация и изменение формы
X_train = X_train.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
X_test = X_test.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0

# Для ускорения возьмём подвыборку
n_samples = 10000
X_train = X_train[:n_samples]
y_train = y_train[:n_samples]

print(f"Форма данных: {X_train.shape}")
X_train_gpu = cp.array(X_train)
y_train_gpu = cp.array(y_train)
X_test_gpu = cp.array(X_test)
y_test_gpu = cp.array(y_test)

# ============ ОПРЕДЕЛЕНИЕ МОДЕЛИ LeNet-5 ============

class LeNet5(Sequential):
    def __init__(self):
        super().__init__(
            Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            Tanh(),
            AvgPool2d(kernel_size=2, stride=2),
            
            Conv2d(in_channels=32, out_channels=128, kernel_size=5, padding=0),
            Tanh(),
            AvgPool2d(kernel_size=2, stride=2),
            
            Flatten(),
            Linear(128 * 5 * 5, 1024),
            Tanh(),
            Linear(1024, 512),
            Tanh(),
            Linear(512, 10)
        )

# ============ ИНИЦИАЛИЗАЦИЯ ============
lenet5 = LeNet5()
print("Модель LeNet5 создана")
print(f"Количество параметров: {sum(p.data.size for p in lenet5.parameters())}")

loss_fn = CrossEntropyLoss()
optimizer = Adam(lenet5.parameters(), lr=1e-3)
# batch_size = 100
batch_size = 4096

# Для хранения истории
test_accuracy_history = []
test_loss_history = []
train_accuracy_history = []
train_loss_history = []

# ============ ОБУЧЕНИЕ ============
print("\nНачинаем обучение...")
print("-" * 70)
print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Acc':>9} | {'Test Loss':>10}")
print("-" * 70)

# Данные для таблицы метрик
metrics_data = []

for epoch in range(30):
    # Обучение
    order = np.random.permutation(len(X_train_gpu))
    
    train_correct = 0
    train_total = 0
    epoch_loss = 0
    n_batches = 0
    
    for start_idx in range(0, len(X_train_gpu), batch_size):
        batch_idx = order[start_idx:start_idx + batch_size]
        
        X_batch = Tensor(X_train_gpu[batch_idx])
        y_batch = Tensor(y_train_gpu[batch_idx])
        
        # Forward pass
        preds = lenet5(X_batch)
        
        # Loss
        loss = loss_fn(preds, y_batch)
        epoch_loss += loss.data.item() if hasattr(loss.data, 'item') else loss.data
        n_batches += 1
        
        # Подсчёт точности
        preds_classes = cp.argmax(preds.data, axis=1)
        train_correct += cp.sum(preds_classes == y_batch.data)
        train_total += len(y_batch.data)
        
        # Backward pass
        optimizer.zero_grad()
        grad = loss_fn.backward()
        lenet5.backward(grad)
        optimizer.step()
    
    train_accuracy = train_correct / train_total
    train_accuracy_history.append(train_accuracy)
    avg_loss = epoch_loss / n_batches
    train_loss_history.append(avg_loss)
    
    # Тестирование
    with no_grad():
        test_preds = lenet5(Tensor(X_test_gpu))
        test_preds_classes = cp.argmax(test_preds.data, axis=1)
        accuracy = cp.mean(test_preds_classes == y_test_gpu).get() 
        test_accuracy_history.append(float(accuracy))
        
        test_loss_val = loss_fn(test_preds, Tensor(y_test_gpu)).data
        if hasattr(test_loss_val, 'get'):
            test_loss_val = test_loss_val.get()
        test_loss_history.append(float(test_loss_val))
    
    train_accuracy_val = float(train_accuracy.get() if hasattr(train_accuracy, 'get') else train_accuracy)
    avg_loss_val = float(avg_loss)

    # Сохраняем метрики для таблицы
    metrics_data.append({
        'Epoch': epoch + 1,
        'Train Loss': round(avg_loss_val, 4),
        'Train Acc (%)': round(train_accuracy_val * 100, 2),
        'Test Acc (%)': round(float(accuracy), 2),
        'Test Loss': round(float(test_loss_val), 4)
    })
    
    print(f"{epoch+1:6d} | {avg_loss_val:10.4f} | {train_accuracy_val*100:8.2f}% | {accuracy*100:8.2f}% | {test_loss_val:10.4f}")

print("-" * 70)

# ============ СОХРАНЕНИЕ МЕТРИК В ФАЙЛ ============
# 1. Сохраняем как таблицу в текстовом формате (training_history.txt)
with open('training_history.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("ИСТОРИЯ ОБУЧЕНИЯ МОДЕЛИ LeNet-5\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc (%)':>12} | {'Test Acc (%)':>12} | {'Test Loss':>10}\n")
    f.write("-" * 80 + "\n")
    for m in metrics_data:
        f.write(f"{m['Epoch']:6d} | {m['Train Loss']:10.4f} | {m['Train Acc (%)']:11.2f}% | {m['Test Acc (%)']:11.2f}% | {m['Test Loss']:10.4f}\n")
    f.write("-" * 80 + "\n\n")
    f.write(f"Лучшая точность на тесте: {max(test_accuracy_history)*100:.2f}% (эпоха {test_accuracy_history.index(max(test_accuracy_history)) + 1})\n")
    f.write(f"Финальная точность на тесте: {test_accuracy_history[-1]*100:.2f}%\n")
    f.write(f"Финальные потери на тесте: {test_loss_history[-1]:.4f}\n")
print("📊 Таблица метрик сохранена как 'training_history.txt'")

# 2. Сохраняем как CSV
df_metrics = pd.DataFrame(metrics_data)
df_metrics.to_csv('training_metrics.csv', index=False)
print("📊 CSV с метриками сохранён как 'training_metrics.csv'")

# ============ ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ ============
with no_grad():
    final_preds = lenet5(Tensor(X_test_gpu)) # Используем GPU данные
    final_preds_classes = cp.argmax(final_preds.data, axis=1)
    final_accuracy = cp.mean(final_preds_classes == y_test_gpu).get()
    print(f"\nФинальная точность на тестовой выборке: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")

# ============ ВИЗУАЛИЗАЦИЯ ============

# График 1: Динамика потерь (Loss) на обучающей выборке
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, 'b-o', linewidth=2, markersize=6, label='Train Loss')
plt.xlabel('Эпоха (Epoch)', fontsize=12)
plt.ylabel('Потери (Loss)', fontsize=12)
plt.title('Динамика потерь на обучающей выборке', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("📈 График потерь сохранён как 'training_loss.png'")

# График 2: Динамика точности
plt.figure(figsize=(10, 5))

# Переносим данные с GPU на CPU для отрисовки
train_acc_cpu = [float(a.get() if hasattr(a, 'get') else a) for a in train_accuracy_history]
test_acc_cpu = [float(a.get() if hasattr(a, 'get') else a) for a in test_accuracy_history]

plt.plot(range(1, len(train_acc_cpu) + 1), [a*100 for a in train_acc_cpu], 
         'g-o', linewidth=2, markersize=6, label='Train Accuracy') # Убрали color='green'
plt.plot(range(1, len(test_acc_cpu) + 1), [a*100 for a in test_acc_cpu], 
         'r-s', linewidth=2, markersize=6, label='Test Accuracy')  # Убрали color='red'
plt.xlabel('Эпоха (Epoch)', fontsize=12)
plt.ylabel('Точность (%)', fontsize=12)
plt.title('Точность модели на обучающей и тестовой выборках', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("📈 График точности сохранён как 'training_accuracy.png'")

# График 3: Совмещённый график потерь и точности
fig, ax1 = plt.subplots(figsize=(12, 5))

# Потери (левая ось)
ax1.set_xlabel('Эпоха (Epoch)', fontsize=12)
ax1.set_ylabel('Потери (Loss)', fontsize=12, color='blue')
# Оставляем только 'b-o', убираем color='blue'
ax1.plot(range(1, len(train_loss_history) + 1), train_loss_history, 'b-o', linewidth=2, label='Train Loss')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True, alpha=0.3)

# Точность (правая ось)
ax2 = ax1.twinx()
ax2.set_ylabel('Точность (%)', fontsize=12, color='red')
# Оставляем только 'r-s', убираем color='red'
ax2.plot(range(1, len(test_acc_cpu) + 1), [a*100 for a in test_acc_cpu], 
         'r-s', linewidth=2, label='Test Accuracy')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('Динамика потерь и точности при обучении LeNet-5', fontsize=14)
fig.tight_layout()
plt.savefig('lenet5_training_results.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("📈 Совмещённый график сохранён как 'lenet5_training_results.png'")

# ============ ВИЗУАЛИЗАЦИЯ ФИЛЬТРОВ ============
def visualize_filters(model, layer_idx=0):
    """Визуализация фильтров свёрточного слоя"""
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Conv2d) and i == layer_idx:
            kernels = layer.W.data  # (out_ch, in_ch, kh, kw)
            n_kernels = min(8, kernels.shape[0])
            
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            axes = axes.flatten()
            
            for j in range(n_kernels):
                kernel = kernels[j, 0].get()
                # Нормализуем для отображения
                kernel_norm = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8)
                axes[j].imshow(kernel_norm, cmap='RdBu', interpolation='nearest')
                axes[j].set_title(f'Фильтр {j+1}', fontsize=10)
                axes[j].axis('off')
            
            for j in range(n_kernels, len(axes)):
                axes[j].axis('off')
            
            plt.suptitle(f'Фильтры свёрточного слоя {i+1}', fontsize=14)
            plt.tight_layout()
            plt.savefig('filters.png', dpi=150, bbox_inches='tight')
            plt.show()
            plt.close()
            print("🔍 Фильтры сохранены как 'filters.png'")
            break

visualize_filters(lenet5, layer_idx=0)

# ============ ВИЗУАЛИЗАЦИЯ ПРЕДСКАЗАНИЙ ============
def show_predictions(model, X, y, num_examples=10):
    """Показывает примеры предсказаний"""
    indices = np.random.choice(len(X), num_examples, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    correct_count = 0
    predictions_list = []
    
    for i, idx in enumerate(indices):
        img = X[idx, 0]
        true_label = y[idx]
        
        with no_grad():
            out = model(Tensor(cp.array(X[idx:idx+1])))
            out_cpu = out.data[0].get()
            pred_label = np.argmax(out.data[0])
            prob = np.max(out.data[0])
        
        if pred_label == true_label:
            correct_count += 1
        
        predictions_list.append({
            'True': true_label,
            'Pred': pred_label,
            'Prob': prob
        })
        
        axes[i].imshow(img, cmap='gray')
        color = 'green' if pred_label == true_label else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', color=color, fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle(f'Примеры предсказаний (правильных: {correct_count}/{num_examples})', fontsize=14)
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Сохраняем список предсказаний в файл
    with open('predictions_list.txt', 'w', encoding='utf-8') as f:
        f.write("Результаты предсказаний:\n")
        f.write("=" * 40 + "\n")
        for p in predictions_list:
            status = "✓" if p['True'] == p['Pred'] else "✗"
            f.write(f"{status} True: {p['True']} → Pred: {p['Pred']} (prob: {p['Prob']:.3f})\n")
    print("🖼️ Примеры предсказаний сохранены как 'predictions.png' и 'predictions_list.txt'")
    
    return correct_count / num_examples

print("\n" + "="*50)
print("ВИЗУАЛИЗАЦИЯ ПРЕДСКАЗАНИЙ")
print("="*50)
prediction_accuracy = show_predictions(lenet5, X_test, y_test, num_examples=10)
print(f"Точность на 10 случайных примерах: {prediction_accuracy*100:.1f}%")

# ============ ФИНАЛЬНАЯ СТАТИСТИКА ============
print("\n" + "="*50)
print("ФИНАЛЬНАЯ СТАТИСТИКА")
print("="*50)
print(f"Лучшая точность на тесте: {max(test_accuracy_history):.4f} ({max(test_accuracy_history)*100:.2f}%) (эпоха {test_accuracy_history.index(max(test_accuracy_history)) + 1})")
print(f"Финальная точность на тесте: {test_accuracy_history[-1]:.4f} ({test_accuracy_history[-1]*100:.2f}%)")
print(f"Финальные потери на тесте: {test_loss_history[-1]:.4f}")
print(f"Количество параметров модели: {sum(p.data.size for p in lenet5.parameters())}")
print("="*50)

# Сохраняем финальную статистику
with open('final_statistics.txt', 'w', encoding='utf-8') as f:
    f.write("ФИНАЛЬНАЯ СТАТИСТИКА ОБУЧЕНИЯ\n")
    f.write("="*50 + "\n")
    f.write(f"Модель: LeNet5\n")
    f.write(f"Датасет: MNIST\n")
    f.write(f"Размер обучающей выборки: {n_samples}\n")
    f.write(f"Размер тестовой выборки: {len(y_test)}\n")
    f.write(f"Количество эпох: {len(train_accuracy_history)}\n")
    f.write(f"Batch size: {batch_size}\n")
    f.write(f"Learning rate: 0.001\n")
    f.write(f"Оптимизатор: Adam\n")
    f.write(f"Функция потерь: CrossEntropyLoss\n")
    f.write(f"\nЛучшая точность на тесте: {max(test_accuracy_history)*100:.2f}%\n")
    f.write(f"Финальная точность на тесте: {test_accuracy_history[-1]*100:.2f}%\n")
    f.write(f"Финальные потери на тесте: {test_loss_history[-1]:.4f}\n")
    f.write(f"Количество параметров: {sum(p.data.size for p in lenet5.parameters())}\n")
print("📄 Финальная статистика сохранена как 'final_statistics.txt'")