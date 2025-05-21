import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Вхідні дані
powers = [5.43, 6.01, 7.35, 8.07, 9.89]
x = np.arange(len(powers)) # x = [0, 1, 2, 3, 4]
y = np.array(powers)
# Кількість точок і інтервалів
n = len(x)
n_intervals = n - 1
# Крок 1: Обчислення різниці між точками по x
h = np.zeros(n_intervals)
for i in range(n_intervals):
 h[i] = x[i+1] - x[i]
# Крок 2: Формування системи рівнянь для обчислення других похідних
# Створюємо трьохдіагональну матрицю A і вектор правих частин b
A = np.zeros((n, n))
b = np.zeros(n)
# Встановлюємо граничні умови для природного сплайна: друга похідна = 0 на кінцях
A[0, 0] = 1.0
A[n-1, n-1] = 1.0
# Заповнюємо матрицю системи та вектор правих частин
for i in range(1, n-1):
 A[i, i-1] = h[i-1]
 A[i, i] = 2 * (h[i-1] + h[i])
 A[i, i+1] = h[i]

 b[i] = 3 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
# Крок 3: Розв'язання системи для отримання других похідних (c коефіцієнтів)
c = np.linalg.solve(A, b)
# Крок 4: Обчислення інших коефіцієнтів сплайна
a = np.zeros(n_intervals)
b = np.zeros(n_intervals)
d = np.zeros(n_intervals)
for i in range(n_intervals):
 a[i] = y[i]
 b[i] = (y[i+1] - y[i]) / h[i] - h[i] * (2 * c[i] + c[i+1]) / 3
 d[i] = (c[i+1] - c[i]) / (3 * h[i])
# Побудова гладкої лінії
x_smooth = np.linspace(x[0], x[-1], 500)
y_smooth = np.zeros_like(x_smooth)
for i in range(n_intervals):
 idx = (x_smooth >= x[i]) & (x_smooth <= x[i+1])
 dx = x_smooth[idx] - x[i]
 y_smooth[idx] = a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
# Виведення інформації про систему рівнянь
print("\n=== Інформація про систему рівнянь ===")
print("\nМатриця системи A:")
print(A)
print("\nВектор правих частин b:")
print(b)
print("\nРозв'язок системи (другі похідні c):")
print(c)
# Виведення інформації про інтервали
print("\n=== Інформація про інтервали ===")
print("\nДовжини інтервалів h:")
print(h)
# Виведення коефіцієнтів сплайна
print("\n=== Коефіцієнти кубічного сплайна ===")
spline_table = pd.DataFrame({
 'Інтервал': [f"[{x[i]}, {x[i+1]}]" for i in range(n_intervals)],
 'a': a,
 'b': b,
 'c': c[:-1],
 'd': d
})
print(spline_table.to_string(index=False))
# Виведення значень сплайна в проміжних точках
print("\n=== Значення сплайна в проміжних точках ===")
x_test = np.linspace(x[0], x[-1], 10) # 10 проміжних точок
y_test = np.zeros_like(x_test)
for i in range(n_intervals):
 idx = (x_test >= x[i]) & (x_test <= x[i+1])
 dx = x_test[idx] - x[i]
 y_test[idx] = a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
test_points = pd.DataFrame({
 'x': x_test,
 'y': y_test
})
print(test_points.to_string(index=False))
# Побудова графіка
plt.figure(figsize=(8, 5))
plt.plot(x_smooth, y_smooth, label='Кубічний сплайн', linewidth=2)
plt.plot(x, y, 'o', label='Задані точки (powers)', markersize=8)
plt.title('Інтерполяція природним кубічним сплайном')
plt.xlabel('Індекс')
plt.ylabel('Power')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
