import numpy as np
import matplotlib.pyplot as plt

# Задані значення дискретного сигналу
y_values = np.array([2.80, 2.94, 3.20, 3.38, 3.53, 3.75])
n = len(y_values)

# Створення рівномірної сітки точок на відрізку [-π,π]
x_values = np.linspace(-np.pi, np.pi, n)


# Частина 1: Апроксимація квадратичним поліномом (без вбудованої функції curve_fit)
def mnk_quadratic(x, y):
    """
    Реалізація методу найменших квадратів для квадратичного полінома без використання
    вбудованих функцій (повна реалізація системи рівнянь)
    """
    n = len(x)

    # Обчислення сум для формування системи рівнянь
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x ** 2)
    sum_x3 = np.sum(x ** 3)
    sum_x4 = np.sum(x ** 4)
    sum_xy = np.sum(x * y)
    sum_x2y = np.sum(x ** 2 * y)

    # Формування коефіцієнтів системи рівнянь
    # Коефіцієнти a, b, c квадратичного поліному a*x^2 + b*x + c
    # Використовуємо формули з прикладу

    # Обчислення a
    a_numerator = (sum_x2y * (sum_x2 * n - sum_x ** 2) -
                   sum_xy * (sum_x3 * n - sum_x * sum_x2) +
                   sum_y * (sum_x3 * sum_x - sum_x2 ** 2))

    a_denominator = (sum_x4 * (sum_x2 * n - sum_x ** 2) -
                     sum_x3 * (sum_x3 * n - sum_x * sum_x2) +
                     sum_x2 * (sum_x3 * sum_x - sum_x2 ** 2))

    a = a_numerator / a_denominator

    # Обчислення b
    b_numerator = (sum_x4 * (sum_xy * n - sum_y * sum_x) -
                   sum_x3 * (sum_x2y * n - sum_y * sum_x2) +
                   sum_x2 * (sum_x2y * sum_x - sum_xy * sum_x2))

    b_denominator = (sum_x4 * (sum_x2 * n - sum_x ** 2) -
                     sum_x3 * (sum_x3 * n - sum_x * sum_x2) +
                     sum_x2 * (sum_x3 * sum_x - sum_x2 ** 2))

    b = b_numerator / b_denominator

    # Обчислення c
    c_numerator = (sum_x4 * (sum_x2 * sum_y - sum_x * sum_xy) -
                   sum_x3 * (sum_x3 * sum_y - sum_x * sum_x2y) +
                   sum_x2 * (sum_x3 * sum_xy - sum_x2 * sum_x2y))

    c_denominator = (sum_x4 * (sum_x2 * n - sum_x ** 2) -
                     sum_x3 * (sum_x3 * n - sum_x * sum_x2) +
                     sum_x2 * (sum_x3 * sum_x - sum_x2 ** 2))

    c = c_numerator / c_denominator

    return np.array([a, b, c])


def quadratic_func(x, a, b, c):
    return a * x ** 2 + b * x + c


# Знаходження коефіцієнтів квадратичного полінома власним методом найменших квадратів
params = mnk_quadratic(x_values, y_values)
a, b, c = params

# Генерація значень полінома для побудови графіка
x_smooth = np.linspace(-np.pi, np.pi, 100)
y_quadratic = quadratic_func(x_smooth, a, b, c)

# Обчислення значень квадратичного полінома для щільної сітки точок
# Це буде вхідними даними для розрахунку ряду Фур'є
x_dense = np.linspace(-np.pi, np.pi, 50)  # Щільна сітка точок
y_quadratic_dense = quadratic_func(x_dense, a, b, c)


# Частина 2: Апроксимація рядом Фур'є
# Тепер ми наближуємо дискретизований квадратичний поліном, а не оригінальні дані
def calculate_fourier_coefficients(y, x, k_max):
    """
    Обчислення коефіцієнтів Фур'є для дискретного сигналу
    """
    n = len(y)

    # Обчислення a0
    a0 = (2 / n) * np.sum(y)

    # Обчислення ak і bk для k від 1 до k_max
    a_k = []
    b_k = []

    for k in range(1, k_max + 1):
        a = (2 / n) * np.sum(y * np.cos(k * x))
        b = (2 / n) * np.sum(y * np.sin(k * x))
        a_k.append(a)
        b_k.append(b)

    return a0, np.array(a_k), np.array(b_k)


# Функція для обчислення значень ряду Фур'є
def fourier_series(x, a0, a_k, b_k):
    result = a0 / 2  # a0/2 згідно з формулою
    for k in range(len(a_k)):
        result += a_k[k] * np.cos((k + 1) * x) + b_k[k] * np.sin((k + 1) * x)
    return result


# Обчислення коефіцієнтів Фур'є для квадратичного полінома
a0, a_k, b_k = calculate_fourier_coefficients(y_quadratic_dense, x_dense, k_max=5)

# Обчислення значень ряду Фур'є для побудови графіка
y_fourier = fourier_series(x_smooth, a0, a_k, b_k)

# Частина 3: Побудова графіків

# Графік дискретного сигналу
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, color='red', s=50, label='Дискретний сигнал')
plt.grid(True)
plt.title('Дискретний сигнал')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('Дискретний_сигнал.png')

# Графік апроксимації квадратичним поліномом
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, color='red', s=50, label='Дискретний сигнал')
plt.plot(x_smooth, y_quadratic, color='blue', linewidth=2,
         label=f'Квадратичний поліном: {a:.3f}x² + {b:.3f}x + {c:.3f}')
plt.grid(True)
plt.title('Апроксимація дискретного сигналу квадратичним поліномом')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('Апроксимація_квадратичним_поліномом.png')

# Графік апроксимації рядом Фур'є
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, color='red', s=50, label='Дискретний сигнал')
plt.plot(x_smooth, y_quadratic, color='blue', linewidth=2,
         label=f'Квадратичний поліном: {a:.3f}x² + {b:.3f}x + {c:.3f}')
plt.plot(x_smooth, y_fourier, color='green', linewidth=2, label='Ряд Фур\'є')
plt.grid(True)
plt.title('Наближення квадратичного полінома рядом Фур\'є')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('Наближення_рядом_Фур\'є.png')

# Частина 4: Обчислення середніх абсолютних похибок
# Обчислення значень квадратичного полінома в точках x_values
y_quadratic_points = quadratic_func(x_values, a, b, c)

# Обчислення значень ряду Фур'є в точках x_values
y_fourier_points = fourier_series(x_values, a0, a_k, b_k)

# Обчислення середніх абсолютних похибок
mae_quadratic = np.mean(np.abs(y_values - y_quadratic_points))
mae_fourier_polynomial = np.mean(np.abs(y_quadratic_points - y_fourier_points))

# Виведення результатів
print(f"\nАпроксимаційний поліном, у якому коефіцієнти визначено за методом найменших квадратів:")
print(f"f(x) = {a:.3f}x² + {b:.3f}x + {c:.3f}")
print(f"\nКоефіцієнт a0 ряду Фур'є: {a0:.3f}")
print("Коефіцієнти ak ряду Фур'є:")
for i, coef in enumerate(a_k):
    if i < 10:  # Виводимо перші 10 коефіцієнтів для компактності
        print(f"a{i + 1} = {coef:.3f}")
print("Коефіцієнти bk ряду Фур'є:")
for i, coef in enumerate(b_k):
    if i < 10:  # Виводимо перші 10 коефіцієнтів для компактності
        print(f"b{i + 1} = {coef:.3f}")

print(f"\nСередня абсолютна похибка квадратичної апроксимації: {mae_quadratic:.3f}")
print(f"Середня абсолютна похибка наближення квадратичного полінома рядом Фур'є: {mae_fourier_polynomial:.3f}")

# Графік порівняння похибок
plt.figure(figsize=(8, 6))
methods = ['Квадратичний поліном', 'Ряд Фур\'є']
errors = [mae_quadratic, mae_fourier_polynomial]
bars = plt.bar(methods, errors, color=['blue', 'green'])
plt.title('Порівняння середніх абсолютних похибок')
plt.ylabel('Середня абсолютна похибка')
plt.grid(axis='y')

# Додавання значень похибок над стовпцями
for bar, error in zip(bars, errors):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
             f'{error:.3f}', ha='center', va='bottom')

plt.savefig('Порівняння_похибок.png')

# Аналіз відхилень ряду Фур'є від квадратичного полінома
plt.figure(figsize=(10, 6))
plt.plot(x_smooth, np.abs(y_quadratic - y_fourier), color='purple', linewidth=2)
plt.grid(True)
plt.title('Відхилення ряду Фур\'є від квадратичного полінома')
plt.xlabel('x')
plt.ylabel('Абсолютна похибка')
plt.savefig('Відхилення_ряду_Фур\'є.png')

# Додаємо цей рядок в кінці, щоб графіки залишались відкритими
plt.show()
