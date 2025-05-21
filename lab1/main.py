import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import json
import sys


# 1. Функція для визначення значення сигналу в точці t
def f(t, T):
    """
    Задана функція сигналу з періодом T.

    Args:
        t: значення аргументу
        T: період функції
    """
    # Обчислюємо половину періоду
    half_T = T / 2

    # Приведення значення t до діапазону [-T/2, T/2]
    t = ((t + half_T) % T) - half_T

    if -half_T < t <= 0:
        return t
    elif 0 < t <= half_T:
        return 2 * t
    else:
        # Цей випадок не повинен виникати після нормалізації
        return 0


# 2. Функції для визначення коефіцієнтів ряду Фур'є
def a0_coefficient(T):
    """
    Обчислення коефіцієнта a0 ряду Фур'є.

    Args:
        T: період функції
    """
    half_T = T / 2
    integral_1, _ = integrate.quad(lambda t: t, -half_T, 0)
    integral_2, _ = integrate.quad(lambda t: 2 * t, 0, half_T)
    return 2 * (integral_1 + integral_2) / T


def an_coefficient(n, T):
    """
    Обчислення коефіцієнта an ряду Фур'є.

    Args:
        n: номер гармоніки
        T: період функції
    """
    half_T = T / 2
    omega = 2 * np.pi / T
    integral_1, _ = integrate.quad(lambda t: t * np.cos(n * omega * t), -half_T, 0)
    integral_2, _ = integrate.quad(lambda t: 2 * t * np.cos(n * omega * t), 0, half_T)
    return 2 * (integral_1 + integral_2) / T


def bn_coefficient(n, T):
    """
    Обчислення коефіцієнта bn ряду Фур'є.

    Args:
        n: номер гармоніки
        T: період функції
    """
    half_T = T / 2
    omega = 2 * np.pi / T
    integral_1, _ = integrate.quad(lambda t: t * np.sin(n * omega * t), -half_T, 0)
    integral_2, _ = integrate.quad(lambda t: 2 * t * np.sin(n * omega * t), 0, half_T)
    return 2 * (integral_1 + integral_2) / T


# 3. Функція для наближення сигналу рядом Фур'є
def fourier_approximation(t, N, T):
    """
    Наближення сигналу рядом Фур'є з N членами.

    Args:
        t: значення аргументу
        N: кількість членів ряду
        T: період функції

    Returns:
        Значення наближення в точці t
    """
    omega = 2 * np.pi / T
    a0 = a0_coefficient(T)
    result = a0 / 2  # Перший член ряду

    for n in range(1, N + 1):
        an = an_coefficient(n, T)
        bn = bn_coefficient(n, T)
        result += an * np.cos(n * omega * t) + bn * np.sin(n * omega * t)

    return result


# 4. Функції для обчислення похибок
def mean_absolute_error(N, T, num_points=1000):
    """
    Обчислення середньої абсолютної похибки наближення.

    Args:
        N: кількість членів ряду Фур'є
        T: період функції
        num_points: кількість точок для оцінки похибки

    Returns:
        Середня абсолютна похибка
    """
    half_T = T / 2
    t_values = np.linspace(-half_T, half_T, num_points)
    error_sum = 0

    for t in t_values:
        original = f(t, T)
        approximation = fourier_approximation(t, N, T)
        error_sum += abs(original - approximation)

    return error_sum / num_points


def mean_square_error(N, T, num_points=1000):
    """
    Обчислення середньої квадратичної похибки наближення.

    Args:
        N: кількість членів ряду Фур'є
        T: період функції
        num_points: кількість точок для оцінки похибки

    Returns:
        Середня квадратична похибка
    """
    # Оптимізація: обчислюємо коефіцієнти заздалегідь, щоб не перераховувати їх для кожної точки
    a0 = a0_coefficient(T)
    a_coeffs = [ an_coefficient(n, T) for n in range(1, N + 1) ]
    b_coeffs = [ bn_coefficient(n, T) for n in range(1, N + 1) ]

    half_T = T / 2
    omega = 2 * np.pi / T
    t_values = np.linspace(-half_T, half_T, num_points)
    error_sum = 0

    for t in t_values:
        original = f(t, T)

        # Обчислюємо наближення з використанням попередньо розрахованих коефіцієнтів
        approximation = a0 / 2
        for n in range(1, N + 1):
            approximation += a_coeffs[ n - 1 ] * np.cos(n * omega * t) + b_coeffs[ n - 1 ] * np.sin(n * omega * t)

        error_sum += (original - approximation) ** 2

    return error_sum / num_points


# 5. Функція для збереження результатів у файл
def save_results_to_file(N, T, T_multiplier, filename="fourier_results.json"):
    """
    Збереження результатів аналізу у файл.

    Args:
        N: кількість членів ряду Фур'є
        T: період функції
        T_multiplier: множник π для періоду
        filename: ім'я файлу для збереження результатів
    """
    # Обчислення коефіцієнтів
    a0 = a0_coefficient(T)
    a_coeffs = [ an_coefficient(n, T) for n in range(1, N + 1) ]
    b_coeffs = [ bn_coefficient(n, T) for n in range(1, N + 1) ]

    # Обчислення похибок
    mae = mean_absolute_error(N, T)
    mse = mean_square_error(N, T)

    # Підготовка даних для збереження
    data = {
        "N": N,
        "T": T,
        "T_multiplier": T_multiplier,
        "a0": a0,
        "a_coefficients": a_coeffs,
        "b_coefficients": b_coeffs,
        "mean_absolute_error": mae,
        "mean_square_error": mse
    }

    # Збереження у файл
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Результати збережено у файл {filename}")


# Функція для визначення потужності сигналу
def calculate_signal_power(N, T):
    """
    Обчислення потужності сигналу за аналітичним виразом та рядом Фур'є.

    Args:
        N: кількість членів ряду Фур'є для розрахунку потужності
        T: період функції
    """
    half_T = T / 2
    # Потужність за аналітичним виразом (інтеграл від квадрата функції)
    power_analytical, _ = integrate.quad(lambda t: f(t, T) ** 2, -half_T, half_T)
    power_analytical /= T

    # Потужність за рядом Фур'є (сума квадратів коефіцієнтів)
    a0 = a0_coefficient(T)
    power_fourier = (a0 ** 2) / 4  # Внесок постійної складової

    for n in range(1, N + 1):
        an = an_coefficient(n, T)
        bn = bn_coefficient(n, T)
        power_fourier += (an ** 2 + bn ** 2) / 2  # Додаємо внесок n-ї гармоніки

    print(f"\nПотужність сигналу:")
    print(f"- За аналітичним виразом: {power_analytical:.6f}")
    print(f"- За рядом Фур'є (N={N}): {power_fourier:.6f}")
    print(f"- Відносна похибка: {abs(power_analytical - power_fourier) / power_analytical * 100:.6f}%")


# Функція для форматування міток осі X
def format_pi_ticks(ticks, T_multiplier):
    """
    Форматування міток осі X у вигляді кратних π.

    Args:
        ticks: значення міток
        T_multiplier: множник π для періоду

    Returns:
        Відформатовані підписи міток
    """
    labels = [ ]
    for t in ticks:
        # Нормалізуємо значення відносно π
        pi_multiple = t / np.pi

        # Форматуємо мітку
        if abs(pi_multiple) < 1e-10:  # Близько до нуля
            labels.append('0')
        elif abs(pi_multiple - 1) < 1e-10:  # Близько до π
            labels.append('π')
        elif abs(pi_multiple + 1) < 1e-10:  # Близько до -π
            labels.append('-π')
        elif abs(pi_multiple - round(pi_multiple)) < 1e-10:  # Ціле число * π
            pi_int = int(round(pi_multiple))
            if pi_int == 1:
                labels.append('π')
            elif pi_int == -1:
                labels.append('-π')
            else:
                labels.append(f'{pi_int}π')
        else:
            # Дробове число * π, округлюємо до 2 знаків
            pi_round = round(pi_multiple, 2)
            if pi_round == 0:
                labels.append('0')
            else:
                labels.append(f'{pi_round}π')

    return labels


# Функція для візуалізації сигналу та наближення
def visualize_signal(N, T, T_multiplier, mae, mse, range_factor=2):
    """
    Візуалізація сигналу, наближення та похибок.

    Args:
        N: кількість членів ряду Фур'є
        T: період функції
        T_multiplier: множник π для періоду
        mae: середня абсолютна похибка
        mse: середня квадратична похибка
        range_factor: множник діапазону для графіка
    """
    # Створюємо нове вікно для графіків
    plt.figure(figsize=(12, 8))

    # Налаштування діапазону відображення
    half_T = T / 2
    x_min = -range_factor * half_T
    x_max = range_factor * half_T

    # Підготовка даних для графіків сигналу з вказаним діапазоном
    t_values = np.linspace(x_min, x_max, 1000)
    original_values = [ f(t, T) for t in t_values ]
    approximation_values = [ fourier_approximation(t, N, T) for t in t_values ]

    # Графік сигналу та наближення
    plt.subplot(2, 1, 1)
    plt.plot(t_values, original_values, 'b-', label='Вихідний сигнал', linewidth=2)
    plt.plot(t_values, approximation_values, 'r-', label=f'Ряд Фур\'є (N={N})', linewidth=2)
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.legend(fontsize=10)
    plt.title(
        f'Наближення сигналу рядом Фур\'є (T={T_multiplier}π)\nСередня абс. похибка: {mae:.6f} | Сер. кв. похибка: {mse:.6f}',
        fontsize=12)
    plt.xlabel('t', fontsize=10)
    plt.ylabel('f(t)', fontsize=10)

    # Встановлюємо межі графіка за введеним діапазоном
    plt.xlim(x_min, x_max)

    # Створюємо динамічний набір міток для осі X
    if range_factor <= 3:
        # Для невеликих діапазонів показуємо більше міток
        num_ticks = 7  # Кількість міток (непарне число для симетрії)
        ticks = np.linspace(x_min, x_max, num_ticks)
        labels = format_pi_ticks(ticks, T_multiplier)
    else:
        # Для великих діапазонів показуємо мітки з кроком в половину періоду
        step = T / 2
        ticks = np.arange(x_min, x_max + step, step)
        labels = format_pi_ticks(ticks, T_multiplier)

    plt.xticks(ticks, labels, fontsize=10)
    plt.yticks(fontsize=10)

    # Обчислення похибок для різних N
    max_N = 20
    n_values = list(range(1, max_N + 1))
    mae_values = [ ]
    mse_values = [ ]
    rmse_values = [ ]

    for n in n_values:
        mae_n = mean_absolute_error(n, T)
        mse_n = mean_square_error(n, T)
        rmse_n = np.sqrt(mse_n)

        mae_values.append(mae_n)
        mse_values.append(mse_n)
        rmse_values.append(rmse_n)

    # Графік похибок
    plt.subplot(2, 1, 2)
    plt.plot(n_values, mae_values, 'b-o', label='Середня абсолютна похибка', linewidth=2, markersize=6)
    plt.plot(n_values, mse_values, 'r-o', label='Середня квадратична похибка', linewidth=2, markersize=6)
    plt.plot(n_values, rmse_values, 'g-o', label='Корінь з середньої квадратичної похибки', linewidth=2, markersize=6)
    plt.grid(True)
    plt.xlabel('Кількість членів ряду (N)', fontsize=10)
    plt.ylabel('Похибка', fontsize=10)
    plt.title('Залежність похибок від кількості членів ряду', fontsize=12)
    plt.legend(fontsize=10)
    plt.xticks(np.arange(1, max_N + 1, 1), fontsize=9)
    plt.yticks(fontsize=9)

    plt.tight_layout()

    # Зберігаємо графік у файл для перевірки
    try:
        plt.savefig('fourier_plots.png')
        print("Графіки збережено у файл 'fourier_plots.png'")
    except Exception as e:
        print(f"Не вдалося зберегти графік: {e}")


# 6. Основна програма
def main():
    """
    Основна програма для виконання наближення та візуалізації.
    """
    # Параметри
    N = int(input("Введіть кількість членів ряду Фур'є (N): "))

    # Запит множника π для періоду
    T_multiplier_input = input("Введіть множник π для періоду (наприклад, 2 для періоду 2π): ")
    if T_multiplier_input.strip():
        try:
            T_multiplier = float(T_multiplier_input)
            if T_multiplier <= 0:
                T_multiplier = 2
                print(f"Множник періоду має бути додатнім числом. Використовуємо стандартний період T = 2π")
        except ValueError:
            T_multiplier = 2
            print(f"Неправильний формат. Використовуємо стандартний період T = 2π")
    else:
        T_multiplier = 2
        print(f"Використовуємо стандартний період T = 2π")

    # Обчислюємо фактичний період
    T = T_multiplier * np.pi
    print(f"Період функції: T = {T_multiplier}π = {T:.6f}")

    # Додаємо запит на введення діапазону графіка
    range_factor_input = input(
        f"Введіть множник діапазону для графіка (наприклад, 2 для діапазону від -{T_multiplier}π до {T_multiplier}π): ")
    if range_factor_input.strip():
        try:
            range_factor = float(range_factor_input)
            if range_factor <= 0:
                range_factor = 2
                print(f"Множник діапазону має бути додатнім числом. Використовуємо стандартний множник = 2")
        except ValueError:
            range_factor = 2
            print(f"Неправильний формат. Використовуємо стандартний множник діапазону = 2")
    else:
        range_factor = 2
        print(f"Використовуємо стандартний множник діапазону = 2")

    # Обчислення похибок
    mae = mean_absolute_error(N, T)
    mse = mean_square_error(N, T)
    rmse = np.sqrt(mse)  # Корінь із середньоквадратичної похибки

    print(f"Середня абсолютна похибка: {mae:.6f}")
    print(f"Середня квадратична похибка: {mse:.6f}")
    print(f"Середня квадратична похибка (корінь): {rmse:.6f}")

    # Збереження результатів
    save_results_to_file(N, T, T_multiplier)

    # Перевірка потужності сигналу
    calculate_signal_power(N, T)

    # Візуалізація сигналу та похибок з вказаним діапазоном
    visualize_signal(N, T, T_multiplier, mae, mse, range_factor)

    print("\nГотово! Графіки відображаються.")
    # У випадку, якщо у вас проблеми з відображенням графіків, ця команда буде блокуючою
    plt.show()


if __name__ == "__main__":
    try:
        # Налаштування для matplotlib
        plt.rcParams[ 'figure.max_open_warning' ] = 50  # Підвищуємо ліміт відкритих фігур

        # Перевіряємо бекенд
        print(f"Поточний бекенд matplotlib: {plt.get_backend()}")

        # Запускаємо основну програму
        main()
    except Exception as e:
        print(f"Помилка при виконанні програми: {e}")

