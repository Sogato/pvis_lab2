from PIL import Image
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import convolve


# Функция применяет сдвиг к изображению на заданные значения по оси X и Y.
def shift_image(img, shift_x, shift_y, fill_color=(187, 38, 73)):
    shifted_image = Image.new('RGB', img.size, fill_color)
    shifted_image.paste(img, (shift_x, shift_y))
    return shifted_image


# Функция применяет свертку к изображению
def apply_convolution(img, kernel):
    img_np = np.array(img)
    r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]

    r = convolve(r, kernel, mode='same')
    g = convolve(g, kernel, mode='same')
    b = convolve(b, kernel, mode='same')

    convoluted = np.stack([r, g, b], axis=2).astype(np.uint8)
    return Image.fromarray(convoluted)


# Функция для обработки изображения: сдвиг и свертка
def process_image(file_path, shift_x, shift_y, kernel):
    img = Image.open(file_path)
    img = shift_image(img, shift_x, shift_y)
    img = apply_convolution(img, kernel)
    return img


# Функция для тестирования производительности обработки изображений
def benchmark_processing(image_sizes, shift_x, shift_y, kernel, thread_counts):
    for size in image_sizes:
        file_path = f"img_{size[0]}x{size[1]}.jpg"
        print(f"Обработка изображения {size[0]}x{size[1]}")
        for num_threads in thread_counts:
            times = []
            for _ in range(3):  # Три прогона для каждого изображения
                start_time = time.time()
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    result = executor.submit(process_image, file_path, shift_x, shift_y, kernel).result()
                times.append(time.time() - start_time)

            avg_time = sum(times) / len(times)
            print(f"Среднее время обработки изображения, "
                  f"{num_threads} потока/ов: {avg_time:.5f} сек.")

            # Сохранение результата обработки для последнего прогона
            if num_threads == thread_counts[-1]:  # только для последнего количества потоков
                result_filename = file_path.split(".")[0] + "_processed." + file_path.split(".")[1]
                result.save(result_filename)
                print(f"Обработанное изображение сохранено как: {result_filename}\n")


# Основная функция программы
def program_b():
    # Параметры для запуска
    image_sizes = [(2560, 1920), (3200, 2400), (5120, 3840)]
    shift_x, shift_y = 10, 10  # Сдвиг на 10 пикселей
    # Ядро для размытия, Гауссово размытие
    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 16
    thread_counts = [2, 4, 6, 8, 10, 12, 14, 16]

    print(f"Начало работы Программы Б...\n")
    # Запуск тестирования
    benchmark_processing(image_sizes, shift_x, shift_y, kernel, thread_counts)


if __name__ == '__main__':
    program_b()
