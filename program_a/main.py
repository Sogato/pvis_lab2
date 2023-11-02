from PIL import Image
import numpy as np
import cv2
import time
import concurrent.futures


# Функция для обработки изображения: загрузки, вычисления интенсивности, применения порога и проведения эрозии
def process_image(image_path, threshold, erosion_step):
    # Загрузка изображения
    img = Image.open(image_path)
    img_array = np.array(img)

    # Вычисление интенсивности
    intensity = np.sum(img_array, axis=2) / 3

    # Применение порога
    binary_image = np.where(intensity < threshold, 0, 1)

    # Эрозия изображения
    kernel = np.ones((erosion_step, erosion_step), np.uint8)
    eroded_image = cv2.erode(binary_image.astype(np.uint8), kernel, iterations=1)

    # Преобразование результата в изображение
    return Image.fromarray((eroded_image * 255).astype(np.uint8))


# Функция для параллельной обработки изображений
def process_images_concurrently(image_paths, threshold, erosion_step, num_threads):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        return list(executor.map(lambda path: process_image(path, threshold, erosion_step), image_paths))


# Основная функция программы
def program_a():
    image_sizes = [(2560, 1920), (3200, 2400), (5120, 3840)]

    # Параметры обработки изображений
    threshold = 150
    erosion_step = 2
    num_threads_list = [2, 4, 6, 8, 10, 12, 14, 16]

    print(f"Начало работы Программы А...")
    for size in image_sizes:
        image_path = f"img_{size[0]}x{size[1]}.jpg"
        image_paths = [image_path] * 3  # Повторяем обработку каждого изображения 3 раза

        print(f"\nОбработка изображения {size[0]}x{size[1]}")

        for num_threads in num_threads_list:
            # Замеряем время начала обработки
            start_time = time.time()
            results = process_images_concurrently(image_paths, threshold, erosion_step, num_threads)
            elapsed_time = time.time() - start_time  # Вычисляем затраченное время

            # Выводим среднее время обработки для каждого числа потоков
            print(f"Среднее время обработки изображения, "
                  f"{num_threads} потока/ов: {elapsed_time / 3:.5f} сек.")

            # Сохранение результата обработки для последнего прогона
            if num_threads == num_threads_list[-1]:
                result_filename = f"img_{size[0]}x{size[1]}_processed.jpg"
                results[0].save(result_filename)
                print(f"Обработанное изображение сохранено как: {result_filename}\n")


if __name__ == "__main__":
    program_a()
