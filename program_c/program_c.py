import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor
import time
import matplotlib.pyplot as plt


# Функция для загрузки и преобразования данных
def load_and_preprocess_data(file):
    df = pd.read_csv(file)
    df = df.dropna(subset=["Creatinine_mean", "HCO3_mean"])  # Удаление Nan значений
    data = df[["Creatinine_mean", "HCO3_mean"]].values

    # Масштабирование данных
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)


# Функция для расчета индекса Данна
def dunn_index(clusters, centroids):
    # Расчет минимального межкластерного расстояния
    min_intercluster_distance = np.min([
        np.linalg.norm(centroids[i] - centroids[j])
        for i in range(len(centroids))
        for j in range(i + 1, len(centroids))
    ])
    # Расчет максимального внутрикластерного расстояния
    max_intracluster_distance = np.max([
        np.max([np.linalg.norm(point - centroids[i]) for point in cluster])
        for i, cluster in enumerate(clusters)
    ])
    return min_intercluster_distance / max_intracluster_distance


# Функция для многопоточного расчета
def evaluate_dunn_index_with_threads(data, labels, centroids, n_threads):
    unique_labels = np.unique(labels)
    clusters = [data[labels == label] for label in unique_labels]

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        future = executor.submit(dunn_index, clusters, centroids)

    return future.result()


# Функция для выполнения кластеризации и оценки
def perform_clustering_and_evaluation(data, n_clusters_list, thread_counts):

    for n_clusters in n_clusters_list:
        print(f"Обработка {n_clusters} кластеров")
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        kmeans.fit(data)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        # Вывод центроидов
        print(f"Центроиды для {n_clusters} кластеров: \n{centroids}")

        for n_threads in thread_counts:
            start_time = time.time()
            dunn = evaluate_dunn_index_with_threads(data, labels, centroids, n_threads)
            elapsed_time = time.time() - start_time

            print(f"Время выполнения программы, {n_threads} потока/ов: {elapsed_time:.5f} сек."
                  f" | Индекс Данна: {dunn}")

        # Визуализация кластеров
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow')
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='black')
        plt.title(f'Кластеризация K-Means с {n_clusters} кластерами')
        plt.xlabel('Creatinine_mean (средний креатинин)')
        plt.ylabel('HCO3_mean (средний HCO3)')
        result_filename = f"cluster_{n_clusters}_visualization.png"
        plt.savefig(result_filename, dpi=200)
        print(f"Визуализация кластеризации сохранена как: {result_filename}\n")


# Основная функция программы
def program_c():
    csv_file_path = "BD-Patients.csv"
    n_clusters_list = [3, 4, 5]
    thread_counts = [2, 4, 6, 8, 10, 12, 14, 16]

    print(f"Начало работы Программы Б...\n")
    data = load_and_preprocess_data(csv_file_path)
    perform_clustering_and_evaluation(data, n_clusters_list, thread_counts)


if __name__ == '__main__':
    program_c()
