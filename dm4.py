import csv
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

def open_csv(filename, number_of_object, dimension):
    #bu fonksiyon csv dosyasını açar.
    #dataseti ve etiketleri okur.
    dataset = np.zeros((number_of_object, dimension))
    labels = number_of_object * [None]
    
    with open(filename, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        i = 0
        for row in csv_reader:
            if i != 0 :
                for j in range(1, 5):
                    dataset[i - 1][j - 1] = row[j]
                labels[i - 1] = row[5] 
            i += 1 
            
    return dataset, labels


def k_means(X, y):
    #bu fonksiyon K-Means modeli oluşturur ve eğitir.
    results = []
    for k in [2, 3, 4]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        # SSE hesaplama
        sse = kmeans.inertia_
        # MSE = SSE / number of sample
        mse = sse / len(X)
        # Örtüşme (Accuracy) Hesaplama
        predicted_labels = kmeans.labels_
        predicted = np.zeros_like(predicted_labels)
        
        for i in range(k):
            mask = (predicted_labels == i)
            if np.sum(mask) > 0:
                # Gerçek etiketler alınır.
                true_labels_in_cluster = y[mask]
                #Modelin oluşturduğu kümedeki objelerin gerçek sınıf değerlerine
                #bakılır. Frekansı en yüksek sınıf, kümenin sınıfı seçilir.
                counts = np.bincount(true_labels_in_cluster)
                most_frequent = np.argmax(counts)
                predicted[mask] = most_frequent
          
        #Kümenin içindeki objelerin gerçek sınıf değerinin, 
        #modelin seçtiği küme sınıfı ile örtüşmesinin oranı 
        accuracy = accuracy_score(y, predicted) * 100
        
        results.append({
            'k': k,
            'SSE': sse,
            'MSE': mse,
            'Accuracy (%)': accuracy
        })

    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))


filename = 'Iris.csv'
number_of_object = 150
dimension = 4 

dataset, labels = open_csv(filename, number_of_object, dimension)
#labels dizisindeki string değerleri eşsiz(unique) integer'lara dönüştürülür.
#bu değerler daha sonra y kümesine atanır.  
unique_labels = np.unique(labels)
label_map = {label: i for i, label in enumerate(unique_labels)}
y = np.array([label_map[label] for label in labels])
#k_means fonksiyonunun çağrılması
k_means(X = dataset, y= y)