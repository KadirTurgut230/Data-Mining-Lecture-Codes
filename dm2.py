import csv
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score


def open_csv(filename, number_of_object, dimension):
    # bu fonksiyon csv dosyasını açar.
    # dataseti ve etiketleri okur.
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


def kFoldSplits(dataset, labels, k_split):
    #k fold oluşturulur, sınıf dağılımını korur.
    skf = StratifiedKFold(n_splits=k_split, shuffle=True, random_state=42)
    splits = list(skf.split(dataset, labels))

    return splits
    

def classifier(train_dataset, test_dataset, train_labels, test_labels, 
               ml_algorithm, k = None, dist_metric = None, kernel = None ):
    
    # ml_algorithm = Kulanılacak algoritmanın adı
    # k = KNN seçildiyse, seçilmesi gereken komşu sayısı
    # dist_metric = uzaklık metriği(öklidyen veya cosine)
    # kernel = svm algoritması seçildiyse, kernel türünü ifade eder.
    # kernel; lineer, polinomsal veya rbf olabilir
    
    # hangi sınıflandırıcı model kullanılacaksa, o model oluşturulur. 
    # Model eğitilir. Test verisinin gerçek etiketleri ile 
    # test verisinin tahmin edilen etiketleri kıyaslanır. 
    # Accuracy(başarı oranı) fonksiyon çıktısı olarak döndürülür.
    
    if ml_algorithm == 'kNN' :
        model = KNeighborsClassifier(
            n_neighbors = k,
            weights = 'uniform',
            algorithm = 'auto',  
            metric = dist_metric ,
            )
    
    elif ml_algorithm == 'naive bayes' :
        model = GaussianNB()
        
        
    elif ml_algorithm == 'decision tree' :
        model = DecisionTreeClassifier(
        criterion='gini',           
        max_depth=None,             
        min_samples_split=2,        
        min_samples_leaf=1,         
        max_features=None,          
        random_state=42,            
        ccp_alpha=0.0               
        )

    elif ml_algorithm == 'svm' :
        
        if kernel == 'linear' : 
            model = LinearSVC(
            C=1.0,
            random_state=42,
            max_iter=1000,
            dual=False #n_samples > n_features
            )
            
        elif kernel == 'poly' :
            model = SVC(
            kernel=kernel,             # Polynomial kernel
            degree=3,                  # Polinom derecesi
            C=1.0,                     # Regularization parametresi
            coef0=1.0,                 # Kernel fonksiyonu için sabit terim
            gamma='scale',             # Kernel katsayısı
            random_state=42
        )
    
        elif kernel == 'rbf' :
            # RBF Kernel ile SVM (varsayılan kernel)
            model = SVC(
            kernel=kernel,              # RBF kernel
            C=1.0,                     # Regularization parametresi
            gamma='scale',             # Kernel katsayısı
            random_state=42
            )
    

    # Modeli eğit
    model.fit(train_dataset, train_labels)
    
    # Test setinde tahmin yap
    test_pred = model.predict(test_dataset)
    
    # Accuracy'yi hesapla
    accuracy = accuracy_score(test_labels, test_pred)

    return accuracy 


def callClassifier(splits, k_split):
    # splits : k fold için oluşturulan index kümeleri
    # k_split : verinin kaç folda ayrıldığı (3 veya 5) 
    
    # splits kulanılarak eğitim seti ve test seti ayrılır.
    # sınıflandırıcılar teker teker çağrılır. 
    # Her sınıflandırıcının accuracy değerlerinin ortalaması alınır.
    # Accuracy değerleri tek tek yazdırılır.
    
    accuracy_list = np.zeros(15)
    
    for i in range(k_split):
        train_index, test_index = splits[i]
        
        train_dataset = dataset [train_index]
        test_dataset = dataset [test_index]
        
        train_labels = [labels[l] for l in train_index]
        test_labels = [labels[l] for l in test_index]
        
        # k=1 için Euclidean ve Cosine
        accuracy_list[0] += classifier(train_dataset, test_dataset,
                                 train_labels, test_labels, 
                       ml_algorithm='kNN', k=1, dist_metric='euclidean')
        
        accuracy_list[1] += classifier(train_dataset, test_dataset,
                                 train_labels, test_labels, 
                       ml_algorithm='kNN', k=1, dist_metric='cosine')
        
        # k=3 için Euclidean ve Cosine
        accuracy_list[2] += classifier(train_dataset, test_dataset, 
                                   train_labels, test_labels, 
                       ml_algorithm='kNN', k=3, dist_metric='euclidean') 
        
        accuracy_list[3] += classifier(train_dataset, test_dataset, 
                                   train_labels, test_labels, 
                       ml_algorithm='kNN', k=3, dist_metric='cosine')
        
        # k=5 için Euclidean ve Cosine
        accuracy_list[4] += classifier(train_dataset, test_dataset, 
                                   train_labels, test_labels, 
                       ml_algorithm='kNN', k=5, dist_metric='euclidean') 
          
        accuracy_list[5] += classifier(train_dataset, test_dataset, 
                                   train_labels, test_labels, 
                       ml_algorithm='kNN', k=5, dist_metric='cosine')
        
        # k=9 için Euclidean ve Cosine
        accuracy_list[6] += classifier(train_dataset, test_dataset, 
                                   train_labels, test_labels, 
                       ml_algorithm='kNN', k=9, dist_metric='euclidean')
        
        accuracy_list[7] += classifier(train_dataset, test_dataset, 
                                   train_labels, test_labels, 
                       ml_algorithm='kNN', k=9, dist_metric='cosine')
        
        # k=15 için Euclidean ve Cosine
        accuracy_list[8] += classifier(train_dataset, test_dataset, 
                                   train_labels, test_labels, 
                       ml_algorithm='kNN', k=15, dist_metric='euclidean')
        
        accuracy_list[9] += classifier(train_dataset, test_dataset, 
                                   train_labels, test_labels, 
                       ml_algorithm='kNN', k=15, dist_metric='cosine')
        
        # Diğer algoritmalar
        accuracy_list[10] += classifier(train_dataset, test_dataset, 
                                   train_labels, test_labels, 
                       ml_algorithm='decision tree')
        
        accuracy_list[11] += classifier(train_dataset, test_dataset, 
                                   train_labels, test_labels, 
                       ml_algorithm='naive bayes')
        
        accuracy_list[12] += classifier(train_dataset, test_dataset, 
                                   train_labels, test_labels, 
                       ml_algorithm='svm', kernel='linear')
        
        accuracy_list[13] += classifier(train_dataset, test_dataset, 
                                   train_labels, test_labels, 
                       ml_algorithm='svm', kernel='poly')
        
        accuracy_list[14] += classifier(train_dataset, test_dataset, 
                                   train_labels, test_labels, 
                       ml_algorithm='svm', kernel='rbf')


    accuracy_list = accuracy_list / k_split
    
    print('K-Fold = ', k_split, '\n')
    
    print('kNN (k : 1, distance metric : euclidean) = ', accuracy_list[0])
    print('kNN (k : 1, distance metric : cosine) = ', accuracy_list[1])
    print('kNN (k : 3, distance metric : euclidean) = ', accuracy_list[2])
    print('kNN (k : 3, distance metric : cosine) = ', accuracy_list[3])
    print('kNN (k : 5, distance metric : euclidean) = ', accuracy_list[4])
    print('kNN (k : 5, distance metric : cosine) = ', accuracy_list[5])
    print('kNN (k : 9, distance metric : euclidean) = ', accuracy_list[6])
    print('kNN (k : 9, distance metric : cosine) = ', accuracy_list[7])
    print('kNN (k : 15, distance metric : euclidean) = ', accuracy_list[8])
    print('kNN (k : 15, distance metric : cosine) = ', accuracy_list[9])
    print('Decision Tree = ', accuracy_list[10])
    print('Naive Bayes = ', accuracy_list[11])
    print('SVM (kernel : linear) = ', accuracy_list[12])
    print('SVM (kernel : polynomial) = ', accuracy_list[13])
    print('SVM (kernel : rbf) = ', accuracy_list[14])

    print('\n\n')



filename = 'Iris.csv'
number_of_object = 150
dimension = 4 

dataset, labels = open_csv(filename, number_of_object, dimension)

# hem 3-fold, hem de 5 fold için split'ler oluşturulur.
# Bu oluşan spliter kullanılarak modeller eğitilir.
k_split = 3
splits = kFoldSplits(dataset, labels, k_split)
callClassifier(splits, k_split)

k_split = 5
splits = kFoldSplits(dataset, labels, k_split)
callClassifier(splits, k_split)