import csv
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
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
               ml_algorithm):
    # ml_algorithm = Kulanılacak algoritmanın adı 
    # hangi sınıflandırıcı model kullanılacaksa, o model oluşturulur. 
    # Model eğitilir. Test verisinin gerçek etiketleri ile 
    # test verisinin tahmin edilen etiketleri kıyaslanır. 
    # Accuracy(başarı oranı) fonksiyon çıktısı olarak döndürülür.
    
    if ml_algorithm == 'random forest' :
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42)
        
        model.fit(train_dataset, train_labels)
        # Test setinde tahmin yap
        test_pred = model.predict(test_dataset) 
        # Accuracy'yi hesapla
        accuracy = accuracy_score(test_labels, test_pred) 
       
          
    elif ml_algorithm == 'xgboost' :
        le = LabelEncoder()
        # Eğitim etiketlerini öğren ve dönüştür
        train_labels_encoded = le.fit_transform(train_labels) 
        # Test etiketlerini sadece dönüştür (eğitimde öğrenilen sınıflara göre)
        test_labels_encoded = le.transform(test_labels)
        
        model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        random_state=42,
        eval_metric='logloss' # Hata almamak için metrik belirtiyoruz
        )

        model.fit(train_dataset, train_labels_encoded)
        test_pred = model.predict(test_dataset)
        accuracy = accuracy_score(test_labels_encoded, test_pred)

    return accuracy 


def callClassifier(splits, k_split):
    # splits : k fold için oluşturulan index kümeleri
    # k_split : verinin kaç folda ayrıldığı (3 veya 5) 
    
    # splits kulanılarak eğitim seti ve test seti ayrılır.
    # sınıflandırıcılar teker teker çağrılır. 
    # Her sınıflandırıcının accuracy değerlerinin ortalaması alınır.
    # Accuracy değerleri tek tek yazdırılır.
    
    random_forest_accuracy = 0
    xgboost_accuracy = 0
    
    for i in range(k_split):
        train_index, test_index = splits[i]
        
        train_dataset = dataset [train_index]
        test_dataset = dataset [test_index]
        
        train_labels = [labels[l] for l in train_index]
        test_labels = [labels[l] for l in test_index]
        

        random_forest_accuracy += classifier(train_dataset, test_dataset,
                                 train_labels, test_labels, 
                       ml_algorithm='random forest')
        
        xgboost_accuracy += classifier(train_dataset, test_dataset,
                                 train_labels, test_labels, 
                       ml_algorithm='xgboost')
        
        
    random_forest_accuracy /= k_split
    xgboost_accuracy /= k_split
    
    
    print('K-Fold = ', k_split, ' Accuracy Değerleri', '\n')
  
    print('Random Forest = ', random_forest_accuracy)
    print('XgBoost = ', xgboost_accuracy)

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
