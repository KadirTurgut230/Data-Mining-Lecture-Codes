import random
import numpy as np
import math
from scipy.spatial import distance
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
from sklearn.metrics.pairwise import cosine_similarity

def hazirlik():
    matrix = np.zeros((10,5))

    for j in range(5):
        a = random.randint(1, 10)
        b = random.randint(a+1, a+10)
        print(j + 1,'. Özellik Aralığı : ',a,' - ' ,b)
        
        for i in range(10):
            matrix[i][j] = random.randint(a,b)
        
    print('\n',matrix,'\n')
    return matrix
    

def soru_1(matrix):
    mean = np.zeros(5, dtype = float)
    variance = np.zeros(5, dtype = float)
    
    for j in range(5):
        sum = 0
        for i in range(10):
            sum += matrix[i][j]    
        mean[j] = sum/10
        
        sum = 0
        for i in range(10):
            sum += pow(matrix[i][j] - mean[j], 2) 
        variance[j] = sum/(10-1)
        deviation = math.sqrt(variance[j])
        
        print(j + 1,'. Özellik Ortalama : ',end = '')
        print(mean[j],' Varyans : ',round(variance[j], 2), end ='')
        print(' Standart Sapma : ',round(deviation, 2))
        
    return mean, variance


def soru_2(matrix, mean, variance):
    covariance_matrix = np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            if i < j:
                sum = 0
                for k in range(10):
                    sum += (matrix[k][i] - mean[i])*(matrix[k][j] - mean[j])
                covariance_matrix[i][j] = sum/(10-1)
                
            elif i == j:
                covariance_matrix[i][j] = variance[i]    
            else:
                covariance_matrix[i][j] = covariance_matrix[j][i]
                
    print('\n','\n')
    
    print("Kovaryans Matrisi : ")
    print(covariance_matrix)
    
    print('\n','\n')
    cov_matrix = np.cov(matrix, rowvar=False)
    print("NumPy Kovaryans Matrisi : ")
    print(cov_matrix)
    print('\n')
    
    return covariance_matrix
    

def soru_3(matrix, covariance_matrix):
    for a in range(3):
        vector1 = matrix[2*a]
        vector2 = matrix[2*a + 1]
        print()
        print(2*a + 1,'. Vector : ', vector1)
        print(2*a + 2,'. Vector : ', vector2)
        
        sum = 0
        for j in range(5):
            sum += pow(vector1[j] - vector2[j], 2)
        euclid_dist = math.sqrt(sum)

        euclidean_dist = distance.euclidean(vector1, vector2)
        print('Euclidean Mesafe : ',euclid_dist, euclidean_dist)
        
        
        sum = 0
        for j in range(5):
            sum += abs(vector1[j] - vector2[j])
        manhattan_dist = sum
    
        manhattan_dist2 = distance.cityblock(vector1, vector2)
        print('Manhattan Mesafesi : ',manhattan_dist, manhattan_dist2)


        cov_matrix_inv = inv(covariance_matrix)
        difference = vector1 - vector2
        mahalonobis = np.sqrt(difference.T @ cov_matrix_inv @ difference)

        mahalonobis2 = mahalanobis(vector1, vector2, cov_matrix_inv)
        print('Mahalonobis Mesafesi : ', mahalonobis, mahalonobis2)


        sum1, sum2, sum3 = 0, 0, 0
        for j in range(5):
            sum1 += vector1[j] * vector2[j]
            sum2 += vector1[j] * vector1[j]
            sum3 += vector2[j] * vector2[j]
        sum2 = math.sqrt(sum2)
        sum3 = math.sqrt(sum3)
        cosine = sum1 / (sum2 * sum3)
        
        cosine2 = cosine_similarity([vector1], [vector2])
        print("Cosine Benzerliği: ",cosine, cosine2[0][0])


matrix = hazirlik()
mean, variance = soru_1(matrix)
covariance_matrix = soru_2(matrix, mean, variance)
soru_3(matrix, covariance_matrix)