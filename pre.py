import numpy as np
from numpy.core.function_base import linspace
from numpy.lib.function_base import average
from k_mean import MyKmeans
import matplotlib.pyplot as plt
import timeit


# from main import kmeans_display

# K:  number of cluster
K = 3
# N: number of pattern
N = 150

def readFile(fileName):
    fileObj = open(fileName, "r") #opens the file in read mode
    words = fileObj.read().splitlines() #puts the file into an array
    fileObj.close()
    return words

#   Hàm chuẩn hóa   list    ->  list
def normalize(arr):
    val_min = min(arr)
    val_max = max(arr)

    tmp = arr
    for i in range(len(arr)):
        tmp[i] = (arr[i] - val_min)/(val_max - val_min)
    return tmp

def kmeans_display(X, label, title = 'title'):
        K = np.amax(label) + 1
        X0 = X[label == 0, :]
        X1 = X[label == 1, :]
        X2 = X[label == 2, :]
        
        # print ("type: " + str(type(X0)))

        plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
        plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
        plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)

        plt.axis('equal')
        plt.plot()
        
        plt.title(title)

        plt.show()



text_file = open("bezdekIris.data", "r")
lines = text_file.readlines()
text_file.close()
#       ===============================================================================================================================
#    1. sepal length in cm
#    2. sepal width in cm
#    3. petal length in cm
#    4. petal width in cm

arr_se_length = []
arr_se_width = []
arr_pe_length = []
arr_pe_width = []

# Iris-setosa tương ứng 0
# Iris-versicolor tương 1
# Iris-virginica tương ứng 2
arr_label = []

for i in range(len(lines)):
    lines[i] = lines[i].split(',')
    arr_se_length.append(float(lines[i][0]))
    arr_se_width.append(float(lines[i][1]))
    arr_pe_length.append(float(lines[i][2]))
    arr_pe_width.append(float(lines[i][3]))

    if ("Iris-setosa" in lines[i][4]):
        arr_label.append(0)
    elif ("Iris-versicolor" in lines[i][4]):
        arr_label.append(1)
    else:
        arr_label.append(2)


    lines[i][0] = float(lines[i][0])
    lines[i][1] = float(lines[i][1])
    lines[i][2] = float(lines[i][2])
    lines[i][3] = float(lines[i][3])

# for i in lines:   
#     print(i)


arr_se_length_normalize = normalize(arr_se_length)
arr_se_width_normalize = normalize(arr_se_width)
arr_pe_length_normalize = normalize(arr_pe_length)
arr_pe_width_normalize = normalize(arr_pe_width)

# data after normalize (n, 4)
data_normalize = []
# (n,5)
data_normalize_wLabel = []

for i in range(N):
    tmp  = []
    tmp_label= []
    tmp.extend([arr_se_length_normalize[i], arr_se_width_normalize[i], arr_pe_length_normalize[i], arr_pe_width_normalize[i]])
    tmp_label = tmp.copy()
    tmp_label.append(arr_label[i])
    data_normalize.append(tmp)
    data_normalize_wLabel.append(tmp_label)

# for i in range(N):
#     # print (lines[i])
#     print (data_normalize_wLabel[i])
#     # print (data_normalize_wLabel[i])

start_kmean = timeit.default_timer()


# run Kmean
print('*'*50)
print('Run Kmean')
print(np.array(data_normalize).shape)
my_k_mean = MyKmeans(num_clusters = 3)
(centroids, labels) = my_k_mean.fit(np.array(data_normalize))

# print (labels)

print()
print('Centers found by our algorithm:')

print(centroids,"\n")
labels = np.array(labels)

print(labels.shape)

print("\n50 first patterns:")
print(labels[:50],"\n")
print("50 second patterns:")
print(labels[50:100],"\n")
print("50 last patterns:")
print(labels[100:],"\n")


stop_kmean = timeit.default_timer()
start_sk = timeit.default_timer()


# Test Kmean sklearn
print('*'*50)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(np.array(data_normalize))
print('Centers found by scikit-learn:')
print(kmeans.cluster_centers_)
pred_label = kmeans.predict(np.array(data_normalize))
pred_label = np.array(pred_label)
print(pred_label.shape)
print("\n50 first patterns:")
print(pred_label[:50])
print("\n50 second patterns:")
print(pred_label[50:100])
print("\n50 last patterns:")
print(pred_label[100:])

print("\n\n================")

stop_sk = timeit.default_timer()



# s0 = 0
# s1 = 0
# s2 = 0
# print(len(pred_label))
# for i in range(len(labels)):
#     if i < 50 and labels[i] == 0:
#         s0 += 1
#     elif 50 <= i and i < 100 and labels[i] == 1:
#         s1 += 1
#     elif 100<= i and i < 150 and labels[i] == 2:
#         s2 += 1
        
# print(s0)
# print(s1)
# print(s2)


# print("label:")
# print(arr_label)

from sklearn.metrics import accuracy_score

print ("\n\nAccuracy score: ", accuracy_score(arr_label, labels) ,"\n\n")


print ("\n\nAccuracy score by sklearn: ", accuracy_score(arr_label, pred_label) ,"\n\n")

print ("Total run time of our algorithm: ", stop_kmean - start_kmean)
print ("\nTotal run time of Sklearn: ", stop_sk - start_sk , "\n\n")
# print (labels)

# kmeans_display(X, labels, 'our kmeans')
