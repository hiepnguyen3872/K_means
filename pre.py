import numpy as np
from numpy.core.function_base import linspace
from k_mean import MyKmeans

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
arr_lable = []

for i in range(len(lines)):
    lines[i] = lines[i].split(',')
    arr_se_length.append(float(lines[i][0]))
    arr_se_width.append(float(lines[i][1]))
    arr_pe_length.append(float(lines[i][2]))
    arr_pe_width.append(float(lines[i][3]))

    if ("Iris-setosa" in lines[i][4]):
        arr_lable.append(0)
    elif ("Iris-versicolor" in lines[i][4]):
        arr_lable.append(1)
    else:
        arr_lable.append(2)

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
    tmp_label.append(arr_lable[i])
    data_normalize.append(tmp)
    data_normalize_wLabel.append(tmp_label)

for i in range(N):
    # print (lines[i])
    print (data_normalize[i])
    # print (data_normalize_wLabel[i])



# run Kmean
print('*'*50)
print('Run Kmean')
print(np.array(data_normalize).shape)
my_k_mean = MyKmeans(num_clusters = 3)
(centroids, labels) = my_k_mean.fit(np.array(data_normalize))
print('Centers found by our algorithm:')
print(centroids)
labels = np.array(labels)
print(labels.shape)
print(labels[:50])
print(labels[50:100])
print(labels[100:])

# Test Kmean sklearn
print('*'*50)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(np.array(data_normalize))
print('Centers found by scikit-learn:')
print(kmeans.cluster_centers_)
pred_label = kmeans.predict(np.array(data_normalize))
pred_label = np.array(pred_label)
print(pred_label.shape)
print(pred_label[:50])
print(pred_label[50:100])
print(pred_label[100:])

