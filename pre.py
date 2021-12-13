import numpy as np 
# K:  number of cluster
K = 3




def readFile(fileName):
    fileObj = open(fileName, "r") #opens the file in read mode
    words = fileObj.read().splitlines() #puts the file into an array
    fileObj.close()
    return words

#   Hàm chuẩn hóa   list    ->  list
def nominal(arr):
    val_min = min(arr)
    val_max = max(arr)
    
    tmp = arr
    for i in range(len(arr)):
        tmp[i] = (arr[i] - val_min)/(val_max - val_min)
    return tmp
    

class Dimension:
    #   chỉ sử dụng đối với dimension có variance max 
    arr_sort1 = []
    arr_sort2 = []
    arr_sort3 = []
    init_median1 = 0
    init_median2 = 0
    init_median3 = 0
    
    
    def __init__(self, name, arr):
        self.data_origin = arr
        self.data_nominal = nominal(arr)
        self.name = name
        
        self.variance = np.var(self.data_nominal)
        
        self.data_sort = self.data_nominal
        self.data_sort.sort()
        self.divideAfterSort()
    
    #   chia cụm ban đầu vs khởi tạo trung tâm cụm (median)
    def divideAfterSort (self):
        for i in range(50):
            self.arr_sort1.append(self.data_sort[i])
            self.arr_sort2.append(self.data_sort[i+50])
            self.arr_sort3.append(self.data_sort[i+100])
        self.init_median1 = np.median(self.arr_sort1)
        self.init_median2 = np.median(self.arr_sort2)
        self.init_median3 = np.median(self.arr_sort3)
        pass

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

for i in range(len(lines)):
    lines[i] = lines[i].split(',')
    arr_se_length.append(float(lines[i][0]))
    arr_se_width.append(float(lines[i][1]))
    arr_pe_length.append(float(lines[i][2]))
    arr_pe_width.append(float(lines[i][3]))


# # test xem co miss data ko
# print ('check length each dimension:')
# print (len(arr_se_length))
# print (len(arr_pe_length))
# print (len(arr_se_width))
# print (len(arr_pe_width))

# print (nominal(arr_se_length))
# print (nominal(arr_se_width))


# print (np.var(arr_se_length))
# print (np.var(arr_se_width))

# here



#   list of 4 dimension
list_dms = []       #   list of dimension
list_dms.append(Dimension('sepal_length', arr_se_length))
list_dms.append(Dimension('sepal_width', arr_se_width))
list_dms.append(Dimension('petal_length', arr_pe_length))
list_dms.append(Dimension('petal_width', arr_pe_width))

# print(list_dms[3].name)
# print(list_dms[0].variance)
maxVariance = max(list_dms[0].variance, list_dms[1].variance, list_dms[2].variance, list_dms[3].variance)

dms_maxVar = 0
# dms_maxVar_Name = 0

for i in list_dms:
    print(i.variance)

j = 0
for i in list_dms:
    if (i.variance == maxVariance):
        dms_maxVar = j
        break
    j += 1
    
print("max variance: " + list_dms[dms_maxVar].name)
# print(list_dms[dms_maxVar].data_sort)

print("=================")
# for i in range(len(list_dms[dms_maxVar].data_sort)):
#     print(list_dms[dms_maxVar].data_sort[i])
    
print()

print("3 trung taam cum ban dau cua " + list_dms[dms_maxVar].name + " (cos variance max):")
print(list_dms[dms_maxVar].init_median1)
print(list_dms[dms_maxVar].init_median2)
print(list_dms[dms_maxVar].init_median3)