import numpy as np
import operator as op
import matplotlib.pyplot as plt
import difflib



Matrix4D="test.csv"
Matrix5D='5DMatrix.csv'
sixHole='6Hole.csv'
denominator=0
molecular=0
HoleLocation_entire=[]
HoleLocation_verify=[]
HoleLocation=[]
verify_hole_index=3

# 定義模型的間隔、寬度、深度、面積、孔洞數量和地質類型數量等參數
interval = 0.5
W = int(70/interval) # 140
D = int(25/interval) # 50
A = W * D
entire_file= np.loadtxt(Matrix4D, delimiter=",",skiprows=1) 
geo_matrix = entire_file[1:,:] # skip first column
verify_matrix=np.delete(geo_matrix,2,1)
test_hole=geo_matrix[:,verify_hole_index]
Hole_distance=entire_file[0]


Hole = verify_matrix.shape[1]
print('共有',Hole,'個孔')


# 獲取不同數字的數量
unique_numbers = np.unique(geo_matrix)
typenumber = len(unique_numbers)
print('共有',typenumber,'種土讓材質')
# 初始化地質類型的分組數組
group_number = np.zeros(A)
# 從CSV文件中讀取地質矩陣數據

# 定義各個孔洞的位置
location=1
HoleLocation_entire.append(location)
for distance in Hole_distance:
    if distance==1:
        continue
    else:
        location=int(distance/interval)
        HoleLocation_entire.append(location)
print('孔洞位置:',HoleLocation_entire)
HoleLocation_verify=np.delete(HoleLocation_entire,2)
print('驗證孔洞位置:',HoleLocation_verify[3])




# 將地質數據中的類型分組存儲到group_number數組中
# Horizontal has two type of soil at the  first layer
# First matrix?
for i in range(1, 74, 1):
    group_number[i - 1] = 1
for i in range(74, 140, 1):
    group_number[i - 1] = 2

# 將各個孔洞位置的地質類型從geo_matrix中放入group_number數組中
# 橫的一行一行過去
# D=50
for i in range(1, D + 1, 1):
    for j in range(Hole):
        group_number[(HoleLocation_verify[j] - 1) + (i - 1) * W] = verify_matrix[i - 1][j] 
       
# 初始化計算地質類型轉移概率的變量
T_t_V = np.zeros(len(verify_matrix))
soiltype_V = {}
# print(len(geo_matrix)) =50
# print(np.size(geo_matrix)) =250

# 統計各地質類型的出現次數
for i in range(np.size(verify_matrix, 1)):
    for j in range(len(verify_matrix)):
        T_t_V[j] = verify_matrix[j][i]
    for k in T_t_V[0:len(T_t_V)]:
        soiltype_V[k] = soiltype_V.get(k, 0) + 1

# 將地質類型按照類型值排序
soiltype_V = sorted(soiltype_V.items(), key=op.itemgetter(0), reverse=False)

# 初始化積分估計轉移概率矩陣和轉移矩陣
VPCM = np.zeros([len(soiltype_V), len(soiltype_V)])
Tmatrix_V = np.zeros([len(soiltype_V), len(soiltype_V)])

# 計算積分估計轉移概率矩陣和轉移矩陣

for i in range(np.size(verify_matrix, 1)):
    for j in range(len(verify_matrix)):
        T_t_V[j] = verify_matrix[j][i]

    
    # 判斷是否轉化
    for k in range(len(T_t_V) - 1): # 49次
        for m in range(len(soiltype_V)): #3次
            for n in range(len(soiltype_V)):
                if T_t_V[k] == soiltype_V[m][0] and T_t_V[k + 1] == soiltype_V[n][0]:
                    VPCM[m][n] += 1
                    Tmatrix_V[m][n] += 1

# 正規化轉移矩陣
count_V = np.sum(Tmatrix_V, axis=1)
for i in range(np.size(Tmatrix_V, 1)):
    for j in range(np.size(Tmatrix_V, 1)):
        Tmatrix_V[i][j] = Tmatrix_V[i][j] / count_V[i]

# 設置常數K
K = 9.3

# 初始化有權重的積分估計轉移概率矩陣和轉移矩陣
HPCM = np.zeros([len(count_V), len(count_V)])
Tmatrix_H = np.zeros([len(count_V), len(count_V)])



# 計算有權重的積分估計轉移概率矩陣和轉移矩陣
for i in range(np.size(Tmatrix_H, 1)):
    for j in range(np.size(Tmatrix_H, 1)):
        if i == j:
            HPCM[i][j] = K * VPCM[i][j]
            Tmatrix_H[i][j] = K * VPCM[i][j]
        else:
            HPCM[i][j] = VPCM[i][j]
            Tmatrix_H[i][j] = VPCM[i][j]


# 正規化轉移矩陣
count_H = np.sum(Tmatrix_H, axis=1)
for i in range(np.size(Tmatrix_H, 1)):
    for j in range(np.size(Tmatrix_H, 1)):
        Tmatrix_H[i][j] = Tmatrix_H[i][j] / count_H[i]

# 初始化地質類型預測的相關變數
L_state = 0
M_state = 0
Q_state = 0
Nx = 0
a = 0
current_matrix = np.array([[0.0, 0.0, 0.0, 0.0,0.0]])
transitionName = np.array([[1, 2, 3, 4,5]])

conditions = {}
for j in range(1, len(HoleLocation_verify)):
    conditions[(HoleLocation_verify[j - 1], HoleLocation_verify[j])] = HoleLocation_verify[j]

# 進行地質類型的預測
for layer in range(2, D + 1):
    for i in range(1, W + 1):
        L_state = 0
        M_state = 0
        Q_state = 0

        # 遍歷條件範圍，找到滿足條件的範圍
        for (lower, upper), nx in conditions.items():
            if lower < i < upper:
                L_state = group_number[(i - 2) + (layer - 1) * W] - 1
                M_state = group_number[(i - 1) + (layer - 2) * W] - 1
                Q_state = group_number[(nx - 1) + (layer - 1) * W] - 1
                Nx = nx
                break

        if i in HoleLocation_verify:
            a += 1
        else:
            TV = Tmatrix_V
            TH = Tmatrix_H
            Nx_TH = Tmatrix_H
            f_sum = 0
            k_sum = 0
            for N in range(1, Nx - i):
                Nx_TH = np.dot(Nx_TH, Tmatrix_H)
            for f in range(typenumber):
                f_item1 = Tmatrix_H[int(L_state)][f]
                f_item2 = Nx_TH[f][int(Q_state)]
                f_item3 = Tmatrix_V[int(M_state)][f]
                f_sum += f_item1 * f_item2 * f_item3
            for k in range(typenumber):
                k_item1 = Tmatrix_H[int(L_state)][k]
                k_item2 = Nx_TH[k][int(Q_state)]
                k_item3 = Tmatrix_V[int(M_state)][k]
                k_sum = k_item1 * k_item2 * k_item3
                current_matrix[0][k] = k_sum / f_sum
            group_number[(i - 1) + (layer - 1) * W] = np.random.choice(transitionName[0], replace=True, p=current_matrix[0])
# 重塑地質類型分組數組為矩陣
group_matrix = group_number.reshape(D, W)
verify_array=group_matrix[:,HoleLocation_verify[verify_hole_index]]
print(verify_array)
for i,x in zip(test_hole,verify_array):
    if i==x:
        denominator+=1
        molecular+=1
    else:
        denominator+=1
correct_rate=molecular/denominator
print('正確率:',correct_rate*100,"%")


# 可視化地質類型分布
plt.imshow(group_matrix, cmap='tab10', origin='upper')
plt.colorbar(label='Geological Type')
plt.title('test hole pic')
plt.xlabel('Width (units)')
plt.ylabel('Depth ')
plt.show() 