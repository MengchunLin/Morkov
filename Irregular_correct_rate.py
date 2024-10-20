import numpy as np
import pandas as pd
import operator as op
import matplotlib.pyplot as plt
from collections import Counter

# -----------testing file----------------
Matrix4D = "test.csv"
Matrix5D = '5DMatrix.csv'
sixHole = '6Hole.csv'
test='test - 複製.csv'
eightSoil='8soil.csv'
predict_hole=4
# -----------testing file----------------
# file preprocessing
entire_file = pd.read_csv(test, delimiter=",").fillna(0).values
entire_matrix = entire_file[1:, :]  # skip first column

# predict_hole_data = entire_file[2:, predict_hole]
# print('predict_hole_data:', predict_hole_data)

# entire_matrix[:,predict_hole]=0
# print('entire_matrix:',entire_matrix)

Hole_distance = entire_file[0]
Hole_distance=np.delete(Hole_distance,predict_hole)
print('Hole_distance:',Hole_distance)

initial_array = entire_file[1]
initial_array=np.delete(initial_array,predict_hole)

# 取得土壤種類
unique_numbers = np.unique(entire_matrix)

# 從unique_numbers過濾掉0
unique_numbers = unique_numbers[unique_numbers != 0]
typenumber = len(unique_numbers)

# 建立土壤代號對應的數字
mapping = {value: index+1 for index, value in enumerate(unique_numbers)}
def map_value(value):
    return mapping.get(value, value)
entire_matrix = np.vectorize(map_value)(entire_matrix)
predict_hole_data=entire_file[2:,predict_hole]
predict_hole_data=np.vectorize(map_value)(predict_hole_data)
print('predict_hole_data:', predict_hole_data)

entire_matrix[:,predict_hole]=0




transitionName = np.arange(1,typenumber+1)
# file preprocessing


# 定義模型的間隔、寬度、深度、面積、孔洞數量和地質類型數量等參數
interval = 0.5
W = int(Hole_distance.max() / interval) + 1
D = int(entire_matrix.shape[0] ) 

denominator = 0
molecular = 0
A = W * D
HoleLocation_entire=(Hole_distance/interval).astype(int)
HoleLocation_entire[0]=0

print('最大位置',Hole_distance.max())
print('最深資料(個):', D)
print('寬度', W,'深度', D)
print('共有', typenumber, '種土壤材質')
print("土壤代號",unique_numbers,entire_matrix[0])
print('mapping:', mapping)
# 定義各個孔洞的位置
print('孔洞完整位置:', HoleLocation_entire)

# 計算轉移概率矩陣的函數
def calculate_transition_matrix(matrix,hole_location):
    group_number = np.zeros((D , W))
    # # 將地質數據中的類型分組存儲到 group_number 數組中
    #   TODO
    # for i in range(W):
    #     if i < int(W/2):
    #         group_number[0][i] = 1
    #     else:
    #         group_number[0][i] = 2
    # for i in range(D):
    #     for j in range(len(hole_location)):
    #         group_number[i][(hole_location[j])] = matrix[i][j]

    # T_t_V = np.zeros(len(matrix))
    # print('T_t_V:',T_t_V)
    # soiltype_V = {}
    for location in Hole_distance:
        for type in initial_array:
            for i in range(W):
                if i<location/interval:
                    group_number[0][i] = mapping[type]
                else:
                    continue
    print('group_number:',group_number[0])
    
    for i in range(D):
        for j in range(len(hole_location)):
            group_number[i][(hole_location[j])] = matrix[i][j]
    T_t_V = np.zeros(len(matrix))
    print('T_t_V:',T_t_V)
    soiltype_V = {}
               

    # ------------------------------------------------------------
    # for i in range(len(matrix[0])):
    #     for j in range(len(matrix)):
    #         if(matrix[j][i] == 0):
    #             continue
    #         T_t_V[j] = matrix[j][i]
    #     for k in T_t_V[0:len(T_t_V)]:
    #         soiltype_V[k] = soiltype_V.get(k, 0) + 1
    # rewrite the code above
    # Count the occurrence times of each soiltype
    # for row in matrix:
    #     for item in row:
    #         if item == 0:
    #             continue
    #         if item in soiltype_V:
    #             soiltype_V[item] += 1
    #         else:
    #             soiltype_V[item] = 1
    #------------------------------------------------------------ 
    soiltype_V = Counter(matrix.flatten())
    del soiltype_V[0]  # Remove count of zeros, if necessary
    soiltype_V = sorted(soiltype_V.items(), key=op.itemgetter(0), reverse=False)
    print('soiltype_V:',soiltype_V)
    VPCM = np.zeros((typenumber, typenumber))
    Tmatrix_V = np.zeros((typenumber, typenumber))

    for i in range(np.size(matrix, 1)):
        for j in range(len(matrix)):
            T_t_V[j] = matrix[j][i]

        for k in range(len(T_t_V) - 1):
            if T_t_V[k]==0 or T_t_V[k+1]==0:
                break
            for m in range(typenumber):
                for n in range(typenumber):
                    if T_t_V[k] == soiltype_V[m][0] and T_t_V[k + 1] == soiltype_V[n][0]:
                        VPCM[m][n] += 1
                        Tmatrix_V[m][n] += 1
    # 正規化
    count_V = np.sum(Tmatrix_V, axis=1)
    for i in range(np.size(Tmatrix_V, 1)):
        for j in range(np.size(Tmatrix_V, 1)):
            Tmatrix_V[i][j] = Tmatrix_V[i][j] / count_V[i]

    K = 9.3
    HPCM = np.zeros([len(count_V), len(count_V)])
    Tmatrix_H = np.zeros([len(count_V), len(count_V)])

    for i in range(np.size(Tmatrix_H, 1)):
        for j in range(np.size(Tmatrix_H, 1)):
            if i == j:
                HPCM[i][j] = K * VPCM[i][j]
                Tmatrix_H[i][j] = K * VPCM[i][j]
            else:
                HPCM[i][j] = VPCM[i][j]
                Tmatrix_H[i][j] = VPCM[i][j]

    count_H = np.sum(Tmatrix_H, axis=1)
    for i in range(np.size(Tmatrix_H, 1)):
        for j in range(np.size(Tmatrix_H, 1)):
            Tmatrix_H[i][j] = Tmatrix_H[i][j] / count_H[i]
            
    return Tmatrix_V, Tmatrix_H ,group_number

# 計算 HoleLocation_entire 的轉移矩陣
Tmatrix_V_entire, Tmatrix_H_entire ,group_number_entire= calculate_transition_matrix(entire_matrix,HoleLocation_entire)
print('Tmatrix_V_entire:\n',Tmatrix_V_entire)
print('Tmatrix_H_entire:\n',Tmatrix_H_entire)
# 預測地質類型的函數
def predict_geological_types(Tmatrix_V, Tmatrix_H, HoleLocation,group_number):
    L_state = 0
    M_state = 0
    Q_state = 0
    Nx = 0
    current_matrix = np.zeros(len(transitionName))

    conditions = {}
    for j in range(1, len(HoleLocation)):
        conditions[(HoleLocation[j - 1], HoleLocation[j])] = HoleLocation[j]
    for layer in range(1,D):
        for i in range(W):
            # 若為位置是有資料的就跳過(為鑽孔位置)
            if group_number[layer][i] :
                continue
            
            L_state = 0
            M_state = 0
            Q_state = 0
            Nx_TH = Tmatrix_H
            f_sum = 0
            k_sum = 0
            Nx=0
            if i in HoleLocation:
                if i!=0:
                    holekey=np.where(HoleLocation == i)[0][0]
                    for holeIndex in range(holekey,-1,-1):
                        if HoleLocation[holeIndex] != 0:
                            Nx = HoleLocation[holeIndex]
                            break
                    # 矩陣乘法更新Nx_TH
                    for _ in range(1, Nx - i):
                        Nx_TH = np.dot(Nx_TH, Tmatrix_H)
            else:
                for (lower, upper), nx in conditions.items():
                    if lower < i < upper:
                        Nx = nx
                        break
                for _ in range(1, Nx - i):
                    Nx_TH = np.dot(Nx_TH, Tmatrix_H)
            L_state = group_number[layer][i-1] - 1
            M_state = group_number[layer-1][i] - 1
            Q_state = group_number[layer][Nx] - 1          
           
            
            for f in range(typenumber):
                f_item1 = Tmatrix_H[int(L_state)][f]
                f_item2 = Nx_TH[f][int(Q_state)]
                f_item3 = Tmatrix_V[int(M_state)][f]
                f_sum += f_item1 * f_item2 * f_item3
            if f_sum == 0 :
                current_matrix= np.ones(typenumber) / typenumber
                # print('current_matrix:',current_matrix)
            else:
                for k in range(typenumber):
                    k_item1 = Tmatrix_H[int(L_state)][k]
                    k_item2 = Nx_TH[k][int(Q_state)]
                    k_item3 = Tmatrix_V[int(M_state)][k]
                    k_sum = k_item1 * k_item2 * k_item3

                    current_matrix[k] = k_sum / f_sum

            # 進行預測
            predict_type = np.random.choice(transitionName, replace=True, p=current_matrix)
            # if i in HoleLocation:
            #     print('k_sum:',k_sum,'f_sum:',f_sum,'Nx:',Nx)

            group_number[layer][i] =predict_type
    return group_number




predict_result_entire = predict_geological_types(Tmatrix_V_entire, Tmatrix_H_entire, HoleLocation_entire,group_number_entire)
correct_location=Hole_distance[predict_hole]
print('coeeect_location:',correct_location)
print('predict_result:',predict_result_entire[1])
predict_result=predict_result_entire[int(Hole_distance[predict_hole]/interval)]
print('predict_result:',predict_result)


print('predict_hole_data:', predict_hole_data)

molecular = 0
denominator = 0
for i, x in zip(predict_hole_data, predict_result):
    if i != 0:
        if i == x:
            molecular += 1
            denominator += 1
        else:
            denominator += 1
correct_rate=molecular/denominator
print('Correct_rate:',correct_rate)




# 可視化地質類型分布
plt.figure(figsize=(6, 4), dpi=150)
plt.imshow(predict_result_entire, cmap='tab10', origin='upper')
cbar = plt.colorbar()
# 獲取mapping所有key
mapping_key = list(mapping.keys())
mapping_value = list(mapping.values())
print('mapping_key:',mapping_key)
print('mapping_value:',mapping_value)
cbar.set_ticks(mapping_value)  # 設置顏色條刻度
cbar.set_ticklabels(mapping_key)  # 設置顏色條標籤
plt.title('Prediction(top to bottom)')
plt.xlabel('Width (units)')
plt.ylabel('Depth (units)')
plt.savefig('Prediction(top to bottom).png')
plt.show()
plt.clf()