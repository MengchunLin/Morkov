import numpy as np
import operator as op
import matplotlib.pyplot as plt

Matrix4D = "test.csv"
Matrix5D = '5DMatrix.csv'
sixHole = '6Hole.csv'
denominator = 0
molecular = 0
HoleLocation_entire = []
HoleLocation_verify = []
verify_hole_index = 3
verify_matrix =np.array([])

# 定義模型的間隔、寬度、深度、面積、孔洞數量和地質類型數量等參數
interval = 0.5
W = int(70 / interval)  # 140
D = int(25 / interval)  # 50
A = W * D
entire_file = np.loadtxt(Matrix4D, delimiter=",", skiprows=1)
entiry_matrix = entire_file[1:, :]  # skip first column
verify_matrix = np.delete(entiry_matrix, verify_hole_index, 1)
test_hole = entiry_matrix[:, verify_hole_index]
Hole_distance = entire_file[0]


# 獲取不同數字的數量
unique_numbers = np.unique(entiry_matrix)
typenumber = len(unique_numbers)
print('共有', typenumber, '種土壤材質')

# 初始化地質類型的分組數組

# 定義各個孔洞的位置
HoleLocation_entire=Hole_distance/interval
HoleLocation_entire=HoleLocation_entire.astype(int)
HoleLocation_entire[0]=1
print('孔洞位置:', HoleLocation_entire)
HoleLocation_verify = np.delete(HoleLocation_entire, verify_hole_index)
print('驗證孔洞位置:', HoleLocation_verify)



# 計算轉移概率矩陣的函數
def calculate_transition_matrix(matrix,hole_location):
    group_number = np.zeros(A)
    print("location size",hole_location)
    # # 將地質數據中的類型分組存儲到 group_number 數組中
    for i in range(1, 74, 1):
        group_number[i - 1] = 1
    for i in range(74, 140, 1):
        group_number[i - 1] = 2

    for i in range(1, D + 1, 1):
        for j in range(len(hole_location)):
            group_number[(hole_location[j] - 1) + (i - 1) * W] = matrix[i - 1][j]

            
    T_t_V = np.zeros(len(matrix))
    soiltype_V = {}

    for i in range(np.size(matrix, 1)):
        for j in range(len(matrix)):
            T_t_V[j] = matrix[j][i]
        for k in T_t_V[0:len(T_t_V)]:
            soiltype_V[k] = soiltype_V.get(k, 0) + 1

    soiltype_V = sorted(soiltype_V.items(), key=op.itemgetter(0), reverse=False)
    

    VPCM = np.zeros([len(soiltype_V), len(soiltype_V)])
    Tmatrix_V = np.zeros([len(soiltype_V), len(soiltype_V)])

    for i in range(np.size(matrix, 1)):
        for j in range(len(matrix)):
            T_t_V[j] = matrix[j][i]

        for k in range(len(T_t_V) - 1):
            for m in range(len(soiltype_V)):
                for n in range(len(soiltype_V)):
                    if T_t_V[k] == soiltype_V[m][0] and T_t_V[k + 1] == soiltype_V[n][0]:
                        VPCM[m][n] += 1
                        Tmatrix_V[m][n] += 1

    count_V = np.sum(Tmatrix_V, axis=1)
    for i in range(np.size(Tmatrix_V, 1)):
        for j in range(np.size(Tmatrix_V, 1)):
            Tmatrix_V[i][j] = Tmatrix_V[i][j] / count_V[i]
    print('Tmatrix_V:', Tmatrix_V)

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

    print('Tmatrix_V:\n', Tmatrix_V)
    print('Tmatrix_H:\n', Tmatrix_H)


    return Tmatrix_V, Tmatrix_H ,group_number


# 計算 HoleLocation_entire 的轉移矩陣
Tmatrix_V_entire, Tmatrix_H_entire ,group_number_entire= calculate_transition_matrix(entiry_matrix,HoleLocation_entire)
# 計算 HoleLocation_verify 的轉移矩陣
Tmatrix_V_verify, Tmatrix_H_verify,group_number_verify = calculate_transition_matrix(verify_matrix,HoleLocation_verify)

# 預測地質類型的函數
def predict_geological_types(Tmatrix_V, Tmatrix_H, HoleLocation,group_number):
    L_state = 0
    M_state = 0
    Q_state = 0
    Nx = 0
    a = 0
    current_matrix = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
    transitionName = np.array([[1, 2, 3, 4, 5]])

    conditions = {}
    for j in range(1, len(HoleLocation)):
        conditions[(HoleLocation[j - 1], HoleLocation[j])] = HoleLocation[j]

    for layer in range(2, D + 1):
        for i in range(1, W + 1):
            L_state = 0
            M_state = 0
            Q_state = 0

            for (lower, upper), nx in conditions.items():
                if lower < i < upper:
                    L_state = group_number[(i - 2) + (layer - 1) * W] - 1
                    M_state = group_number[(i - 1) + (layer - 2) * W] - 1
                    Q_state = group_number[(nx - 1) + (layer - 1) * W] - 1
                    Nx = nx
                    break

            if i in HoleLocation:
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

    return group_number

# 複製 group_number 以避免互相干擾
# group_number_entire = group_number.copy()
# group_number_verify = group_number.copy()

# 預測並計算 HoleLocation_entire 的地質類型
predict_result_entire = predict_geological_types(Tmatrix_V_entire, Tmatrix_H_entire, HoleLocation_entire,group_number_entire)
# 預測並計算 HoleLocation_verify 的地質類型
predict_result_verify = predict_geological_types(Tmatrix_V_verify, Tmatrix_H_verify, HoleLocation_verify,group_number_verify)

# 重塑地質類型分組數組為矩陣
predict_result_entire = predict_result_entire.reshape(D, W)
predict_result_verify = predict_result_verify.reshape(D, W)
# print('預測鑽孔編號:',verify_hole_index,  HoleLocation_verify[verify_hole_index])
print('預測鑽孔編號:',verify_hole_index)
verify_array = predict_result_verify[:, HoleLocation_entire[verify_hole_index]]
for i, x in zip(test_hole, verify_array):
    if i == x:
        denominator += 1
        molecular += 1
    else:
        denominator += 1
correct_rate = molecular / denominator
print('正確率:', correct_rate * 100, "%")

# 可視化地質類型分布
plt.imshow(predict_result_entire, cmap='tab10', origin='upper')
plt.colorbar(label='Geological Type')
plt.title('entiry hole pic')
plt.xlabel('Width (units)')
plt.ylabel('Depth (units)')
plt.savefig('geological_type_prediction_entiry.png')
plt.clf()

plt.imshow(predict_result_verify, cmap='tab10', origin='upper')
plt.colorbar(label='Geological Type')
plt.title('test hole pic')
plt.xlabel('Width (units)')
plt.ylabel('Depth (units)')
plt.savefig('geological_type_prediction_verify.png')