import numpy as np
import pandas as pd
import operator as op
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import subprocess
import json
import tkinter as tk
import time
# -----------testing file----------------
Matrix4D = "test.csv"
Matrix5D = '5DMatrix.csv'
sixHole = '6Hole.csv'
test='test - 複製.csv'
eightSoil='8soil.csv'
CECI='markov_matrix.csv'
CTCI='CTCI.csv'
# -----------testing file----------------
# 記錄開始時間
start_time = time.time()
# -----------call simplify_data.py----------------
#　subprocess.run(["python", "Data_processing.py"])
# 打開檔案'合併結果.csv'
entire_file = pd.read_csv('markov_matrix.csv')
# -----------call simplify_data.py----------------

# file preprocessing
entire_file = pd.read_csv(CECI, delimiter=",").fillna(0).values # 讀取文件空值全部補0
entire_matrix = entire_file[1:, :]  # skip first column 第一行是位置
Hole_distance = entire_file[0]
initial_array = entire_file[1]
# 取得土壤種類
unique_numbers = np.unique(entire_matrix)
# 從unique_numbers過濾掉0
unique_numbers = unique_numbers[unique_numbers != 0]
typenumber = len(unique_numbers)
# 建立土壤代號對應的數字
mapping = {value: index+1 for index, value in enumerate(unique_numbers)}
# 將公司土壤代號轉換為 1 ~ ...
def map_value(value):
    return mapping.get(value, value)  # 如果 value 在 mapping 中有對應的值，則映射；否則保持原值
entire_matrix =  np.vectorize(map_value)(entire_matrix)
transitionName = np.arange(1,typenumber+1)
# file preprocessing


# 定義模型的間隔、寬度、深度、面積、孔洞數量和地質類型數量等參數
interval = 1
W = int(Hole_distance.max() / interval) + 1
D = int(entire_matrix.shape[0]/ interval)
print('D:',D)
print('W:',W)


denominator = 0
molecular = 0
A = W * D
print('A:',A)
HoleLocation_entire=(Hole_distance/interval).astype(int)
HoleLocation_entire[0]=0

# print('最大位置',Hole_distance.max())
# print('最深資料(個):', D)
# print('寬度', W,'深度', D)
# print('共有', typenumber, '種土壤材質')
# print("土壤代號",unique_numbers,entire_matrix[0])
# print('mapping:', mapping)
# # 定義各個孔洞的位置
# print('孔洞完整位置:', HoleLocation_entire)

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
    # print('group_number:',group_number[0])
    
    for i in range(D):
        for j in range(len(hole_location)):
            group_number[i][(hole_location[j])] = matrix[i][j]
    T_t_V = np.zeros(len(matrix))
    # print('T_t_V:',T_t_V)
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
    # print('soiltype_V:',soiltype_V)
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
# print('Tmatrix_V_entire:\n',Tmatrix_V_entire)
# print('Tmatrix_H_entire:\n',Tmatrix_H_entire)
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
            Nx_TH = np.eye(Tmatrix_H.shape[0])
            for _ in range(Nx - i):
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

print('predicting...')
predict_result_entire = predict_geological_types(Tmatrix_V_entire, Tmatrix_H_entire, HoleLocation_entire,group_number_entire)
print(len(predict_result_entire))
# 儲存預測結果
np.savetxt('predict_result_entire.csv', predict_result_entire, delimiter=',', fmt='%d')
# print('predict_result_entire:\n',predict_result_entire)

def irregular_shift(predict_result_entire, max_shift):
    """
    根據線性規則將矩陣逐列向下移動，形成不規則矩陣。
    
    Args:
        predict_result_entire (ndarray): 原始矩陣，大小為 (D, W)
        max_shift (int): 最大移動深度（最右列的移動量）
    
    Returns:
        ndarray: 不規則矩陣
    """
    D, W = predict_result_entire.shape  # 原矩陣大小

    # 計算新矩陣的高度
    new_height = D + max_shift

    # 初始化新矩陣，填充為 NaN 或其他佔位符
    irregular_matrix = np.full((new_height, W), np.nan)

    # 設定線性移動規則，移動量從 0 到 max_shift
    shift_values = np.linspace(0, max_shift, W, dtype=int)

    # 逐列移動
    for col in range(W):
        shift = shift_values[col]  # 當前列的移動深度
        irregular_matrix[shift:shift + D, col] = predict_result_entire[:, col]

    return irregular_matrix

# 不規則矩陣
max_shift = 0  # 最大移動深度
irregular_matrix = irregular_shift(predict_result_entire, max_shift)
# print('irregular_matrix:\n',irregular_matrix)

D=irregular_matrix.shape[0]


# 可視化地質類型分布
mapping_key = list(mapping.keys())
mapping_value = list(mapping.values())
# print('mapping_key:',mapping_key)
# print('mapping_value:',mapping_value)
original_ticks = np.arange(0, int(Hole_distance.max() / interval), 100)
# 缩放后的 x 轴刻度标签
scaled_labels = (original_ticks * interval).astype(int)
# 定義自定義顏色映射
soil_colors = {
    1: 'lightsalmon',       
    2: 'lightsteelblue',      
    3: 'plum',     
    4: 'darkkhaki',      
    5: 'burlywood',   
}

import matplotlib.colors as mcolors
# 創建自定義的離散顏色映射
colors = [soil_colors[i] for i in range(1, len(soil_colors) + 1)]
cmap = mcolors.ListedColormap(colors)

# 設置圖形大小
plt.figure(figsize=(8, 4), dpi=150)

# 使用自定義的顏色映射來顯示預測結果
im = plt.imshow(irregular_matrix, cmap=cmap, origin='upper', aspect='auto')

# 創建圖例
patches = [mpatches.Patch(color=soil_colors[i], label=f"Type {int(mapping_key[i-1])}") 
          for i in range(1, len(mapping) + 1)]

plt.legend(
    handles=patches,
    loc='center left',  
    bbox_to_anchor=(1.05, 0.5), 
    title="Geological Types",
    fontsize=8,
    title_fontsize=10,
    frameon=True
)

# 繪製鑽孔位置的垂直線
for i in HoleLocation_entire:
    plt.axvline(x=i, color='black', linestyle='--', linewidth=0.5)
    plt.axvline(x=i-interval, color='black', linestyle='--', linewidth=0.5)

# 設置刻度
original_ticks = np.arange(0, int(Hole_distance.max() / interval), 100)
scaled_labels = (original_ticks * interval).astype(int)

plt.xticks(
    ticks=original_ticks,
    labels=scaled_labels,
    fontsize=10
)
plt.yticks(
    ticks=np.arange(0, D, 200),
    labels=np.arange(0, D, 200),
    fontsize=10
)

# 設置圖形外觀
plt.gca().set_aspect('auto')
plt.xlim(0, W)
plt.ylim(D, 0)
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.title('Prediction (top to bottom)')
plt.xlabel('Width (units)')
plt.ylabel('Depth (units)')
plt.tight_layout()
plt.savefig('Prediction_with_legend_side.png')
plt.show()

# 記錄結束時間
end_time = time.time()

# 計算運行時間（以秒為單位）
execution_time = end_time - start_time
print(f"程式運行時間: {execution_time:.2f} 秒")
# -----------------預測位置輸入視窗-----------------
# def predict_location_input():
#     def submit():
#         # Get input from the entry widget
#         location = entry.get()
#         # Store it in a variable and close the GUI
#         nonlocal user_input
#         user_input = location
#         root.destroy()

#     root = tk.Tk()
#     root.title("預測位置輸入")
#     root.geometry("300x150")

#     # Label
#     label = tk.Label(root, text="請輸入預測位置:")
#     label.pack(pady=10)

#     # Entry widget for user input
#     entry = tk.Entry(root, width=25)
#     entry.pack(pady=5)

#     # Button to submit the input
#     button = tk.Button(root, text="確認", command=submit)
#     button.pack(pady=10)

#     # Variable to store user input
#     user_input = None

#     # Start the Tkinter event loop
#     root.mainloop()

#     return user_input

# # Get the user input
# user_input = predict_location_input()

# def predict_data(predict_result_entire, predict_location):
#     predict_location = int(predict_location)
#     col = int(predict_location / interval)
#     predict_location_soiltype = predict_result_entire[:, col]
#     print(f"Predicted soil types at location {predict_location_soiltype}")

#     with open('file_path_list.json', 'r') as file:
#         file_path_list = json.load(file)
#     print('file_path_list:', file_path_list)

#     predict_df = pd.DataFrame({
#         "Location": np.arange(len(predict_location_soiltype)),
#         "SoilType": predict_location_soiltype,
#         "Depth (m)": np.arange(len(predict_location_soiltype)) * 0.02,
#         "qc (MPa)": np.zeros(len(predict_location_soiltype)),
#         "fs (MPa)": np.zeros(len(predict_location_soiltype)),
#         "u (MPa)": np.zeros(len(predict_location_soiltype)),
#     })

#     dfs = [pd.read_excel(file_path) for file_path in file_path_list]
#     print('dfs:', dfs)

#     for i in range(len(predict_location_soiltype)):
#         exist_qc, exist_fs, exist_u = [], [], []

#         for j, df in enumerate(dfs):
#             if i < len(df) and df['Soil Type'][i] == predict_location_soiltype[i]:
#                 exist_qc.append(df['qc (MPa)'][i])
#                 exist_fs.append(df['fs (MPa)'][i])
#                 exist_u.append(df['u (MPa)'][i])

#         if exist_qc:
#             predict_df.loc[i, 'qc (MPa)'] = np.random.choice(exist_qc)
#             predict_df.loc[i, 'fs (MPa)'] = np.random.choice(exist_fs)
#             predict_df.loc[i, 'u (MPa)'] = np.random.choice(exist_u)
#         else:
#             depth_list, qc_list, fs_list, u_list = [], [], [], []
#             for k in range(1, min(11, i + 1)):
#                 if predict_location_soiltype[i - k] == predict_location_soiltype[i]:
#                     depth_list.append(predict_df.loc[i - k, 'Depth (m)'])
#                     qc_list.append(predict_df.loc[i - k, 'qc (MPa)'])
#                     fs_list.append(predict_df.loc[i - k, 'fs (MPa)'])
#                     u_list.append(predict_df.loc[i - k, 'u (MPa)'])

#             if len(depth_list) >= 3:
#                 qc_trend = np.polyfit(depth_list, qc_list, 2)
#                 fs_trend = np.polyfit(depth_list, fs_list, 2)
#                 u_trend = np.polyfit(depth_list, u_list, 2)
#                 predict_df.loc[i, 'qc (MPa)'] = np.polyval(qc_trend, (i + 1) * 0.02)
#                 predict_df.loc[i, 'fs (MPa)'] = np.polyval(fs_trend, (i + 1) * 0.02)
#                 predict_df.loc[i, 'u (MPa)'] = np.polyval(u_trend, (i + 1) * 0.02)
#             else:
#                 predict_df.loc[i, ['qc (MPa)', 'fs (MPa)', 'u (MPa)']] = np.nan
#     print('predict_df:',predict_df)
#     # 將預測結果保存為Excel文件
#     predict_df.to_excel('predict_result.xlsx', index=False)
#     return predict_df




                




# predict_data = predict_data(predict_result_entire, user_input)
# print('done')
