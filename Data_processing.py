import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog
import json

created_file = []
# 選擇檔案
def select_file():
    root = tk.Tk()
    root.withdraw()
    file= filedialog.askopenfilename(title="選擇檔案")
    return file

# 設定合併層厚度閾值
def get_thickness_threshold():
    root = tk.Tk()
    root.withdraw()
    thickness_threshold = (simpledialog.askinteger("合併層厚度", "請輸入合併厚度 (cm):")) / 2
    if thickness_threshold is None:
        thickness_threshold = 5  # 默認為5cm
    root.destroy()
    return thickness_threshold

# 分類土壤類型
def classify_soil_type(Ic):
    # 把Ic轉為float

    Ic = round(Ic, 2)
    if Ic <= 2.05:
        return 1
    elif 2.05 < Ic <= 2.3:
        return 2
    elif 2.3 < Ic < 2.6:
        return 3
    elif 2.6 < Ic < 2.95:
        return 4
    elif Ic >= 2.95:
        return 5

# 標記差異
def mark(previous_data, current_data):
    mark_list = [''] * len(current_data)
    for i in range(len(current_data)):
        if i < len(previous_data) and previous_data[i] != current_data[i]:
            mark_list[i] = '*'
    return mark_list

# 數據組織 (土壤類型、厚度、Ic 平均)
def data_array(Soil_Type, Ic):
    layer, thickness, ic_avg = [], [], []
    current_soil_type = None
    current_count = 0

    for i in range(len(Soil_Type)):
        soil_type = Soil_Type[i]
        if soil_type != current_soil_type and current_soil_type is not None:
            layer.append(current_soil_type)
            thickness.append(current_count)
            ic_avg.append(np.mean(Ic[i - current_count:i]))
            current_count = 0

        current_soil_type = soil_type
        current_count += 1

    if current_soil_type is not None:
        layer.append(current_soil_type)
        thickness.append(current_count)
        ic_avg.append(np.mean(Ic[len(Soil_Type) - current_count:]))

    return [layer, thickness, ic_avg]

# 合併層
def merge_layer(soil_data, thickness_threshold):
    i = 0  # Start index at 0 and manually control the iteration
    while i < len(soil_data):
        if soil_data.iloc[i, 1] <= thickness_threshold:
            if i == len(soil_data) - 1:  # If it's the last row
                soil_data.iloc[i - 1, 1] += soil_data.iloc[i, 1]  # Merge with the previous row
                soil_data.iloc[i, 0] = soil_data.iloc[i - 1, 0]  # Adjust layer designation
            else:
                if soil_data.iloc[i + 1, 0] == soil_data.iloc[i - 1, 0]:  # Merge with surrounding layers
                    soil_data.iloc[i - 1, 1] += soil_data.iloc[i, 1]
                elif soil_data.iloc[i - 1, 0] != soil_data.iloc[i + 1, 0]:
                    if abs(soil_data.iloc[i - 1, 2] - soil_data.iloc[i, 2]) > abs(soil_data.iloc[i, 2] - soil_data.iloc[i + 1, 2]):
                        soil_data.iloc[i + 1, 1] += soil_data.iloc[i, 1]  # Merge with the next row
                    else:
                        soil_data.iloc[i - 1, 1] += soil_data.iloc[i, 1]  # Merge with the previous row
            
            # Drop the current row after merging
            soil_data = soil_data.drop(i).reset_index(drop=True)
            i -= 1  # Move back one index to compensate for the dropped row
        
        i += 1  # Move to the next index

    return soil_data

# 簡化合併soil data
def merge_processed_data(soil_data):
    i = 0
    while i < len(soil_data) - 1:
        # 如果相邻两行的土壤类型相同
        if soil_data.iloc[i, 0] == soil_data.iloc[i + 1, 0]:
            # 合并厚度
            soil_data.iloc[i, 1] += soil_data.iloc[i + 1, 1]
            
            # 删除合并后的行
            soil_data = soil_data.drop(i + 1).reset_index(drop=True)
        else:
            # 只有在没有合并的情况下才增加索引
            i += 1
    
    return soil_data



# 生成最終數據
def write_merged_data(soil_data):
    data_input = []
    for i in range(len(soil_data)):
        soil_type = soil_data.iloc[i, 0]
        thickness = int(soil_data.iloc[i, 1])
        data_input.extend([soil_type] * thickness)
    return data_input

def process_file(file, thickness_threshold):
    df = pd.read_excel(file, header=0)
    df_copy = df.copy()

    # 資料處理
    # 把Ic轉為float
    df_copy['Ic'] = df_copy['Ic'].replace(' ', 0).astype(float)
    df_copy['Ic'] = df_copy['Ic'].interpolate(method='linear')
    df_copy['Soil Type'] = df_copy['Soil Type'].ffill()

    # 分類土壤類型
    Soil_Type_5 = df_copy['Ic'].apply(classify_soil_type)
    df_copy['Soil Type 5 type'] = Soil_Type_5
    df_copy['Mark1'] = ''

    # 計算層數、厚度和 Ic 平均值
    layers, thicknesses, ic_avgs = data_array(Soil_Type_5, df_copy['Ic'])
    result_df = pd.DataFrame({'Soil Type': layers, 'Thickness': thicknesses, 'Ic_avg': ic_avgs})

    # 第一次合併（合併厚度 <= 5cm）
    result_array1 = merge_layer(result_df, 5)

    # 寫入第一次處理後的數據
    data_input1 = write_merged_data(result_array1)
    df_copy['10cm'] = data_input1
    df_copy['Mark2'] = ''
    result_array1 = merge_processed_data(result_array1)
    
    # 確保數據長度匹配
    if len(data_input1) > len(df_copy):
        data_input1 = data_input1[:len(df_copy)]  # 截斷數據以匹配長度
    elif len(data_input1) < len(df_copy):
        data_input1.extend([''] * (len(df_copy) - len(data_input1)))  # 填充空值以匹配長度

    #對比soil type 5 和 5cm
    mark_array = mark(Soil_Type_5, data_input1)

    # 標記第一次合併後的數據
    df_copy['Mark1'] = mark_array

    # 第二次合併（基於用戶輸入的厚度閾值）
    result_array2 = merge_layer(result_array1, thickness_threshold)

    # 寫入第二次處理後的數據
    data_input = write_merged_data(result_array2)

    # 確保數據長度匹配
    if len(data_input) > len(df_copy):
        data_input = data_input[:len(df_copy)]  # 截斷數據以匹配長度
    elif len(data_input) < len(df_copy):
        data_input.extend([''] * (len(df_copy) - len(data_input)))  # 填充空值以匹配長度

    df_copy['合併後'] = data_input

    # 標記第二次合併後的數據
    mark_array = mark(data_input1, data_input)
    df_copy['Mark2'] = mark_array  # 標記第二次合併的變化

    # 將處理後的資料存入新的 Excel 檔案
    processed_file = file.replace('.xlsx', '_processed.xlsx')
    # 把處理好的資料路徑存入列表
    
    df_copy.to_excel(processed_file, index=False)
    created_file.append(processed_file)
    with open('created_file.json', 'w') as f:
        json.dump(created_file, f)
    print(f'處理完成，已儲存：{processed_file}')
    return processed_file

def how_much_file_to_input():
    root = tk.Tk()
    root.withdraw()
    file_count = simpledialog.askinteger("檔案數量", "請輸入要處理的檔案數量:")
    if file_count is None:
        file_count = 2  # 默認為2
    root.destroy()
    return file_count

# 輸入鑽孔資訊
def input_drilling_info(file_count):
    """
    顯示自定義視窗，讓使用者在同一個視窗中輸入鑽孔名稱及位置資訊。
    
    Args:
        file_count (int): 需要輸入的鑽孔數量。
    
    Returns:
        tuple: 兩個列表，分別儲存鑽孔名稱和位置。
    """
    hole_names = []  # 儲存鑽孔名稱
    hole_locations = []  # 儲存鑽孔位置

    # 自定義輸入窗口
    def ask_drilling_info(i):
        """
        創建輸入窗口，供用戶輸入名稱和位置。
        """
        # 創建新視窗
        popup = tk.Toplevel()
        popup.title(f"第 {i + 1} 組鑽孔資訊")

        # 標籤和輸入框：鑽孔名稱
        tk.Label(popup, text="鑽孔名稱:").grid(row=0, column=0, padx=10, pady=10)
        hole_name_var = tk.StringVar()
        tk.Entry(popup, textvariable=hole_name_var).grid(row=0, column=1, padx=10, pady=10)

        # 標籤和輸入框：鑽孔位置
        tk.Label(popup, text="鑽孔位置:").grid(row=1, column=0, padx=10, pady=10)
        hole_location_var = tk.StringVar()
        tk.Entry(popup, textvariable=hole_location_var).grid(row=1, column=1, padx=10, pady=10)

        # 確定按鈕
        def submit():
            hole_names.append(hole_name_var.get())  # 將名稱儲存到 hole_names 列表
            hole_locations.append(hole_location_var.get())  # 將位置儲存到 hole_locations 列表
            popup.destroy()

        tk.Button(popup, text="確定", command=submit).grid(row=2, column=0, columnspan=2, pady=10)

        # 等待視窗完成輸入
        popup.grab_set()
        popup.wait_window(popup)

    # 創建主窗口（隱藏）
    root = tk.Tk()
    root.withdraw()

    # 按次數顯示輸入窗口
    for i in range(file_count):
        ask_drilling_info(i)

    root.destroy()  # 關閉主窗口
    return hole_names, hole_locations  # 返回兩個列表


# 用清單儲存每次處理後的檔案路徑
processed_files = []
import pandas as pd

def main():
    # 獲取閾值並初始化列表儲存處理結果
    thickness_threshold = get_thickness_threshold()
    processed_files = []
    file_list = []
    file_count = how_much_file_to_input()
    hole_names, hole_locations = input_drilling_info(file_count)  # 獲取名稱和位置

    # 初始化 Markov 矩陣，包含 hole_names 和 hole_locations
    markov_matrix = pd.DataFrame(columns=hole_names)
    markov_matrix.loc[0] = hole_locations  # 第 2 行為鑽孔位置

    # 處理檔案資料
    for i in range(file_count):
        try:
            file = select_file()
            processed_file = process_file(file, thickness_threshold)
            processed_files.append(processed_file)

            # 嘗試讀取 Excel 資料
            df = pd.read_excel(processed_file)
            if '合併後' in df.columns:  # 確保存在 '合併後' 欄位
                # 取得 '合併後' 欄位資料
                new_data = df['合併後']
                print(f"new_data (檔案{i+1}): {new_data}")

                # 動態擴展行數（如果需要）
                additional_rows = len(new_data) - (len(markov_matrix) - 2)
                if additional_rows > 0:
                    for _ in range(additional_rows):
                        markov_matrix.loc[len(markov_matrix)] = [None] * len(markov_matrix.columns)

                # 插入新資料到對應列
                markov_matrix.iloc[1:1+len(new_data), i] = new_data.values
            else:
                print(f"警告: 檔案 {processed_file} 缺少 '合併後' 欄位，插入空值。")
                # 插入空值列
                markov_matrix.iloc[1:, i] = None

        except Exception as e:
            print(f"處理檔案 {file} 時出錯: {e}")
            # 發生錯誤時插入空值列
            markov_matrix.iloc[1:, i] = None
    # 刪掉最後一行
    markov_matrix = markov_matrix.dropna(axis=1, how='all')
    # 將 NaN 填充為 0，若有需要
    markov_matrix = markov_matrix.fillna(0)
    print("Markov 矩陣:")
    print(markov_matrix)

    # 儲存為 CSV
    markov_matrix.to_csv("markov_matrix.csv", index=False)
    print("Markov 矩陣已儲存為 'markov_matrix.csv'。")



    
    # 將處理後的檔案列表寫入文件
    with open("processed_files.xlsx", "w") as f:
        f.write("\n".join(processed_files))
    print("所有檔案簡化土層完成")
    print(f"已儲存的檔案：{processed_files}")

if __name__ == "__main__":
    main()