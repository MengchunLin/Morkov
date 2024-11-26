import numpy as np
import operator as op
import matplotlib.pyplot as plt
import pandas as pd
import json
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
# 打開文件
predict_result_entire = np.loadtxt('predict_result_entire.csv', delimiter=",", skiprows=1)
interval = 0.5
# 打開現存鑽孔
with open('created_file.json', 'r') as file:
    file_path_list = json.load(file)

print('file_path_list:', file_path_list)

def predict_location_input():
    def submit():
        # Get input from the entry widget
        location = entry.get()
        # Store it in a variable and close the GUI
        nonlocal user_input
        user_input = location
        root.destroy()

    root = tk.Tk()
    root.title("預測位置輸入")
    root.geometry("300x150")

    # Label
    label = tk.Label(root, text="請輸入預測位置:")
    label.pack(pady=10)

    # Entry widget for user input
    entry = tk.Entry(root, width=25)
    entry.pack(pady=5)

    # Button to submit the input
    button = tk.Button(root, text="確認", command=submit)
    button.pack(pady=10)

    # Variable to store user input
    user_input = None

    # Start the Tkinter event loop
    root.mainloop()

    return user_input

# Get the user input
user_input = predict_location_input()

def predict_data(predict_result_entire, predict_location):
    predict_location = int(predict_location)
    col = int(predict_location / interval)
    predict_location_soiltype = predict_result_entire[:, col]
    print(f"Predicted soil types at location {predict_location_soiltype}")

    with open('created_file.json', 'r') as file:
        file_path_list = json.load(file)
    print('file_path_list:', file_path_list)

    predict_df = pd.DataFrame({
        "Location": np.arange(len(predict_location_soiltype)),
        "SoilType": predict_location_soiltype,
        "Depth (m)": np.arange(len(predict_location_soiltype)) * 0.02,
        "qc (MPa)": np.zeros(len(predict_location_soiltype)),
        "fs (MPa)": np.zeros(len(predict_location_soiltype)),
        "u (MPa)": np.zeros(len(predict_location_soiltype)),
    })
    # 讀取所有鑽孔文件
    dfs = []
    for file_path in file_path_list:
        df = pd.read_excel(file_path)
        # 重組df
        # 只留下Depth, 合併後, qc, fs, u
        df = df[['Depth (m)', '合併後', 'qc (MPa)', 'fs (MPa)', 'u (MPa)']]
        dfs.append(df)
    print('dfs:', dfs)

    for i in range(len(predict_location_soiltype)):
        exist_qc, exist_fs, exist_u = [], [], []

        for j, df in enumerate(dfs):
            if i < len(df) and df['合併後'][i] == predict_location_soiltype[i]:
                exist_qc.append(df['qc (MPa)'][i])
                exist_fs.append(df['fs (MPa)'][i])
                exist_u.append(df['u (MPa)'][i])

        if exist_qc:
            predict_df.loc[i, 'qc (MPa)'] = np.random.choice(exist_qc)
            predict_df.loc[i, 'fs (MPa)'] = np.random.choice(exist_fs)
            predict_df.loc[i, 'u (MPa)'] = np.random.choice(exist_u)

        else:
            depth_list, qc_list, fs_list, u_list = [], [], [], []
            for k in range(1, min(11, i + 1)):
                if predict_location_soiltype[i - k] == predict_location_soiltype[i]:
                    depth_list.append(predict_df.loc[i - k, 'Depth (m)'])
                    qc_list.append(predict_df.loc[i - k, 'qc (MPa)'])
                    fs_list.append(predict_df.loc[i - k, 'fs (MPa)'])
                    u_list.append(predict_df.loc[i - k, 'u (MPa)'])

            if len(depth_list) >= 3:
                qc_trend = np.polyfit(depth_list, qc_list, 2)
                fs_trend = np.polyfit(depth_list, fs_list, 2)
                u_trend = np.polyfit(depth_list, u_list, 2)
                predict_df.loc[i, 'qc (MPa)'] = np.polyval(qc_trend, (i + 1) * 0.02)
                predict_df.loc[i, 'fs (MPa)'] = np.polyval(fs_trend, (i + 1) * 0.02)
                predict_df.loc[i, 'u (MPa)'] = np.polyval(u_trend, (i + 1) * 0.02)
            else:
                # 若無法找到相同土層的資料，則向前找到最近的相同土層資料，並取趨勢線
                for k in range(1, i + 1):
                    if predict_location_soiltype[i - k] == predict_location_soiltype[i]:
                        # 再向前找10個點
                        for z in range(k, k-100, -1):
                            depth_list.append(predict_df.loc[i - k, 'Depth (m)'])
                            qc_list.append(predict_df.loc[i - k, 'qc (MPa)'])
                            fs_list.append(predict_df.loc[i - k, 'fs (MPa)'])
                            u_list.append(predict_df.loc[i - k, 'u (MPa)'])
                        qc_trend = np.polyfit(depth_list, qc_list, 2)
                        fs_trend = np.polyfit(depth_list, fs_list, 2)
                        u_trend = np.polyfit(depth_list, u_list, 2)
                        predict_df.loc[i, 'qc (MPa)'] = np.polyval(qc_trend, (i + 1) * 0.02)
                        predict_df.loc[i, 'fs (MPa)'] = np.polyval(fs_trend, (i + 1) * 0.02)
                        predict_df.loc[i, 'u (MPa)'] = np.polyval(u_trend, (i + 1) * 0.02)
                        

    print('predict_df:',predict_df)
    # 將預測結果保存為Excel文件
    predict_df.to_excel('predict_result.xlsx', index=False)
    return predict_df




                




predict_data = predict_data(predict_result_entire, user_input)
print('done')
