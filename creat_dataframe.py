import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import pandas as pd
import json

file_path_list = []  # 用於存儲檔案名稱
file_distance_list = []  # 用於存儲檔案座標


def select_files_and_set_coordinates():
    global file_path_list, file_distance_list  # Use global variables
    root = tk.Tk()
    root.withdraw()  # 隱藏主視窗

    # 提示使用者輸入需要選擇檔案的次數
    num_files = simpledialog.askinteger("檔案選擇次數", "請輸入需要選擇檔案的次數：", parent=root)
    if not num_files or num_files <= 0:
        messagebox.showerror("錯誤", "無效的檔案數量。程式結束。")
        return None, None, None

    for i in range(num_files):
        # 選擇檔案
        file_path = filedialog.askopenfilename(title=f"選擇第 {i + 1} 個檔案")
        if not file_path:
            messagebox.showwarning("警告", f"第 {i + 1} 次未選擇檔案，跳過。")
            continue

        # 座標輸入視窗
        coordinate_window = tk.Toplevel(root)
        coordinate_window.title(f"第 {i + 1} 個檔案座標")
        coordinate_window.geometry("300x150")
        coordinate_window.grab_set()  # 設置為模態視窗

        x_var = tk.StringVar()

        tk.Label(coordinate_window, text="X 座標：").grid(row=0, column=0, padx=10, pady=10)
        x_entry = tk.Entry(coordinate_window, textvariable=x_var)
        x_entry.grid(row=0, column=1, padx=10, pady=10)

        def save_coordinates():
            try:
                x = float(x_var.get())
                file_path_list.append(file_path)
                file_distance_list.append(x)
                coordinate_window.destroy()
            except ValueError:
                messagebox.showerror("錯誤", "請輸入有效的座標值")

        tk.Button(coordinate_window, text="確定", command=save_coordinates).grid(row=2, columnspan=2, pady=10)

        # 等待座標視窗關閉
        root.wait_window(coordinate_window)

    # 輸入合併厚度
    combine_thickness = simpledialog.askinteger("合併厚度", "請輸入合併厚度：", parent=root)
    if not combine_thickness or combine_thickness <= 0:
        messagebox.showerror("錯誤", "無效的合併厚度。程式結束。")
        return None, None, None

    # 顯示選取結果
    file_data = list(zip(file_path_list, file_distance_list))
    print(file_data)

    # 回傳選取檔案路徑及座標
    # 把file_path_list 轉成json路徑
    with open('file_path_list.json', 'w') as f:
        json.dump(file_path_list, f)

    return file_path_list, file_distance_list, combine_thickness


def read_files_and_coordinates(file_path_list, combine_thickness):
    if not file_path_list or combine_thickness is None:
        print("未提供檔案路徑或合併厚度。")
        return

    combined_results = {}  # 使用字典儲存，鍵為檔案名稱，值為合併結果

    # 讀取選取的檔案路徑
    for file_path in file_path_list:
        try:
            # 讀取檔案
            df = pd.read_excel(file_path)

            if 'Soil Type' not in df.columns:
                print(f"檔案 {file_path} 不包含 'Soil Type' 欄位，跳過。")
                continue

            soil_type = df['Soil Type']
            print(f"合併厚度: {combine_thickness}")
            half_range = combine_thickness // 2  # 整數除法
            print(f"半厚度範圍: {half_range}")

            # 每 range 筆資料合併
            combined_soil_types = []
            for i in range(0, len(soil_type), half_range):
                data_to_combine = soil_type[i:i + half_range]
                if not data_to_combine.empty:
                    mode_result = data_to_combine.mode()
                    if not mode_result.empty:
                        data_to_combine_mode = mode_result.iloc[0]
                        combined_soil_types.append(data_to_combine_mode)
                    else:
                        combined_soil_types.append(None)  # 處理無眾數情況
                else:
                    combined_soil_types.append(None)  # 處理空資料範圍

            print(f"合併結果: {combined_soil_types}")
            combined_results[file_path] = combined_soil_types

        except Exception as e:
            print(f"讀取檔案 {file_path} 時發生錯誤: {e}")

    # 將合併結果轉為 DataFrame，欄為檔案名稱，列為位置
    max_length = max(len(row) for row in combined_results.values()) if combined_results else 0
    combined_df = pd.DataFrame(
        {f"檔案 {i+1}": value + [None] * (max_length - len(value)) for i, value in enumerate(combined_results.values())}
    )

    # 在 DataFrame 頂部插入距離作為新行
    distance_row = pd.DataFrame([file_distance_list], columns=combined_df.columns)
    combined_df = pd.concat([distance_row, combined_df], ignore_index=True)

    # 把combined_df裡的NAN值替換成0
    combined_df.fillna(0, inplace=True)
    
    # 儲存合併結果為csv檔案
    combined_df.to_csv("合併結果.csv", index=False)

    # 儲存合併結果為 JSON 檔案
    combined_df.to_json("合併結果.json")


    return combined_df







if __name__ == "__main__":
    paths, distances, thickness = select_files_and_set_coordinates()
    if paths and distances and thickness:
        combined_df = read_files_and_coordinates(paths, thickness)
        print("\n保存結果 DataFrame:")
        print(combined_df)



