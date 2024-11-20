import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import pandas as pd

file_path_list = []  # 用於存儲檔案名稱
file_distance_list = []  # 用於存儲檔案座標
markov_matrix = []  # 用於存儲檔案的markov matrix
combine_thickness = []  # 用於存儲檔案的厚度

def select_files_and_set_coordinates():
    root = tk.Tk()
    root.withdraw()  # 隱藏主視窗

    # 提示使用者輸入需要選擇檔案的次數
    num_files = simpledialog.askinteger("檔案選擇次數", "請輸入需要選擇檔案的次數：", parent=root)
    if not num_files or num_files <= 0:
        messagebox.showerror("錯誤", "無效的檔案數量。程式結束。")
        return

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

        # 用於控制迴圈的標誌
        coordinates_entered = False

        def save_coordinates():
            nonlocal coordinates_entered
            try:
                x = float(x_var.get())
                file_path_list.append(file_path)
                file_distance_list.append(x)
                coordinates_entered = True
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
        return
    
    # 顯示選取結果
    file_data = list(zip(file_path_list, file_distance_list))
    print(file_data)

    # markov_matrix第一行為位置
    markov_matrix.append(file_distance_list)

    # 回傳選取檔案路徑及座標
    return file_path_list, file_distance_list, combine_thickness

# 讀取選取的檔案路徑及座標
def read_files_and_coordinates():
    # 讀取選取的檔案路徑
    for file_path in file_path_list:
        # 讀取檔案
        df = pd.read_excel(file_path)
        # 處理檔案
        Soil_type = df['Soil Type']
        range = combine_thickness/2
        print(range)
        # 每range筆資料合併
        for i in range(len(Soil_type)):
            data_to_combime = Soil_type[i:i+range]
            #統計data_to_combime內出現最多次的值
            data_to_combime_mode = data_to_combime.mode()
            print(data_to_combime_mode)
            
            
        

        



if __name__ == "__main__":
    select_files_and_set_coordinates()
    read_files_and_coordinates()