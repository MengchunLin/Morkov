import tkinter as tk
from tkinter import filedialog
import pandas as pd

# 創建選擇文件的窗口
root = tk.Tk()
root.withdraw()
# 選擇文件
file_name = filedialog.askopenfilename(title="選擇Excel文件", filetypes=[("Excel files", "*.xlsx")])

if file_name:
    # 加載Excel文件
    data = pd.read_excel(file_name)
    
    # 四舍五入 Upper Depth 和 Lower Depth 到個位數
    data["Upper Depth"] = data["Upper Depth"].round(0).astype(int)
    data["Lower Depth"] = data["Lower Depth"].round(0).astype(int)

    # 創建新數據結構
    new_data = []

    for index, row in data.iterrows():
        count = int(row["Lower Depth"] - row["Upper Depth"] + 1)  # 確保整數
        new_data.extend([row["Type"]] * count)

# 將結果保存為新檔案
result = pd.DataFrame({"Type": new_data})
result.to_excel("新的檔案路徑.xlsx", index=False)
print("處理完成，檔案已保存！")