import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

times = 0

# 創建輸入 times 的窗口
root = tk.Tk()
root.title("輸入 times")
entry = tk.Entry(root)
entry.pack()
entry.focus_set()

# 確認輸入
def on_button_click():
    global times
    try:
        times = int(entry.get())
        root.destroy()
    except ValueError:
        messagebox.showerror("錯誤", "請輸入有效的整數！")

button = tk.Button(root, text="確認", command=on_button_click)
button.pack()
root.mainloop()

# 檢查 times 是否為有效數字
if times <= 0:
    print("請輸入大於 0 的數字")
else:
    # 創建一個空的 DataFrame 來存儲最終結果
    result = pd.DataFrame()

    # 根據輸入的 times，重複選擇和處理文件
    for i in range(times):
        # 創建文件選擇窗口
        root = tk.Tk()
        root.withdraw()
        # 選擇文件
        file_name = filedialog.askopenfilename(title="選擇 Excel 文件", filetypes=[("Excel files", "*.xlsx")])

        if file_name:
            # 加載 Excel 文件
            try:
                data = pd.read_excel(file_name)

                # 四舍五入 Upper Depth 和 Lower Depth 到個位數
                data["Upper Depth"] = data["Upper Depth"].round(0).astype(int)
                data["Lower Depth"] = data["Lower Depth"].round(0).astype(int)

                # 創建新數據結構並擴展至新行
                new_data = []
                for index, row in data.iterrows():
                    count = int(row["Lower Depth"] - row["Upper Depth"] + 1)  # 確保整數
                    new_data.extend([row["Type"]] * count)

                # 新增標題
                new_data = pd.DataFrame(new_data, columns=["Type"f"_{i+1}"])
                

            except Exception as e:
                messagebox.showerror("錯誤", f"無法讀取檔案 {file_name}: {e}")
                continue

    # 把結果寫入新的 Excel 文件
    try:
        result.to_excel("處理結果.xlsx", index=False, header=False)  # 不輸出 header
        print("處理完成，檔案已保存！")
    except Exception as e:
        messagebox.showerror("錯誤", f"無法保存結果檔案: {e}")
