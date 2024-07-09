import numpy as np

# 創建一個二維的NumPy矩陣
matrix = np.array([
    [1, 2, 3],
    [2, 3, 1],
    [1, 2, 1]
])

# 使用一個字典來統計出現次數
count_map = {}

# 迭代矩陣中的每個元素
for row in matrix:
    for item in row:
        if item in count_map:
            count_map[item] += 1
        else:
            count_map[item] = 1

print("每個元素的出現次數:")
print(count_map)