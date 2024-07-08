import numpy as np

# 原始陣列
array = np.array([0, 112, 202, 242, 264, 303, 307, 424,1,2])

# 將陣列排序
sorted_indices = np.argsort(array)
print("排序後的索引：", sorted_indices)
sorted_array = array[sorted_indices]

# 為排序後的陣列分配編號
sorted_with_indices = [(i, val) for i, val in enumerate(sorted_array)]

print("排序後的陣列和編號：")
for idx, val in sorted_with_indices:
    print(f"編號: {idx + 1}, 值: {val}")
