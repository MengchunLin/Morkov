import numpy as np
import operator as op
import matplotlib.pyplot as plt

Matrix4D = "test.csv"
denominator = 0
molecular = 0
HoleLocation_entire = []

# Define model parameters
interval = 0.5
W = int(70 / interval)  # 140
D = int(25 / interval)  # 50
A = W * D

def load_data(file):
    entire_file = np.loadtxt(file, delimiter=",", skiprows=1)
    geo_matrix = entire_file[1:, :]  # skip first column
    verify_matrix = np.delete(geo_matrix, 3, 1)
    test_hole = geo_matrix[:, 2]
    distance = entire_file[0]
    return geo_matrix, verify_matrix, test_hole, distance

def calculate_hole_locations(distance):
    locations = [1]
    for d in distance:
        if d == 1:
            continue
        else:
            location = int(d / interval)
            locations.append(location)
    return locations

def initialize_group_numbers():
    group_number = np.zeros(A)
    for i in range(1, 74):
        group_number[i - 1] = 1
    for i in range(74, 140):
        group_number[i - 1] = 2
    return group_number

def populate_group_numbers(verify_matrix, group_number, hole_locations):
    Hole = verify_matrix.shape[1]
    for i in range(1, D + 1):
        for j in range(Hole):
            group_number[(hole_locations[j] - 1) + (i - 1) * W] = verify_matrix[i - 1][j]
    return group_number

def calculate_transition_matrices(verify_matrix, soiltype_V):
    VPCM = np.zeros([len(soiltype_V), len(soiltype_V)])
    Tmatrix_V = np.zeros([len(soiltype_V), len(soiltype_V)])
    
    T_t_V = np.zeros(len(verify_matrix))
    for i in range(np.size(verify_matrix, 1)):
        for j in range(len(verify_matrix)):
            T_t_V[j] = verify_matrix[j][i]
        
        for k in range(len(T_t_V) - 1):
            for m in range(len(soiltype_V)):
                for n in range(len(soiltype_V)):
                    if T_t_V[k] == soiltype_V[m][0] and T_t_V[k + 1] == soiltype_V[n][0]:
                        VPCM[m][n] += 1
                        Tmatrix_V[m][n] += 1
    return VPCM, Tmatrix_V

def normalize_transition_matrix(Tmatrix):
    count = np.sum(Tmatrix, axis=1)
    for i in range(np.size(Tmatrix, 1)):
        for j in range(np.size(Tmatrix, 1)):
            Tmatrix[i][j] = Tmatrix[i][j] / count[i]
    return Tmatrix

def calculate_weighted_transition_matrices(VPCM, K=9.3):
    HPCM = np.zeros(VPCM.shape)
    Tmatrix_H = np.zeros(VPCM.shape)
    
    for i in range(np.size(Tmatrix_H, 1)):
        for j in range(np.size(Tmatrix_H, 1)):
            if i == j:
                HPCM[i][j] = K * VPCM[i][j]
                Tmatrix_H[i][j] = K * VPCM[i][j]
            else:
                HPCM[i][j] = VPCM[i][j]
                Tmatrix_H[i][j] = VPCM[i][j]
    return HPCM, Tmatrix_H

def predict_geological_types(W, D, HoleLocation, group_number, Tmatrix_V, Tmatrix_H, typenumber):
    conditions = {}
    for j in range(1, len(HoleLocation)):
        conditions[(HoleLocation[j - 1], HoleLocation[j])] = HoleLocation[j]
    
    current_matrix = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
    transitionName = np.array([[1, 2, 3, 4, 5]])
    a = 0

    for layer in range(2, D + 1):
        for i in range(1, W + 1):
            L_state = M_state = Q_state = Nx = 0

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
                Nx_TH = Tmatrix_H
                f_sum = k_sum = 0
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
    
    group_matrix = group_number.reshape(D, W)
    return group_matrix

def calculate_correct_rate(test_hole, verify_array):
    denominator = molecular = 0
    for i, x in zip(test_hole, verify_array):
        if i == x:
            denominator += 1
            molecular += 1
        else:
            denominator += 1
    return molecular / denominator

def visualize_geological_types(group_matrix):
    plt.imshow(group_matrix, cmap='tab10', origin='upper')
    plt.colorbar(label='Geological Type')
    plt.title('Geological Type Prediction')
    plt.xlabel('Width (units)')
    plt.ylabel('Depth')
    plt.show()

# Main Execution
geo_matrix, verify_matrix, test_hole, distance = load_data(Matrix4D)
HoleLocation_entire = calculate_hole_locations(distance)
print('孔洞位置:', HoleLocation_entire)

group_number = initialize_group_numbers()
group_number = populate_group_numbers(verify_matrix, group_number, HoleLocation_entire)

unique_numbers = np.unique(geo_matrix)
typenumber = len(unique_numbers)
print('共有', typenumber, '種土讓材質')

soiltype_V = {k: 0 for k in unique_numbers}
T_t_V = np.zeros(len(verify_matrix))

for i in range(np.size(verify_matrix, 1)):
    for j in range(len(verify_matrix)):
        T_t_V[j] = verify_matrix[j][i]
    for k in T_t_V:
        soiltype_V[k] = soiltype_V.get(k, 0) + 1

soiltype_V = sorted(soiltype_V.items(), key=op.itemgetter(0), reverse=False)
VPCM, Tmatrix_V = calculate_transition_matrices(verify_matrix, soiltype_V)
Tmatrix_V = normalize_transition_matrix(Tmatrix_V)
HPCM, Tmatrix_H = calculate_weighted_transition_matrices(VPCM)
Tmatrix_H = normalize_transition_matrix(Tmatrix_H)

group_matrix = predict_geological_types(W, D, HoleLocation_entire, group_number, Tmatrix_V, Tmatrix_H, typenumber)
verify_array = group_matrix[:, 91]
correct_rate = calculate_correct_rate(test_hole, verify_array)
print('正確率:', correct_rate)

visualize_geological_types(group_matrix)
